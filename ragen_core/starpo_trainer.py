"""
StarPO 训练器 / StarPO Trainer
-----------------------------------
RAGEN 论文核心框架：State-Thinking-Actions-Reward Policy Optimization。

在"纯 RL + 多轮环境"之上叠加了两个稳定化改进：
1. **Format Reward Penalty**：要求输出符合 `<think>...</think><answer>...</answer>` 格式，否则加负奖励。
2. **Variance-based Rollout Filtering**：同一 prompt 下采样 num_rollouts 条轨迹形成一个 group，
   只保留 top-k 高方差组参与训练（缓解 Echo Trap，即模型过早收敛到单一解）。

训练循环语义（重要）：
    for step in range(total_training_steps):
        sample prompt(s) → collect rollouts → (variance filter) → rl_algo.train_step → (optional) evaluate
    eval_interval 控制每多少步做一次验证，eval_episodes 控制每次验证跑多少个 episode。
"""

import os
import random
import time
from typing import Any, Dict, List, Optional
import torch, gc

import numpy as np

from .rollout_utils import (
    batched_rollout_for_prompt,
    check_format,
    judge_success,
    rollout_one_trajectory,
)
from envs import make_env
from .trajectory_buffer import TrajectoryBuffer
from configs.constants import CKPT_DIR
from utils.logger import logger
from utils.tracker import TrainingTracker
from evaluation.metrics import (
    EvaluatorMetrics,
    compute_reward_variance,
    compute_in_group_reward_std,
)
from tqdm import tqdm


class StarPOTrainer:
    """
    StarPO (State-Thinking-Actions-Reward Policy Optimization) 训练器。

    关键配置来源：
        config: ExperimentConfig
            - total_training_steps: 训练步数
            - eval_interval: 每多少步做一次 evaluate
            - eval_episodes: 每次 evaluate 跑多少个 episode
            - save_interval: 每多少步保存一次 checkpoint
            - rl_algo_config, ragen_config, env_config, agent_config

        config.ragen_config:
            - num_rollouts: 每个 prompt 采样的轨迹数 (GRPO group size)
            - use_format_reward / format_penalty: 格式奖励相关
            - variance_filter_ratio: 方差过滤保留比例
            - prompt_batch_size (可选, 默认 1): 一次训练步用多少不同 prompt
    """

    def __init__(self, config: Any, env: Any, agent: Any, rl_algo: Any):
        self.config = config
        self.env = env
        self.agent = agent
        self.rl_algo = rl_algo

        self.buffer = TrajectoryBuffer()

        # ---- RAGEN 超参（直接属性访问，默认值仅来自 scripts/train.py） ----
        rcfg = config.ragen_config
        self.num_rollouts = rcfg.num_rollouts
        self.use_format_reward = rcfg.use_format_reward
        self.format_penalty = rcfg.format_penalty
        self.variance_filter_ratio = rcfg.variance_filter_ratio
        self.prompt_batch_size = rcfg.prompt_batch_size
        # LLM turn 级预算，和 env.max_steps（原子 env step 级预算）是两层独立截断。
        # 任一触发 → truncated=True，episode 结束。RAGEN 主力 FrozenLake/Sokoban = 1。
        self.max_turn = rcfg.max_turn

        # ---- 训练循环超参 ----
        self.total_training_steps = config.total_training_steps
        self.eval_interval = config.eval_interval
        self.eval_episodes = config.eval_episodes
        self.save_interval = config.save_interval

        # ---- 指标追踪器 ----
        self.tracker = TrainingTracker(
            exp_name=config.exp_name,
            use_wandb=False,
        )

        logger.info(
            f"Initialized StarPOTrainer | total_training_steps={self.total_training_steps} "
            f"eval_interval={self.eval_interval} num_rollouts={self.num_rollouts} "
            f"max_turn={self.max_turn} variance_filter_ratio={self.variance_filter_ratio} "
            f"prompt_batch_size={self.prompt_batch_size}"
        )

    # ----------------------------------------------------------------
    # 1. Format check
    # ----------------------------------------------------------------

    @staticmethod
    def _check_format(response: str) -> bool:
        """检查输出是否符合 `<think>...</think>...<answer>...</answer>` 格式。"""
        # 保持原有公共入口，委托给 rollout_utils 里的 check_format 以保证全局口径一致。
        return check_format(response)

    # ----------------------------------------------------------------
    # 2. Rollout collection（委托给 ragen_core.rollout_utils 统一实现）
    # ----------------------------------------------------------------

    def _rollout_one_trajectory(self, seed: Optional[int]) -> List[Dict[str, Any]]:
        """对给定 seed 采样一条完整 trajectory（调用公共 rollout 工具）。
        system prompt 由 env.agent_system_prompt 提供，rollout_utils 内部直接读。"""
        return rollout_one_trajectory(
            env=self.env,
            agent=self.agent,
            seed=seed,
            max_turn=self.max_turn,
            use_format_reward=self.use_format_reward,
            format_penalty=self.format_penalty,
        )

    # ----------------------------------------------------------------
    # Rollout 后端开关：True = 同 prompt 内 R 条 trajectory batch generate（推荐）
    # 数学上严格等价于串行版本（见 batched_rollout_for_prompt docstring），
    # 仅作为"出意外时一行回滚"的逃生通道存在；正式训练应保持 True。
    # ----------------------------------------------------------------
    _USE_BATCHED_ROLLOUT: bool = True

    def collect_rollouts(self, prompt_states: List[Dict[str, Any]]) -> None:
        """对每个 prompt 采 num_rollouts 条轨迹放入 buffer。"""
        if self._USE_BATCHED_ROLLOUT:
            self._collect_rollouts_batched(prompt_states)
        else:
            self._collect_rollouts_sequential(prompt_states)

    def _collect_rollouts_batched(self, prompt_states: List[Dict[str, Any]]) -> None:
        """同 prompt 内 R 条 traj 凑 batch 调 batched_chat_request；prompt 间仍串行。
        每个 prompt 临时构造 num_rollouts 个独立 env 实例，结束后立刻 close。"""
        env_cfg = self.config.env_config
        for state_info in tqdm(prompt_states, desc="Collecting rollouts (batched)"):
            seed = state_info.get("seed") if isinstance(state_info, dict) else None
            envs = [make_env(env_cfg) for _ in range(self.num_rollouts)]
            try:
                trajs = batched_rollout_for_prompt(
                    envs=envs,
                    agent=self.agent,
                    seed=seed,
                    max_turn=self.max_turn,
                    use_format_reward=self.use_format_reward,
                    format_penalty=self.format_penalty,
                )
                for traj in trajs:
                    if len(traj) > 0:
                        self.buffer.add_trajectory(traj)
            finally:
                for env in envs:
                    try:
                        env.close()
                    except Exception:
                        pass

    def _collect_rollouts_sequential(self, prompt_states: List[Dict[str, Any]]) -> None:
        """原始串行实现，作为 _USE_BATCHED_ROLLOUT=False 时的 fallback / 对比 baseline。"""
        for state_info in prompt_states:
            seed = state_info.get("seed") if isinstance(state_info, dict) else None
            for _ in tqdm(range(self.num_rollouts), desc="Collecting rollouts"):
                traj = self._rollout_one_trajectory(seed)
                if len(traj) > 0:
                    self.buffer.add_trajectory(traj)

    # ----------------------------------------------------------------
    # 3. 一次训练步
    # ----------------------------------------------------------------

    def train_iteration(self, step_idx: int, prompt_states: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        一次 StarPO 训练步：Rollout → Variance Filter → RL Update。
        返回训练指标 dict（供 tracker 记录）。
        """
        self.buffer.clear()
        t0 = time.time()

        # (a) 采样
        logger.info(f"[StarPO step={step_idx}] rolling out with {len(prompt_states)} prompts × {self.num_rollouts} rollouts")
        self.collect_rollouts(prompt_states)
        t_rollout = time.time() - t0

        # 采完的原始 reward 统计（过滤前）—— 三类口径：
        #   raw_reward_mean       对齐论文 Figure 6 ① Average Reward
        #   in_group_reward_std   对齐论文 Figure 6 ② In-Group Reward Std（按 prompt 分组算 std 再均值）
        #   raw_reward_var        cross-prompt + within-prompt 混合方差（保留作 supplemental，非论文核心指标）
        raw_returns = self.buffer.compute_returns()
        raw_reward_mean = float(np.mean(raw_returns)) if raw_returns else 0.0
        raw_reward_var = float(np.var(raw_returns)) if raw_returns else 0.0
        in_group_reward_std = compute_in_group_reward_std(raw_returns, self.num_rollouts)

        # (b) 方差过滤（StarPO-S 稳定化）
        size_before = len(self.buffer.trajectories)
        self.buffer.filter_by_variance(
            group_size=self.num_rollouts,
            retain_ratio=self.variance_filter_ratio,
        )
        size_after = len(self.buffer.trajectories)
        # 只在 filter 真正生效（删了至少 1 条 traj）时打 log，避免 baseline filter=1.0 刷屏
        if size_after != size_before:
            logger.info(
                f"[StarPO step={step_idx}] variance filter: {size_before} -> {size_after}"
            )

        # (c) RL 更新
        metrics: Dict[str, Any] = {}
        batch_data = self.buffer.get_all_data()
        if len(batch_data) > 0:
            t1 = time.time()
            metrics = self.rl_algo.train_step(batch_data)
            t_update = time.time() - t1
        else:
            logger.warning("[StarPO] No trajectories left after filter, skipping update.")
            t_update = 0.0

        # (d) 汇总指标 —— 对齐 RAGEN paper Figure 6 的 4 个 collapse indicators：
        #   ① Average Reward      = train/raw_reward_mean
        #   ② In-Group Reward Std = train/in_group_reward_std       ← 早期预警 (early warning)
        #   ③ Gradient Norm       = train/grad_norm / grad_norm_max ← 由 rl_algo.train_step 提供
        #   ④ Entropy Loss        = train/entropy                   ← 由 rl_algo.train_step 提供
        merged = {
            "train/raw_reward_mean": raw_reward_mean,
            "train/raw_reward_var": raw_reward_var,
            "train/in_group_reward_std": in_group_reward_std,
            "timing/rollout_sec": t_rollout,
            "timing/update_sec": t_update,
        }
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                merged[f"train/{k}"] = v
            else:
                merged[f"train/{k}"] = str(v)

        # 每步末尾强制回收 Python & CUDA 内存池：
        # - batch_data 是 buffer.trajectories 的引用，删掉只是释放局部引用（buffer 自身
        #   会在下个 step 开头 clear），不影响 trajectories 生命周期，但至少让 GC 可见。
        # - gc.collect() 强制回收循环引用 / 大对象。
        # - torch.cuda.empty_cache() 把 CUDA caching allocator 里未使用的 bin 归还给
        #   driver，避免 Windows WDDM 把溢出部分搬到"共享 GPU 内存"(扣系统 RAM)。
        del batch_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return merged

    # ----------------------------------------------------------------
    # 4. 验证
    # ----------------------------------------------------------------

    def evaluate(self, step_idx: int) -> Dict[str, float]:
        """
        在 eval_episodes 条 episode 上跑在线推理，记录 avg_reward / success_rate / avg_length。
        验证时用固定低温采样 (尊重 agent 现有温度配置，不做修改)。

        与训练侧 collect_rollouts 共用 batched 路径：每个 batch 包含 ``num_rollouts`` 个独立 env，
        每个 env 一个独立 seed（跟训练时同 prompt 共用 seed 的语义不同，所以 seed 以 list 形式传）。
        最后一个 batch 在 eval_episodes 不被 num_rollouts 整除时自然变小，
        ``batched_rollout_for_prompt`` 内部按 ``len(envs)`` 跑，无需特殊处理。
        """
        logger.info(f"[StarPO step={step_idx}] evaluate on {self.eval_episodes} episodes")
        em = EvaluatorMetrics()
        eval_rewards: List[float] = []

        if self._USE_BATCHED_ROLLOUT:
            env_cfg = self.config.env_config
            # 复用训练侧的 R 作为 eval batch size：显存足够跑 R 条 traj 的 batch generate，
            # 评估期同样的 batch 量级即可；不需要单独引入 eval_batch_size 配置。
            B = max(1, int(self.num_rollouts))
            n_eval = int(self.eval_episodes)
            for start in tqdm(range(0, n_eval, B), desc="Evaluating (batched)..."):
                actual_b = min(B, n_eval - start)
                seeds = [random.randint(0, 2**31 - 1) for _ in range(actual_b)]
                envs = [make_env(env_cfg) for _ in range(actual_b)]
                try:
                    trajs = batched_rollout_for_prompt(
                        envs=envs,
                        agent=self.agent,
                        seed=seeds,
                        max_turn=self.max_turn,
                        use_format_reward=self.use_format_reward,
                        format_penalty=self.format_penalty,
                    )
                    for traj in trajs:
                        if not traj:
                            continue
                        total_reward = float(sum(step["env_reward"] for step in traj))
                        success = judge_success(traj)
                        em.add_episode_from_trajectory(traj, success=success)
                        eval_rewards.append(total_reward)
                finally:
                    for env in envs:
                        try:
                            env.close()
                        except Exception:
                            pass
        else:
            for _ in tqdm(range(self.eval_episodes), desc="Evaluating..."):
                seed = random.randint(0, 2**31 - 1)
                traj = self._rollout_one_trajectory(seed)
                if not traj:
                    continue
                total_reward = float(sum(step["env_reward"] for step in traj))
                # 严格 RAGEN 口径：terminated and not truncated；
                # env 在 info 里显式提供 is_success/success 时以 info 为准（见 judge_success）。
                success = judge_success(traj)
                em.add_episode_from_trajectory(traj, success=success)
                eval_rewards.append(total_reward)

        summary = em.summary()
        summary["eval/reward_variance"] = compute_reward_variance(eval_rewards)
        return summary

    # ----------------------------------------------------------------
    # 5. 主循环
    # ----------------------------------------------------------------

    def run(self) -> None:
        """训练主循环：每一步 = 一次 RL 更新；eval_interval 控制验证频率。"""
        logger.info(
            f"Starting StarPO training | steps={self.total_training_steps} eval_every={self.eval_interval} "
            f"save_every={self.save_interval}"
        )

        try:
            for step in range(1, self.total_training_steps + 1):
                prompt_states = [
                    {"seed": random.randint(0, 2**31 - 1)}
                    for _ in range(self.prompt_batch_size)
                ]
                metrics = self.train_iteration(step, prompt_states)
                self.tracker.log(metrics, step=step)

                # 评估
                if self.eval_interval > 0 and step % self.eval_interval == 0:
                    eval_metrics = self.evaluate(step)
                    self.tracker.log(eval_metrics, step=step)

                # 保存 checkpoint
                if self.save_interval > 0 and step % self.save_interval == 0:
                    ckpt_path = os.path.join(CKPT_DIR, f"{self.config.exp_name}_step{step}")
                    try:
                        self.rl_algo.save(ckpt_path)
                    except Exception as e:
                        logger.warning(f"[StarPO] Failed to save checkpoint at step {step}: {e}")

            # 训练结束：最后再评估一次
            final_eval = self.evaluate(self.total_training_steps)
            self.tracker.log(final_eval, step=self.total_training_steps)
            logger.info(f"Training finished. Final eval metrics: {final_eval}")
        finally:
            self.tracker.close()
