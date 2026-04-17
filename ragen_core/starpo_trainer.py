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
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .trajectory_buffer import TrajectoryBuffer
from configs.constants import CKPT_DIR
from utils.logger import logger
from utils.tracker import TrainingTracker
from evaluation.metrics import EvaluatorMetrics, compute_reward_variance, check_echo_trap_signs


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

        # ---- Echo Trap 检测用的滑动指标 ----
        self.history_reward_var: List[float] = []
        self.history_entropy: List[float] = []

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
        pattern = r'<think>.*?</think>\s*<answer>.*?</answer>'
        return bool(re.search(pattern, response, re.DOTALL))

    # ----------------------------------------------------------------
    # 2. Rollout collection
    # ----------------------------------------------------------------

    def _rollout_one_trajectory(self, seed: Optional[int]) -> List[Dict[str, Any]]:
        """对给定 seed 采样一条完整 trajectory。"""
        obs, info = self.env.reset(seed=seed)
        system_prompt = self.agent.config.system_prompt

        # RAGEN 风格：第一轮 user message 里夹带环境玩法说明（含多动作序列示例）。
        # 后续 turn 不再重复注入，避免 context 累积过快。
        env_instruction = self.env.get_env_instruction()
        first_user = ""
        if env_instruction:
            first_user = env_instruction.rstrip() + "\n\n"
        first_user += f"{obs}\n{self.env.get_valid_actions()}\nPlease reason step by step."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user},
        ]

        trajectory: List[Dict[str, Any]] = []
        terminated, truncated = False, False
        turn_idx = 0  # 已完成的 LLM turn 数

        while not (terminated or truncated):
            # ---- Turn-level 硬截断（RAGEN `agent_proxy.max_turn` 对齐） ----
            # 在进入下一次 chat_request 之前判定：如果已经用满 max_turn，
            # 就把上一条 entry 标注为 truncated-by-agent-budget 并退出循环。
            if 0 < self.max_turn <= turn_idx:
                if trajectory:
                    trajectory[-1]["truncated"] = True
                    last_info = trajectory[-1].get("info") or {}
                    last_info["truncated_reason"] = "max_turn_reached"
                    last_info["max_turn"] = self.max_turn
                    trajectory[-1]["info"] = last_info
                truncated = True
                break

            response = self.agent.chat_request(messages)

            step_penalty = 0.0
            if self.use_format_reward and not self._check_format(response):
                step_penalty = self.format_penalty

            next_obs, reward, terminated, truncated, info = self.env.step(response)
            total_reward = reward + step_penalty

            trajectory.append({
                "obs": obs,
                "messages": list(messages),
                "response": response,
                "reward": total_reward,
                "env_reward": reward,
                "format_penalty": step_penalty,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
                "turn_idx": turn_idx,
            })
            turn_idx += 1

            obs = next_obs
            messages.append({"role": "assistant", "content": response})
            if not (terminated or truncated):
                messages.append({
                    "role": "user",
                    "content": f"Observation: {obs}\nReward for last step: {reward}\nNext action?",
                })

        return trajectory

    def collect_rollouts(self, prompt_states: List[Dict[str, Any]]) -> None:
        """对每个 prompt 采 num_rollouts 条轨迹放入 buffer。"""
        for state_info in prompt_states:
            seed = state_info.get("seed") if isinstance(state_info, dict) else None
            from tqdm import tqdm
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

        # 采完的原始 reward 统计（过滤前）
        raw_returns = self.buffer.compute_returns()
        raw_reward_mean = float(np.mean(raw_returns)) if raw_returns else 0.0
        raw_reward_var = float(np.var(raw_returns)) if raw_returns else 0.0

        # (b) 方差过滤（StarPO-S 稳定化）
        original_size = len(self.buffer.trajectories)
        self.buffer.filter_by_variance(
            group_size=self.num_rollouts,
            retain_ratio=self.variance_filter_ratio,
        )
        filtered_size = len(self.buffer.trajectories)
        logger.info(f"[StarPO step={step_idx}] variance filter: {original_size} -> {filtered_size}")

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

        # (d) 汇总指标
        # Echo Trap 检测
        self.history_reward_var.append(raw_reward_var)
        if "entropy" in metrics:
            self.history_entropy.append(float(metrics["entropy"]))
        echo_trap = check_echo_trap_signs(self.history_reward_var, self.history_entropy) \
            if len(self.history_entropy) >= 5 else False

        merged = {
            "train/raw_reward_mean": raw_reward_mean,
            "train/raw_reward_var": raw_reward_var,
            "train/num_trajectories_before_filter": original_size,
            "train/num_trajectories_after_filter": filtered_size,
            "train/filter_retain_ratio": (filtered_size / max(1, original_size)),
            "train/echo_trap_sign": int(echo_trap),
            "timing/rollout_sec": t_rollout,
            "timing/update_sec": t_update,
        }
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                merged[f"train/{k}"] = v
            else:
                merged[f"train/{k}"] = str(v)
        return merged

    # ----------------------------------------------------------------
    # 4. 验证
    # ----------------------------------------------------------------

    def evaluate(self, step_idx: int) -> Dict[str, float]:
        """
        在 eval_episodes 条 episode 上跑在线推理，记录 avg_reward / success_rate / avg_length。
        验证时用固定低温采样 (尊重 agent 现有温度配置，不做修改)。
        """
        logger.info(f"[StarPO step={step_idx}] evaluate on {self.eval_episodes} episodes")
        em = EvaluatorMetrics()
        eval_rewards: List[float] = []
        for ep in range(self.eval_episodes):
            seed = random.randint(0, 2**31 - 1)
            traj = self._rollout_one_trajectory(seed)
            total_reward = sum(step["env_reward"] for step in traj)
            # success: env 在 info 里提供 is_success；否则以 env_reward>0.5 作为近似
            last_info = traj[-1]["info"] if traj else {}
            success = last_info.get("is_success", total_reward > 0.5) if isinstance(last_info, dict) else (total_reward > 0.5)
            em.add_episode(reward=total_reward, success=bool(success), length=len(traj))
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
