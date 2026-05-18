"""
纯 RL 训练器 / Pure RL Trainer
-----------------------------------
作为 StarPOTrainer 的 baseline：与 StarPOTrainer 保持相同的循环结构和日志接口，
但**关掉**两个 RAGEN 特性：
- Format reward / penalty
- Variance-based rollout filtering

这样它能公平地对比 RAGEN 的稳定化改进带来的收益。
"""

import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .rollout_utils import (
    batched_rollout_for_prompt,
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


class PureRLTrainer:
    """纯 RL 训练器，接口与 StarPOTrainer 一致，去掉 RAGEN 特有稳定化手段。"""

    def __init__(self, config: Any, env: Any, agent: Any, rl_algo: Any):
        self.config = config
        self.env = env
        self.agent = agent
        self.rl_algo = rl_algo

        self.buffer = TrajectoryBuffer()

        # 直接属性访问；默认值仅来自 scripts/train.py
        rcfg = config.ragen_config
        self.num_rollouts = rcfg.num_rollouts
        self.prompt_batch_size = rcfg.prompt_batch_size
        # 与 StarPOTrainer 对齐：turn 级预算（RAGEN `agent_proxy.max_turn`）。
        self.max_turn = rcfg.max_turn

        self.total_training_steps = config.total_training_steps
        self.eval_interval = config.eval_interval
        self.eval_episodes = config.eval_episodes
        self.save_interval = config.save_interval

        self.tracker = TrainingTracker(
            exp_name=config.exp_name + "_pureRL",
            use_wandb=False,
        )

        logger.info(
            f"Initialized PureRLTrainer (no format reward / no variance filter) | "
            f"total_training_steps={self.total_training_steps} num_rollouts={self.num_rollouts} "
            f"max_turn={self.max_turn}"
        )

    # ----------------------------------------------------------------
    # Rollout / train / evaluate（与 StarPO 对齐，只去掉稳定化特性）
    # ----------------------------------------------------------------

    def _rollout_one_trajectory(self, seed: Optional[int]) -> List[Dict[str, Any]]:
        """对给定 seed 采样一条完整 trajectory（委托给公共 rollout 工具，禁用 format reward）。
        system prompt 由 env.agent_system_prompt 提供，rollout_utils 内部直接读。"""
        return rollout_one_trajectory(
            env=self.env,
            agent=self.agent,
            seed=seed,
            max_turn=self.max_turn,
            use_format_reward=False,  # PureRL baseline: 不加格式惩罚
            format_penalty=0.0,
        )

    # 与 StarPOTrainer 对齐：默认走 batched rollout，详细语义见
    # batched_rollout_for_prompt docstring。出意外时一行切回 False 即可。
    _USE_BATCHED_ROLLOUT: bool = True

    def collect_rollouts(self, prompt_states: List[Dict[str, Any]]) -> None:
        if self._USE_BATCHED_ROLLOUT:
            self._collect_rollouts_batched(prompt_states)
        else:
            self._collect_rollouts_sequential(prompt_states)

    def _collect_rollouts_batched(self, prompt_states: List[Dict[str, Any]]) -> None:
        env_cfg = self.config.env_config
        for state_info in prompt_states:
            seed = state_info.get("seed") if isinstance(state_info, dict) else None
            envs = [make_env(env_cfg) for _ in range(self.num_rollouts)]
            try:
                trajs = batched_rollout_for_prompt(
                    envs=envs,
                    agent=self.agent,
                    seed=seed,
                    max_turn=self.max_turn,
                    use_format_reward=False,  # PureRL baseline: 不加格式惩罚
                    format_penalty=0.0,
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
        for state_info in prompt_states:
            seed = state_info.get("seed") if isinstance(state_info, dict) else None
            for _ in range(self.num_rollouts):
                traj = self._rollout_one_trajectory(seed)
                if len(traj) > 0:
                    self.buffer.add_trajectory(traj)

    def train_iteration(self, step_idx: int, prompt_states: List[Dict[str, Any]]) -> Dict[str, float]:
        self.buffer.clear()
        t0 = time.time()
        logger.info(f"[PureRL step={step_idx}] rolling out {len(prompt_states)} prompts × {self.num_rollouts}")
        self.collect_rollouts(prompt_states)
        t_rollout = time.time() - t0

        # 与 StarPOTrainer 保持完全一致的指标 schema（便于事后画图对比 PureRL vs StarPO）：
        #   raw_reward_mean       对齐论文 Figure 6 ① Average Reward
        #   in_group_reward_std   对齐论文 Figure 6 ② In-Group Reward Std
        #   raw_reward_var        cross-prompt 全体方差（supplemental）
        raw_returns = self.buffer.compute_returns()
        raw_reward_mean = float(np.mean(raw_returns)) if raw_returns else 0.0
        raw_reward_var = float(np.var(raw_returns)) if raw_returns else 0.0
        in_group_reward_std = compute_in_group_reward_std(raw_returns, self.num_rollouts)

        metrics: Dict[str, Any] = {}
        batch_data = self.buffer.get_all_data()
        if len(batch_data) > 0:
            t1 = time.time()
            metrics = self.rl_algo.train_step(batch_data)
            t_update = time.time() - t1
        else:
            logger.warning("[PureRL] No trajectories collected, skipping update.")
            t_update = 0.0

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
        return merged

    def evaluate(self, step_idx: int) -> Dict[str, float]:
        """与 StarPOTrainer.evaluate 同源 batched 路径：N 个独立 env + N 个独立 seed 一次性 batch。
        最后一个 batch 在 eval_episodes 不被 num_rollouts 整除时自然变小。"""
        logger.info(f"[PureRL step={step_idx}] evaluate on {self.eval_episodes} episodes")
        em = EvaluatorMetrics()
        eval_rewards: List[float] = []

        if self._USE_BATCHED_ROLLOUT:
            env_cfg = self.config.env_config
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
                        # PureRL baseline：评估同样不叠 format penalty，跟训练侧保持一致。
                        use_format_reward=False,
                        format_penalty=0.0,
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
            for _ in range(self.eval_episodes):
                seed = random.randint(0, 2**31 - 1)
                traj = self._rollout_one_trajectory(seed)
                if not traj:
                    continue
                total_reward = float(sum(step["env_reward"] for step in traj))
                success = judge_success(traj)
                em.add_episode_from_trajectory(traj, success=success)
                eval_rewards.append(total_reward)

        summary = em.summary()
        summary["eval/reward_variance"] = compute_reward_variance(eval_rewards)
        return summary

    def run(self) -> None:
        logger.info(
            f"Starting PureRL training | steps={self.total_training_steps} eval_every={self.eval_interval}"
        )
        try:
            for step in range(1, self.total_training_steps + 1):
                prompt_states = [
                    {"seed": random.randint(0, 2**31 - 1)}
                    for _ in range(self.prompt_batch_size)
                ]
                metrics = self.train_iteration(step, prompt_states)
                self.tracker.log(metrics, step=step)

                if self.eval_interval > 0 and step % self.eval_interval == 0:
                    eval_metrics = self.evaluate(step)
                    self.tracker.log(eval_metrics, step=step)

                if self.save_interval > 0 and step % self.save_interval == 0:
                    ckpt_path = os.path.join(CKPT_DIR, f"{self.config.exp_name}_pureRL_step{step}")
                    try:
                        self.rl_algo.save(ckpt_path)
                    except Exception as e:
                        logger.warning(f"[PureRL] Failed to save checkpoint at step {step}: {e}")

            final_eval = self.evaluate(self.total_training_steps)
            self.tracker.log(final_eval, step=self.total_training_steps)
            logger.info(f"Training finished. Final eval metrics: {final_eval}")
        finally:
            self.tracker.close()
