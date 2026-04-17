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

from .trajectory_buffer import TrajectoryBuffer
from configs.constants import CKPT_DIR
from utils.logger import logger
from utils.tracker import TrainingTracker
from evaluation.metrics import EvaluatorMetrics, compute_reward_variance


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
        obs, info = self.env.reset(seed=seed)
        system_prompt = self.agent.config.system_prompt

        # 与 StarPOTrainer 保持一致：第一轮 user message 注入环境玩法说明。
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
        turn_idx = 0

        while not (terminated or truncated):
            # 与 StarPOTrainer 对齐的 turn-level 硬截断
            if self.max_turn > 0 and turn_idx >= self.max_turn:
                if trajectory:
                    trajectory[-1]["truncated"] = True
                    last_info = trajectory[-1].get("info") or {}
                    last_info["truncated_reason"] = "max_turn_reached"
                    last_info["max_turn"] = self.max_turn
                    trajectory[-1]["info"] = last_info
                truncated = True
                break

            response = self.agent.chat_request(messages)
            next_obs, reward, terminated, truncated, info = self.env.step(response)

            trajectory.append({
                "obs": obs,
                "messages": list(messages),
                "response": response,
                "reward": reward,  # 不加 format penalty
                "env_reward": reward,
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

        raw_returns = self.buffer.compute_returns()
        raw_reward_mean = float(np.mean(raw_returns)) if raw_returns else 0.0
        raw_reward_var = float(np.var(raw_returns)) if raw_returns else 0.0

        # 不做过滤
        n_traj = len(self.buffer.trajectories)

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
            "train/num_trajectories": n_traj,
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
        logger.info(f"[PureRL step={step_idx}] evaluate on {self.eval_episodes} episodes")
        em = EvaluatorMetrics()
        eval_rewards: List[float] = []
        for _ in range(self.eval_episodes):
            seed = random.randint(0, 2**31 - 1)
            traj = self._rollout_one_trajectory(seed)
            total_reward = sum(step["env_reward"] for step in traj)
            last_info = traj[-1]["info"] if traj else {}
            success = last_info.get("is_success", total_reward > 0.5) if isinstance(last_info, dict) else (total_reward > 0.5)
            em.add_episode(reward=total_reward, success=bool(success), length=len(traj))
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
