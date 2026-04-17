"""
GRPO (Group Relative Policy Optimization) 算法 / GRPO Algorithm
-----------------------------------
DeepSeek 系论文提出的 actor-only PPO 变体，RAGEN 默认使用。

核心差异点（与 PPO 相比）：
1. **无 Critic**：直接用"组相对 reward 归一化"作为每条样本的 advantage（标量）。
2. **无 GAE**：不估计 TD 误差，把 trajectory 总 reward 的 z-score 作为整条 response 上所有 token 的 advantage。
3. 组定义：同一个 "prompt" 下采样的 N 条轨迹称为一个 group，归一化在组内完成。
   由于我们 StarPO 的 rollout 里每个 step 都用同一个初始 env 做多次采样，不同 trajectory 共享同一个初始
   prompt，因此自然构成一个组。对应到训练，每次 collect_rollouts 返回的 trajectories 整体就是一个 group。

数据处理复用 PPO 的 `trajectory_utils`，保证和 PPO 完全一致的 tokenize / collate / 前向。
"""

import os
import copy
import random
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .base_algo import BaseRLAlgo
from .optimizer_utils import build_optimizer
from .trajectory_utils import tokenize_trajectory, collate_fn, forward_logprobs_and_entropy
from utils.logger import logger


class GRPO(BaseRLAlgo):
    """
    GRPO 算法实现。

    训练接口：
        `train_step(batch_data: List[List[Dict]])`。
        假设 batch_data 中所有 trajectory 同属一个 group，用于组相对 advantage 归一化。
    """

    def __init__(self, config: Any, agent: Any):
        super().__init__(config, agent)

        # ---------- 超参（直接属性访问；默认值仅来自 scripts/train.py） ----------
        self.lr = config.learning_rate
        self.ppo_epochs = config.ppo_epochs
        self.mini_batch_size = config.mini_batch_size
        self.clip_ratio = config.clip_ratio
        self.ent_coef = config.ent_coef
        self.kl_coef = config.kl_coef
        self.target_kl = config.target_kl                  # Optional[float]; None 即不启用
        self.max_seq_length = config.max_seq_length
        self.use_ref = config.use_ref
        self.optimizer_name = config.optimizer

        if not self.use_ref and self.kl_coef != 0.0:
            logger.warning(
                f"GRPO: use_ref=False but kl_coef={self.kl_coef} != 0; "
                f"forcing kl_coef=0 because no reference distribution is available."
            )
            self.kl_coef = 0.0

        self.device = agent.device
        self.tokenizer = agent.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if not hasattr(self.agent, 'model'):
            logger.warning("Agent has no local 'model' attribute. Training will fail. (Evaluate-only agent?)")
            return

        self.actor = self.agent.model

        self.ref_model: Optional[nn.Module]
        if self.use_ref:
            logger.info("GRPO: creating frozen reference model (for KL) via deepcopy...")
            self.ref_model = copy.deepcopy(self.actor)
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
        else:
            logger.info("GRPO: use_ref=False, skipping ref_model (saves ~1GB VRAM, disables KL anchor).")
            self.ref_model = None

        self.optimizer = build_optimizer(
            name=self.optimizer_name,
            params=list(self.actor.parameters()),
            lr=self.lr,
            actor=self.actor,
        )
        logger.info(
            f"GRPO initialized | optimizer={self.optimizer_name} lr={self.lr} epochs={self.ppo_epochs} "
            f"mini_bs={self.mini_batch_size} use_ref={self.use_ref} kl_coef={self.kl_coef} "
            f"clip={self.clip_ratio} ent_coef={self.ent_coef}"
        )

    # ----------------------------------------------------------------
    # 1. 数据准备 + 组相对 advantage 计算
    # ----------------------------------------------------------------

    def _prepare_data(self, batch_data: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        将 batch trajectories tokenize 成长序列字典列表，并在**组内**对 trajectory 总 reward
        做 z-score 归一化，作为该条 trajectory 上所有 response token 的共享 advantage（标量）。
        """
        data: List[Dict[str, Any]] = []
        traj_scores: List[float] = []
        for traj in batch_data:
            if len(traj) == 0:
                continue
            item = tokenize_trajectory(self.tokenizer, traj, self.max_seq_length)
            if item["loss_mask"].sum().item() == 0:
                continue
            # 组内归一化使用的分数：这条 trajectory 的总 reward
            total_reward = float(item["token_level_rewards"].sum().item())
            item["group_score"] = total_reward
            traj_scores.append(total_reward)
            data.append(item)

        if not data:
            return []

        # 组内 z-score 归一化：(score - mean) / (std + eps)
        scores_tensor = torch.tensor(traj_scores, dtype=torch.float32)
        mean = scores_tensor.mean()
        std = scores_tensor.std(unbiased=False)
        eps = 1e-6
        normalized = (scores_tensor - mean) / (std + eps)

        for item, adv_scalar in zip(data, normalized.tolist()):
            L = item["input_ids"].size(0)
            adv_tensor = torch.full((L,), adv_scalar, dtype=torch.float32)
            # 只在 response 位置生效，其他置 0
            adv_tensor = adv_tensor * item["loss_mask"].float()
            item["advantages"] = adv_tensor
            item["group_adv_scalar"] = adv_scalar
        return data

    # ----------------------------------------------------------------
    # 2. 主训练循环
    # ----------------------------------------------------------------

    def get_action(self, state: Any, evaluate: bool = False) -> Any:
        return self.agent.chat_request(state)

    def train_step(self, batch_data: List[List[Dict[str, Any]]]) -> Dict[str, Union[float, str]]:
        """
        一次 GRPO 更新：tokenize → 组相对 advantage → 采 old_log_probs/ref_log_probs → 训练循环。
        """
        if not hasattr(self.agent, 'model'):
            return {"error": "Cannot train without a local model."}

        data = self._prepare_data(batch_data)
        if not data:
            return {"skipped": 1.0, "reason": "empty_batch"}

        logger.info(f"[GRPO] Phase A: tokenized {len(data)} trajectories; group adv normalized.")

        # -------- B. 预采 old_log_probs / ref_log_probs --------
        self.actor.eval()
        if self.ref_model is not None:
            self.ref_model.eval()

        with torch.no_grad():
            for i in range(0, len(data), self.mini_batch_size):
                batch = data[i:i + self.mini_batch_size]
                collated = collate_fn(self.tokenizer, batch)
                input_ids = collated["input_ids"].to(self.device)
                attn = collated["attention_mask"].to(self.device)

                old_log_probs, _, _ = forward_logprobs_and_entropy(self.actor, input_ids, attn)
                # use_ref=False 时 ref_log_probs := old_log_probs，k3 估计自动算为 0
                if self.ref_model is not None:
                    ref_log_probs, _, _ = forward_logprobs_and_entropy(self.ref_model, input_ids, attn)
                else:
                    ref_log_probs = old_log_probs

                for j, item in enumerate(batch):
                    L_j = item["input_ids"].size(0)
                    item["old_log_probs"] = old_log_probs[j, :L_j].cpu()
                    item["ref_log_probs"] = ref_log_probs[j, :L_j].cpu()

        # -------- C. 训练循环 --------
        self.actor.train()

        stats = {
            "actor_loss": 0.0, "entropy": 0.0, "kl_penalty": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0,
            "group_adv_mean": float(torch.tensor([d["group_adv_scalar"] for d in data]).mean().item()),
            "group_adv_std": float(torch.tensor([d["group_adv_scalar"] for d in data]).std(unbiased=False).item()),
        }

        logger.info(f"[GRPO] Phase C: optimizing for {self.ppo_epochs} epochs, mini_batch={self.mini_batch_size}")
        early_stop = False
        for epoch in range(self.ppo_epochs):
            if early_stop:
                break
            random.shuffle(data)
            for i in range(0, len(data), self.mini_batch_size):
                batch = data[i:i + self.mini_batch_size]

                collated = collate_fn(self.tokenizer, batch)
                input_ids = collated["input_ids"].to(self.device)
                attn = collated["attention_mask"].to(self.device)
                loss_mask = collated["loss_mask"].to(self.device)

                old_log_probs = torch.nn.utils.rnn.pad_sequence(
                    [b["old_log_probs"] for b in batch], batch_first=True, padding_value=0.0
                ).to(self.device)
                ref_log_probs = torch.nn.utils.rnn.pad_sequence(
                    [b["ref_log_probs"] for b in batch], batch_first=True, padding_value=0.0
                ).to(self.device)
                advantages = torch.nn.utils.rnn.pad_sequence(
                    [b["advantages"] for b in batch], batch_first=True, padding_value=0.0
                ).to(self.device)

                new_log_probs, entropy_per_pos, _ = forward_logprobs_and_entropy(
                    self.actor, input_ids, attn
                )

                mask = loss_mask.float()
                mask_sum = mask.sum().clamp(min=1e-8)

                # PPO-Clip actor loss（GRPO 也沿用 PPO-Clip 的形式）
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                actor_loss = -torch.min(surr1, surr2)
                actor_loss = (actor_loss * mask).sum() / mask_sum

                # KL (k3) & Entropy
                kl_per_pos = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1.0
                kl_loss = (kl_per_pos * mask).sum() / mask_sum
                entropy_loss = (entropy_per_pos * mask).sum() / mask_sum

                loss = actor_loss + self.kl_coef * kl_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (new_log_probs - old_log_probs)) * mask
                    approx_kl = approx_kl.sum() / mask_sum
                    clip_frac = (((ratio - 1.0).abs() > self.clip_ratio).float() * mask).sum() / mask_sum

                stats["actor_loss"] += actor_loss.item()
                stats["entropy"]    += entropy_loss.item()
                stats["kl_penalty"] += kl_loss.item()
                stats["approx_kl"]  += approx_kl.item()
                stats["clip_frac"]  += clip_frac.item()
                stats["n_updates"]  += 1

                if self.target_kl is not None and approx_kl.item() > 1.5 * self.target_kl:
                    logger.warning(
                        f"[GRPO] target_kl exceeded ({approx_kl.item():.4f} > 1.5 * {self.target_kl}), early stop"
                    )
                    early_stop = True
                    break

        n = max(1, stats["n_updates"])
        for k in ["actor_loss", "entropy", "kl_penalty", "approx_kl", "clip_frac"]:
            stats[k] = stats[k] / n

        return stats

    # ----------------------------------------------------------------
    # 3. 保存 / 加载
    # ----------------------------------------------------------------

    def save(self, path: str) -> None:
        if not hasattr(self.agent, 'model') or not hasattr(self.agent, 'tokenizer'):
            return
        os.makedirs(path, exist_ok=True)
        self.agent.model.save_pretrained(path)
        self.agent.tokenizer.save_pretrained(path)
        logger.info(f"[GRPO] Saved actor+tokenizer to {path}")

    def load(self, path: str) -> None:
        # Actor/Tokenizer 由 HFAgent 负责加载，无额外 critic 需要加载
        logger.info(f"[GRPO] Model loading handled by HFAgent, path={path}")
