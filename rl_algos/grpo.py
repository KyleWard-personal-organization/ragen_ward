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
        self.micro_batch_size = config.micro_batch_size
        self.gradient_accumulation = max(1, int(config.gradient_accumulation))
        # 等效的 "mini_batch_size"（乘法派生，仅用于日志 / 对外展示）：
        # - gradient_accumulation == 1 时等价于"每个 micro_batch 立即 step"
        # - gradient_accumulation > 1 时走梯度累积
        self.mini_batch_size = self.micro_batch_size * self.gradient_accumulation
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

        # Gradient checkpointing: 强制启用。forward 时只保留 √L 层的激活，backward 时重算。
        # 代价：forward 时间 +30% 左右；收益：激活显存 ~0.5×，让小 VRAM 机能跑更大 micro_batch / max_seq_length。
        # ⚠️ HF 实现细节：`gradient_checkpointing_enable()` 会**全局**把 `config.use_cache = False`，
        # 这个 config 不区分 train/eval。如果不显式打开，rollout 阶段 `model.generate()` 也会被
        # 强制关闭 KV cache → autoregressive decode 退化为 O(L²) 而非 O(L)，慢 3-10×。
        # 修复：checkpointing 启用后立刻把 use_cache 改回 True。训练阶段所有 forward 入口
        # （`trajectory_utils.forward_logprobs_and_entropy`）都已经显式传 `use_cache=False`，
        # 所以 config 这里设 True 不会影响训练正确性，只会让 rollout 的 generate 用上 cache。
        if hasattr(self.actor, "gradient_checkpointing_enable"):
            self.actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.actor, "enable_input_require_grads"):
                self.actor.enable_input_require_grads()
            self.actor.config.use_cache = True
            logger.info("GRPO: gradient_checkpointing enabled on actor (non-reentrant); use_cache restored to True for rollout speed.")

        self.ref_model: Optional[nn.Module]
        if self.use_ref:
            logger.info("GRPO: creating frozen reference model (for KL) via deepcopy...")
            self.ref_model = copy.deepcopy(self.actor)
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
            # ref_model 只做 forward + no_grad，checkpointing 对它没用，关掉以省重算时间
            if hasattr(self.ref_model, "gradient_checkpointing_disable"):
                self.ref_model.gradient_checkpointing_disable()
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
            f"micro_bs={self.micro_batch_size} grad_accum={self.gradient_accumulation} "
            f"(effective mini_bs={self.mini_batch_size}) "
            f"use_ref={self.use_ref} kl_coef={self.kl_coef} "
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
            for i in range(0, len(data), self.micro_batch_size):
                batch = data[i:i + self.micro_batch_size]
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
            # ---- Gradient norm 相关（对齐 RAGEN paper Figure 6 ③ Gradient Norm）----
            # grad_norm:     按 optimizer step 取均值（趋势线，对应论文 EMA-smoothed 曲线）
            # grad_norm_max: 本次 train_step 内所有 optimizer step 的 max（spike detection 用）
            # n_grad_steps:  本次 train_step 实际触发 optimizer.step 的次数（用于求均值）
            "grad_norm": 0.0, "grad_norm_max": 0.0, "n_grad_steps": 0,
            "group_adv_mean": float(torch.tensor([d["group_adv_scalar"] for d in data]).mean().item()),
            "group_adv_std": float(torch.tensor([d["group_adv_scalar"] for d in data]).std(unbiased=False).item()),
        }

        logger.info(
            f"[GRPO] Phase C: optimizing for {self.ppo_epochs} epochs, "
            f"micro_batch={self.micro_batch_size}, grad_accum={self.gradient_accumulation} "
            f"(effective mini_batch={self.mini_batch_size})"
        )
        early_stop = False
        for epoch in range(self.ppo_epochs):
            if early_stop:
                break
            random.shuffle(data)
            # 每个 epoch 起始清零梯度；循环内按 accum 节奏 step（末尾强制 flush，防止数据不整除时梯度被丢）
            self.optimizer.zero_grad()
            accum_counter = 0
            for i in range(0, len(data), self.micro_batch_size):
                batch = data[i:i + self.micro_batch_size]

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

                # Gradient accumulation: 把 loss 缩放后 backward，梯度累加到 .grad；
                # 累满 accum 步或到达 epoch 末尾时才 clip + step + zero_grad。
                (loss / self.gradient_accumulation).backward()
                accum_counter += 1
                is_last_micro = (i + self.micro_batch_size >= len(data))
                if (accum_counter % self.gradient_accumulation == 0) or is_last_micro:
                    # clip_grad_norm_ 的返回值是**裁剪前**的 ℓ2 total norm，
                    # 这正好是论文 Figure 6 ③ "Gradient Norm" 想要的量（spike detection
                    # 必须看 pre-clip，否则 clip 会把所有 spike 都压成 max_norm=1.0 看不出来）。
                    raw_total_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    gn = float(raw_total_norm)
                    stats["grad_norm"] += gn
                    stats["grad_norm_max"] = max(stats["grad_norm_max"], gn)
                    stats["n_grad_steps"] += 1

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
        # grad_norm 用 optimizer.step 实际触发次数取均值（不是 micro_batch 数）
        n_g = max(1, stats["n_grad_steps"])
        stats["grad_norm"] = stats["grad_norm"] / n_g

        # ⚠️ 必须切回 eval 模式：train_step 结束后 trainer 会进入下一轮 rollout（model.generate）。
        # 如果留在 train 模式，Qwen2 forward 内部 `if self.gradient_checkpointing and self.training:`
        # 会判 True，强制把 use_cache 关掉 → autoregressive decode 退化为 O(L²)，rollout 慢 3-10×。
        # 切回 eval 后 self.training=False → checkpointing 路径不激活 → KV cache 正常工作。
        self.actor.eval()

        # Phase C 刚结束是整条 pipeline 的 VRAM 峰值点（activation + gradient + AdamW 临时 buffer
        # 都在 allocator pool 里）。在此调用 empty_cache 把闲置 bin 归还 CUDA driver，
        # 小 VRAM 环境下能显著降低 Windows WDDM 把 tensor 搬到 "共享 GPU 内存" 的概率，
        # 从而减少系统 RAM 被间接吃掉的那部分。单次开销 ~100-300 ms，相对数百秒的
        # train_step 可忽略；对数值 bit-exact 无任何影响。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
