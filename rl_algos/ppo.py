"""
PPO (Proximal Policy Optimization) 算法 / PPO Algorithm
-----------------------------------
工业级 PPO 的轨迹级实现，严格对齐 RAGEN/veRL 的训练逻辑：

1. 把一整条多轮 trajectory 拼成**一条长序列**（prompt + assistant_1 + obs_1 + assistant_2 + obs_2 + ... + assistant_N），
   而不是把每个 turn 当做独立样本。这是复现多轮 credit assignment 的前提。
2. 用 `loss_mask` 精确标记出需要学习的 assistant token，在非 assistant 位置上 mask 掉 loss。
3. token_level_rewards 只在每个 assistant turn 的末尾位置放入该 turn 的 reward。
4. 支持 **bi-level GAE**（turn 级 + token 级），通过 `bi_level_gae` 开关切换单层/双层。
5. GAE 计算全部向量化（`gae_utils.py` 内部用 torch.flip 倒序递推）。
6. 完整的 Actor-Critic 架构（Critic 是共享 backbone 的 value head）+ KL 约束 + PPO-Clip。
"""

import os
import copy
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base_algo import BaseRLAlgo
from .gae_utils import compute_bi_level_gae_advantage_return, compute_gae_advantage_return
from .optimizer_utils import build_optimizer
from .trajectory_utils import tokenize_trajectory, collate_fn, forward_logprobs_and_entropy
from utils.logger import logger
from tqdm import tqdm


class PPO(BaseRLAlgo):
    """
    PPO 算法实现。

    关键接口：
    - `train_step(batch_data: List[List[Dict]])`：接收一个 batch 的 trajectories，每条 trajectory
       是一个 dict 列表 (每个 dict 代表一个 turn，含 messages/response/reward/terminated 字段)。
    - `save(path)` / `load(path)`：保存/加载 Actor 和 Critic 权重。
    """

    def __init__(self, config: Any, agent: Any):
        super().__init__(config, agent)

        # ---------- 超参（直接属性访问，缺失立即 AttributeError；默认值仅来自 scripts/train.py） ----------
        self.lr = config.learning_rate
        self.gamma = config.gamma
        self.lam = config.lam
        self.bi_level_gae = config.bi_level_gae
        self.high_level_gamma = config.high_level_gamma
        self.clip_ratio = config.clip_ratio
        self.ppo_epochs = config.ppo_epochs
        self.micro_batch_size = config.micro_batch_size
        self.gradient_accumulation = max(1, int(config.gradient_accumulation))
        # 等效的 "mini_batch_size"（乘法派生，仅用于日志 / 对外展示）：
        # - gradient_accumulation == 1 时等价于"每个 micro_batch 立即 step"
        # - gradient_accumulation > 1 时走梯度累积
        self.mini_batch_size = self.micro_batch_size * self.gradient_accumulation
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.kl_coef = config.kl_coef
        self.target_kl = config.target_kl                  # Optional[float]; None 即不启用
        self.max_seq_length = config.max_seq_length
        self.use_ref = config.use_ref
        self.optimizer_name = config.optimizer

        if not self.use_ref and self.kl_coef != 0.0:
            logger.warning(
                f"PPO: use_ref=False but kl_coef={self.kl_coef} != 0; "
                f"forcing kl_coef=0 because no reference distribution is available."
            )
            self.kl_coef = 0.0

        # ---------- 依赖 Agent ----------
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
            logger.info("PPO: gradient_checkpointing enabled on actor (non-reentrant); use_cache restored to True for rollout speed.")

        self.ref_model: Optional[nn.Module]
        if self.use_ref:
            logger.info("PPO: creating frozen reference model (for KL) via deepcopy...")
            self.ref_model = copy.deepcopy(self.actor)
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
            # ref_model 只做 forward + no_grad，checkpointing 对它没用，关掉以省重算时间
            if hasattr(self.ref_model, "gradient_checkpointing_disable"):
                self.ref_model.gradient_checkpointing_disable()
        else:
            logger.info("PPO: use_ref=False, skipping ref_model (saves ~1GB VRAM, disables KL anchor).")
            self.ref_model = None

        logger.info("PPO: creating Critic value head (Linear on top of actor hidden_states)...")
        hidden_size = self.actor.config.hidden_size
        self.critic = nn.Linear(hidden_size, 1, dtype=self.actor.dtype).to(self.device)

        self.optimizer = build_optimizer(
            name=self.optimizer_name,
            params=list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
            actor=self.actor,
        )
        logger.info(
            f"PPO initialized | optimizer={self.optimizer_name} lr={self.lr} epochs={self.ppo_epochs} "
            f"micro_bs={self.micro_batch_size} grad_accum={self.gradient_accumulation} "
            f"(effective mini_bs={self.mini_batch_size}) "
            f"use_ref={self.use_ref} kl_coef={self.kl_coef} "
            f"bi_level_gae={self.bi_level_gae} high_level_gamma={self.high_level_gamma} "
            f"gamma={self.gamma} lam={self.lam} clip={self.clip_ratio}"
        )

    # ----------------------------------------------------------------
    # 1. 数据准备：把 trajectory 拼成长序列（复用 trajectory_utils 的实现）
    # ----------------------------------------------------------------

    def _prepare_data(self, batch_data: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """将 batch trajectories tokenize 成长序列字典列表。"""
        data = []
        for traj in batch_data:
            if len(traj) == 0:
                continue
            item = tokenize_trajectory(self.tokenizer, traj, self.max_seq_length)
            if item["loss_mask"].sum().item() == 0:
                # 纯 prompt 没有任何 assistant token（意外情况），跳过
                continue
            data.append(item)
        return data

    # ----------------------------------------------------------------
    # 2. 前向传播：获取 log_probs 与 values
    # ----------------------------------------------------------------

    def _forward_actor_with_critic(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向 actor 并由 critic head 从最后一层 hidden_states 计算 values。"""
        log_probs, entropy, hidden = forward_logprobs_and_entropy(
            self.actor, input_ids, attention_mask, return_hidden_states=True,
        )
        values = self.critic(hidden).squeeze(-1).float()  # (B, L)
        return log_probs, entropy, values

    def _forward_ref(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """前向参考模型，只取 log_probs。仅在 self.ref_model is not None 时调用。"""
        assert self.ref_model is not None, "_forward_ref called while ref_model is None"
        log_probs, _, _ = forward_logprobs_and_entropy(
            self.ref_model, input_ids, attention_mask, return_hidden_states=False,
        )
        return log_probs

    # ----------------------------------------------------------------
    # 3. 主训练循环
    # ----------------------------------------------------------------

    def get_action(self, state: Any, evaluate: bool = False) -> Any:
        return self.agent.chat_request(state)

    def train_step(self, batch_data: List[List[Dict[str, Any]]]) -> Dict[str, Union[float, str]]:
        """
        接收一个 batch 的 trajectories，做一次完整的 PPO 更新。

        Pipeline:
            A. Tokenize trajectories → 长序列张量
            B. 预计算阶段：对每个样本前向得到 old_log_probs / ref_log_probs / values / advantages
            C. 训练阶段：ppo_epochs 轮，对每个 mini-batch 做 PPO-Clip 更新
            D. 返回指标
        """
        if not hasattr(self.agent, 'model'):
            return {"error": "Cannot train without a local model."}

        data = self._prepare_data(batch_data)
        if not data:
            return {"skipped": 1.0, "reason": "empty_batch"}

        logger.info(f"[PPO] Phase A: tokenized {len(data)} trajectories into sequences")

        # -------- B. 预计算：old_log_probs / ref_log_probs / values / advantages / returns --------
        self.actor.eval()
        if self.ref_model is not None:
            self.ref_model.eval()
        self.critic.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(data), self.micro_batch_size), desc=f"[PPO] Phase A"):
                batch = data[i:i + self.micro_batch_size]
                collated = collate_fn(self.tokenizer, batch)
                input_ids = collated["input_ids"].to(self.device)
                attn = collated["attention_mask"].to(self.device)
                loss_mask = collated["loss_mask"].to(self.device)
                token_rewards = collated["token_level_rewards"].to(self.device)

                old_log_probs, _, values = self._forward_actor_with_critic(input_ids, attn)
                # use_ref=False 时 ref_log_probs := old_log_probs，k3 估计自动算为 0
                if self.ref_model is not None:
                    ref_log_probs = self._forward_ref(input_ids, attn)
                else:
                    ref_log_probs = old_log_probs

                # 计算 GAE（优先 bi-level，否则 flat）
                if self.bi_level_gae:
                    advantages, returns = compute_bi_level_gae_advantage_return(
                        token_level_rewards=token_rewards,
                        values=values,
                        loss_mask=loss_mask,
                        gamma=self.gamma,
                        lam=self.lam,
                        high_level_gamma=self.high_level_gamma,
                    )
                else:
                    advantages, returns = compute_gae_advantage_return(
                        token_level_rewards=token_rewards,
                        values=values,
                        response_mask=loss_mask,
                        gamma=self.gamma,
                        lam=self.lam,
                    )

                # 写回到单条样本 (去 padding)
                for j, item in enumerate(batch):
                    L_j = item["input_ids"].size(0)
                    item["old_log_probs"] = old_log_probs[j, :L_j].cpu()
                    item["ref_log_probs"] = ref_log_probs[j, :L_j].cpu()
                    item["values"]        = values[j, :L_j].cpu()
                    item["advantages"]    = advantages[j, :L_j].cpu()
                    item["returns"]       = returns[j, :L_j].cpu()

        # -------- C. 训练：ppo_epochs 次遍历，mini-batch 更新 --------
        self.actor.train()
        self.critic.train()

        stats = {
            "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "kl_penalty": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0,
        }

        logger.info(
            f"[PPO] Phase C: optimizing for {self.ppo_epochs} epochs, "
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
            for i in tqdm(range(0, len(data), self.micro_batch_size), desc=f"[PPO] Phase C"):
                batch = data[i:i + self.micro_batch_size]

                # 重新 collate 这个 micro-batch（因为 padding 长度需要按当前 batch 定）
                collated = collate_fn(self.tokenizer, batch)
                input_ids = collated["input_ids"].to(self.device)
                attn = collated["attention_mask"].to(self.device)
                loss_mask = collated["loss_mask"].to(self.device)

                # 预计算值 padding 到同一长度
                old_log_probs = torch.nn.utils.rnn.pad_sequence(
                    [b["old_log_probs"] for b in batch], batch_first=True, padding_value=0.0
                ).to(self.device)
                ref_log_probs = torch.nn.utils.rnn.pad_sequence(
                    [b["ref_log_probs"] for b in batch], batch_first=True, padding_value=0.0
                ).to(self.device)
                advantages = torch.nn.utils.rnn.pad_sequence(
                    [b["advantages"] for b in batch], batch_first=True, padding_value=0.0
                ).to(self.device)
                returns = torch.nn.utils.rnn.pad_sequence(
                    [b["returns"] for b in batch], batch_first=True, padding_value=0.0
                ).to(self.device)

                # 本次前向（actor 要有梯度；critic 共享 actor backbone 的 hidden_states）
                new_log_probs, entropy_per_pos, new_values = self._forward_actor_with_critic(input_ids, attn)

                mask = loss_mask.float()
                mask_sum = mask.sum().clamp(min=1e-8)

                # ---- PPO-Clip actor loss ----
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                actor_loss = -torch.min(surr1, surr2)
                actor_loss = (actor_loss * mask).sum() / mask_sum

                # ---- Critic MSE loss ----
                # (注：也可以做 value clipping，这里先用朴素 MSE 保持简单)
                critic_loss = (new_values - returns) ** 2
                critic_loss = (critic_loss * mask).sum() / mask_sum

                # ---- KL (k3 estimator，来自 Schulman's blog，稳定性更好) ----
                kl_per_pos = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1.0
                kl_loss = (kl_per_pos * mask).sum() / mask_sum

                # ---- Entropy bonus ----
                entropy_loss = (entropy_per_pos * mask).sum() / mask_sum

                loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    + self.kl_coef * kl_loss
                    - self.ent_coef * entropy_loss
                )

                # Gradient accumulation: 把 loss 缩放后 backward，梯度累加到 .grad；
                # 累满 accum 步或到达 epoch 末尾时才 clip + step + zero_grad。
                (loss / self.gradient_accumulation).backward()
                accum_counter += 1
                is_last_micro = (i + self.micro_batch_size >= len(data))
                if (accum_counter % self.gradient_accumulation == 0) or is_last_micro:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()), 1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # ---- 统计（off-graph） ----
                with torch.no_grad():
                    # approx_kl 按 Schulman 近似：E[(ratio-1)-log(ratio)]
                    approx_kl = ((ratio - 1.0) - (new_log_probs - old_log_probs)) * mask
                    approx_kl = approx_kl.sum() / mask_sum
                    clip_frac = (((ratio - 1.0).abs() > self.clip_ratio).float() * mask).sum() / mask_sum

                stats["actor_loss"]  += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"]     += entropy_loss.item()
                stats["kl_penalty"]  += kl_loss.item()
                stats["approx_kl"]   += approx_kl.item()
                stats["clip_frac"]   += clip_frac.item()
                stats["n_updates"]   += 1

                # 可选：target_kl 早停
                if self.target_kl is not None and approx_kl.item() > 1.5 * self.target_kl:
                    logger.warning(
                        f"[PPO] target_kl exceeded ({approx_kl.item():.4f} > 1.5 * {self.target_kl}), early stop"
                    )
                    early_stop = True
                    break

        # 求平均
        n = max(1, stats["n_updates"])
        for k in ["actor_loss", "critic_loss", "entropy", "kl_penalty", "approx_kl", "clip_frac"]:
            stats[k] = stats[k] / n

        # ⚠️ 必须切回 eval 模式：train_step 结束后 trainer 会进入下一轮 rollout（model.generate）。
        # 如果留在 train 模式，Qwen2 forward 内部 `if self.gradient_checkpointing and self.training:`
        # 会判 True，强制把 use_cache 关掉 → autoregressive decode 退化为 O(L²)，rollout 慢 3-10×。
        # 切回 eval 后 self.training=False → checkpointing 路径不激活 → KV cache 正常工作。
        self.actor.eval()
        self.critic.eval()

        # Phase C 刚结束是整条 pipeline 的 VRAM 峰值点（activation + gradient + AdamW 临时 buffer
        # 都在 allocator pool 里）。在此调用 empty_cache 把闲置 bin 归还 CUDA driver，
        # 小 VRAM 环境下能显著降低 Windows WDDM 把 tensor 搬到 "共享 GPU 内存" 的概率，
        # 从而减少系统 RAM 被间接吃掉的那部分。单次开销 ~100-300 ms，相对数百秒的
        # train_step 可忽略；对数值 bit-exact 无任何影响。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return stats

    # ----------------------------------------------------------------
    # 4. 保存 / 加载
    # ----------------------------------------------------------------

    def save(self, path: str) -> None:
        if not hasattr(self.agent, 'model') or not hasattr(self.agent, 'tokenizer'):
            return
        os.makedirs(path, exist_ok=True)
        self.agent.model.save_pretrained(path)
        self.agent.tokenizer.save_pretrained(path)
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        logger.info(f"[PPO] Saved actor+tokenizer+critic to {path}")

    def load(self, path: str) -> None:
        # Actor/Tokenizer 由 HFAgent 负责；此处仅加载 critic（如果存在）
        critic_path = os.path.join(path, "critic.pt")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            logger.info(f"[PPO] Loaded critic from {critic_path}")
        else:
            logger.info(f"[PPO] No critic.pt under {path}, skip critic loading.")
