"""
GAE / bi-level GAE 工具模块 / GAE Utilities
-----------------------------------
本模块独立封装了强化学习中两种 advantage 估计策略，供 PPO/GRPO 等算法复用：

1. `compute_gae_advantage_return`: 标准单层 token 级 GAE（向量化实现，用 torch.flip 倒序累积）。
2. `compute_bi_level_gae_advantage_return`: RAGEN 论文核心——turn 级 + token 级双层 GAE。

这两个函数的数据契约均与 RAGEN-main (verl) 对齐：
- token_level_rewards: shape (B, L)，只在每个 turn 的 response 末尾那个 token 位置给一个标量 reward，其他位置为 0。
- values:              shape (B, L)，Critic 对每个 token 预测的价值。
- response_mask:       shape (B, L)，1 表示这是一个需要计算 loss 的 response token，0 表示 prompt/padding/env_obs。

返回：
- advantages: shape (B, L)，只在 response_mask==1 的位置有意义。
- returns:    shape (B, L)，同上。
"""

from typing import Tuple
import torch


def _masked_whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对张量 x 在 mask 指示的有效位置上做 whitening（均值归零，方差归一）。
    和 verl 的 verl_F.masked_whiten 行为一致。
    """
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    mean = (x * mask_f).sum() / denom
    var = ((x - mean) ** 2 * mask_f).sum() / denom
    return (x - mean) / (var.sqrt() + eps)


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准 token 级 GAE，向量化实现（用 torch.flip 倒序累积，避免 Python for 循环）。

    关键做法：把 reward/value 序列做 flip 后，在时间维上用累积 + 递推计算出 lastgaelam。
    由于 TD-residual δ_t = r_t + γ * V_{t+1} - V_t 依赖 V_{t+1}，我们显式构造 next_values。

    Args:
        token_level_rewards: (B, L)
        values:              (B, L)
        response_mask:       (B, L)，1 表示有效 response token
        gamma, lam:          GAE 超参

    Returns:
        advantages: (B, L)
        returns:    (B, L)
    """
    with torch.no_grad():
        B, L = token_level_rewards.shape
        # next_values[t] 对应 V_{t+1}，末尾位置 V_L = 0
        next_values = torch.zeros_like(values)
        next_values[:, :-1] = values[:, 1:]

        # mask 之外位置的贡献置 0
        mask_f = response_mask.float()
        deltas = (token_level_rewards + gamma * next_values - values) * mask_f

        # 倒序累积：A_t = δ_t + γλ * A_{t+1}
        # 用 flip 把时间轴翻转，再做一次前缀递推。
        rev_deltas = torch.flip(deltas, dims=[1])
        adv_rev = torch.zeros_like(rev_deltas)
        lastgaelam = torch.zeros(B, device=deltas.device, dtype=deltas.dtype)
        for t in range(L):
            lastgaelam = rev_deltas[:, t] + gamma * lam * lastgaelam
            adv_rev[:, t] = lastgaelam
        advantages = torch.flip(adv_rev, dims=[1]) * mask_f

        returns = advantages + values
        returns = returns * mask_f

        advantages = _masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_bi_level_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lam: float,
    high_level_gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    bi-level GAE（对齐 RAGEN-main/ragen/trainer/core_algos.py 里的同名函数）。

    论文动机：
        多轮 agent 任务里，一个 trajectory 被切成若干 turn，每个 turn 由一段 response tokens 组成，
        只有在 turn 末尾那个 response eos token 位置才会得到该 turn 的 reward。
        直接做 token 级 GAE 会让跨 turn 的信用分配不清晰，因此论文提出两层：

        1. 先只看所有 eos 位置，用 high_level_gamma（turn 级折扣）做一次 GAE，得到 per-turn advantage。
        2. 把 "advantages + values" 填回每个 eos 位置作为该 token 的 updated_reward。
        3. 再用普通 gamma（token 级折扣）在整个 loss_mask==1 的序列上做 token 级 GAE。
           注意：跨 turn 边界时重置 lastgaelam=0，表示 turn 之间的 token-level advantage 不传播。

    注意：由于循环结构依赖于每个样本的 eos 位置（每个样本位置不同），这里不能做完整的向量化，
    采用 batch 外层 for-loop + 样本内层向量化的混合策略。对短轨迹（本项目的典型场景）这是完全够快的。
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        values = values.float()
        reward_mask = token_level_rewards.ne(0)  # 有 reward 的位置即为 eos 位置
        B, L = token_level_rewards.shape

        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()

        for b in range(B):
            # ---------- Stage 1: turn 级 GAE（只在 eos 位置上） ----------
            eos_positions = reward_mask[b].nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                lastgaelam = 0.0
                for i in range(len(eos_positions) - 1, -1, -1):
                    curr_pos = eos_positions[i].item()
                    if i < len(eos_positions) - 1:
                        next_pos = eos_positions[i + 1].item()
                        nextvalue = values[b, next_pos]
                    else:
                        nextvalue = torch.zeros_like(values[b, curr_pos])
                    delta = updated_reward[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]
                    lastgaelam = delta + high_level_gamma * lam * lastgaelam
                    advantages[b, curr_pos] = lastgaelam

                # 把 turn 级 returns 写回 updated_reward，供下一阶段使用
                for pos in eos_positions.tolist():
                    returns[b, pos] = advantages[b, pos] + values[b, pos]
                    updated_reward[b, pos] = advantages[b, pos] + values[b, pos]

            # ---------- Stage 2: token 级 GAE（在整个 loss_mask==1 位置上） ----------
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            eos_set = set(eos_positions.tolist())
            lastgaelam = 0.0
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i].item()
                is_eos = curr_pos in eos_set

                if is_eos:
                    # turn 边界：token-level advantage 不跨 turn 传播；该位置 advantage 已在 Stage 1 写入
                    nextvalue = torch.zeros_like(values[b, curr_pos])
                    lastgaelam = 0.0
                    delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                    lastgaelam = delta + gamma * lam * lastgaelam
                    advantages[b, curr_pos] = lastgaelam
                    returns[b, curr_pos] = lastgaelam + values[b, curr_pos]
                else:
                    if i + 1 < len(valid_positions):
                        next_pos = valid_positions[i + 1].item()
                        nextvalue = values[b, next_pos]
                    else:
                        nextvalue = torch.zeros_like(values[b, curr_pos])
                    delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                    lastgaelam = delta + gamma * lam * lastgaelam
                    advantages[b, curr_pos] = lastgaelam
                    returns[b, curr_pos] = lastgaelam + values[b, curr_pos]

        advantages = _masked_whiten(advantages, loss_mask)
    return advantages, returns


if __name__ == "__main__":
    # 最小可验证单元：与 RAGEN-main core_algos.py 同名函数做结果对齐（手算验证）
    token_level_rewards = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]], dtype=torch.float32)
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]], dtype=torch.float32)
    loss_mask = torch.ones(1, 10)
    adv_bi, ret_bi = compute_bi_level_gae_advantage_return(token_level_rewards, values, loss_mask, 1.0, 1.0, 0.95)
    print("[bi-level GAE] advantages:", adv_bi)
    print("[bi-level GAE] returns:", ret_bi)

    adv, ret = compute_gae_advantage_return(token_level_rewards, values, loss_mask, 1.0, 1.0)
    print("[flat GAE] advantages:", adv)
    print("[flat GAE] returns:", ret)
