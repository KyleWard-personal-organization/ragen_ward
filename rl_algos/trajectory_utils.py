"""
轨迹 tokenize / collate 公共工具 / Trajectory Tokenization Utilities
-----------------------------------
PPO 和 GRPO 共用的数据处理逻辑：
- `tokenize_trajectory`: 把一整条多轮对话的 trajectory 拼成一条长 token 序列 +
   精确对齐的 loss_mask + token_level_rewards。
- `collate_fn`: 把一个 mini-batch 的不等长样本做右端 padding 对齐。
- `forward_logprobs_and_entropy`: 对任何 CausalLM 做一次前向，返回每个位置的
   token log_prob 和熵，并把最后一位 pad 到 0，使形状与 input_ids 对齐，方便下游
   直接按 (B, L) 使用 loss_mask/token_level_rewards。
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def tokenize_trajectory(
    tokenizer: Any,
    trajectory: List[Dict[str, Any]],
    max_seq_length: int = 4096,
) -> Dict[str, torch.Tensor]:
    """
    基于 chat_template 字符串增量的多轮 trajectory tokenize。

    核心思想：每一个 turn 结束时，用 apply_chat_template 得到"从开头到本 turn 的
    完整字符串"，取相对于上一轮字符串的后缀增量，单独 tokenize 后追加到 cur_ids。
    这样既能精确打上 loss_mask，也能让 tokenizer 正确处理 `<|im_start|>` 这类
    特殊 token 的边界问题。

    Args:
        tokenizer:      HF AutoTokenizer 实例
        trajectory:     List[{"messages": [...], "response": "...", "reward": float, ...}]
        max_seq_length: 超长截断阈值（保留头部，截掉尾部）

    Returns:
        dict:
          input_ids:           (L,) long
          loss_mask:           (L,) long，1 代表 response token，0 代表 prompt/obs/padding
          token_level_rewards: (L,) float，只有每个 turn response 末尾位置有非零值
          prompt_len:          int，第一条 prompt 的 token 长度
    """
    if len(trajectory) == 0:
        return {
            "input_ids": torch.zeros(0, dtype=torch.long),
            "loss_mask": torch.zeros(0, dtype=torch.long),
            "token_level_rewards": torch.zeros(0, dtype=torch.float32),
            "prompt_len": 0,
        }

    # 初始 prompt（通常是 system + 第一个 user message）
    initial_text = tokenizer.apply_chat_template(
        trajectory[0]["messages"], tokenize=False, add_generation_prompt=False
    )
    cur_ids: List[int] = list(tokenizer(initial_text, add_special_tokens=False).input_ids)
    prompt_len = len(cur_ids)
    loss_mask: List[int] = [0] * len(cur_ids)
    token_rewards: List[float] = [0.0] * len(cur_ids)
    prev_text = initial_text

    for k, step in enumerate(trajectory):
        # ---- 1) 本 turn 的 assistant response 段 ----
        full_messages = step["messages"] + [{"role": "assistant", "content": step["response"]}]
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        response_delta = full_text[len(prev_text):]
        if response_delta:
            response_ids = list(tokenizer(response_delta, add_special_tokens=False).input_ids)
            cur_ids.extend(response_ids)
            loss_mask.extend([1] * len(response_ids))
            token_rewards.extend([0.0] * len(response_ids))
            if response_ids:
                # turn 级 reward 打在本 turn response 的最后一个 token 上
                token_rewards[-1] = float(step["reward"])
        prev_text = full_text

        # ---- 2) 下一 turn 的 user_obs 段（若还存在下一 turn）----
        if k + 1 < len(trajectory):
            next_text = tokenizer.apply_chat_template(
                trajectory[k + 1]["messages"], tokenize=False, add_generation_prompt=False
            )
            obs_delta = next_text[len(prev_text):]
            if obs_delta:
                obs_ids = list(tokenizer(obs_delta, add_special_tokens=False).input_ids)
                cur_ids.extend(obs_ids)
                loss_mask.extend([0] * len(obs_ids))
                token_rewards.extend([0.0] * len(obs_ids))
            prev_text = next_text

    # 超长截断：保留头部 prompt + 尽量多的前段 turn
    if len(cur_ids) > max_seq_length:
        cur_ids = cur_ids[:max_seq_length]
        loss_mask = loss_mask[:max_seq_length]
        token_rewards = token_rewards[:max_seq_length]

    return {
        "input_ids": torch.tensor(cur_ids, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        "token_level_rewards": torch.tensor(token_rewards, dtype=torch.float32),
        "prompt_len": prompt_len,
    }


def collate_fn(
    tokenizer: Any,
    batch: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    将一个 mini-batch 的变长样本右端 padding 到统一长度。
    返回 dict: input_ids / attention_mask / loss_mask / token_level_rewards。
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
    )
    loss_mask = torch.nn.utils.rnn.pad_sequence(
        [b["loss_mask"] for b in batch], batch_first=True, padding_value=0
    )
    token_level_rewards = torch.nn.utils.rnn.pad_sequence(
        [b["token_level_rewards"] for b in batch], batch_first=True, padding_value=0.0
    )
    attention_mask = (input_ids != pad_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "token_level_rewards": token_level_rewards,
    }


def forward_logprobs_and_entropy(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    return_hidden_states: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    前向模型并取每个位置上的 token log_prob 和熵，与 `input_ids` 严格**逐位对齐**。

    本函数的对齐约定（重要）：
        log_probs[b, t] = log P(input_ids[b, t] | input_ids[b, :t])
        也就是说：位置 t 处的 log_prob 是"模型预测位置 t 上这个 token 的概率"。
        因此 loss_mask[b, t]=1 时，loss_mask * log_probs 就是这个 response token 的 log-prob。

    由于 transformer 前向 logits[:, t-1, :] 才是预测位置 t 的分布，所以我们在**前面 pad 一个 0**，
    而不是后面。位置 0 上 log_prob=0 纯属占位（下游 loss_mask[:, 0] 必然为 0，因为首 token 是 BOS/prompt）。

    values 的对齐做成同样的约定：values[b, t] = V(前 t 个 token 之后的状态)。

    Returns:
        log_probs:    (B, L) float
        entropy:      (B, L) float
        hidden_states:(B, L, H) float 或 None
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=return_hidden_states,
        use_cache=False,
    )
    logits = outputs.logits.float()                              # (B, L, V)
    log_probs_slice = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (B, L-1, V)
    labels = input_ids[:, 1:]                                    # (B, L-1)
    token_log_probs = log_probs_slice.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    probs = torch.exp(log_probs_slice)
    entropy_per_pos = -(probs * log_probs_slice).sum(dim=-1)     # (B, L-1)

    # 前面 pad 一个 0，让索引 t 上的值对齐 input_ids[:, t]
    pad_zero = torch.zeros_like(token_log_probs[:, :1])
    token_log_probs = torch.cat([pad_zero, token_log_probs], dim=1)  # (B, L)
    entropy_per_pos = torch.cat([pad_zero, entropy_per_pos], dim=1)  # (B, L)

    hidden_states = outputs.hidden_states[-1] if return_hidden_states else None
    return token_log_probs, entropy_per_pos, hidden_states
