"""
公共 Rollout 工具 / Shared Rollout Utilities
-----------------------------------
把"用一个 agent 在一个 env 里跑一条完整 trajectory"的逻辑集中到这里，供：
- `StarPOTrainer._rollout_one_trajectory`
- `PureRLTrainer._rollout_one_trajectory`
- `scripts/evaluate.py`
三处共享，消除之前三份各自演化、容易漂移的代码重复。

核心语义（与 RAGEN 论文 agent_proxy + ctx_manager 对齐）：
1. **第一轮 user message 注入 `env.get_env_instruction()`**（RAGEN 风格环境说明 +
   多动作协议示例）。后续 turn 不再重复注入，避免 context 膨胀。
2. **Turn-level 硬截断**：每次 chat_request 之前先查 turn_idx 是否已 >= max_turn，
   若触发则把末尾 entry 标注为 truncated_by_agent_budget 并 break。这对齐 RAGEN
   `agent_proxy.max_turn`。`max_env_steps` 是另一层独立截断（env 层），BaseEnv.step
   内部会自己维护，任一触发 → truncated=True。
3. **可选 format penalty**：训练时 StarPO 要扣分；纯 RL baseline 不扣；评估时不扣。
   由调用方用 `use_format_reward` / `format_penalty` 开关控制。
4. **格式检查**：`<think>...</think>\s*<answer>...</answer>` 正则（RAGEN `enable_think=True`
   时的默认格式）。评估时虽然不扣分，但仍把每 turn 的 `format_ok` 写入 entry 便于
   事后统计 format_compliance 指标。
"""

from typing import Any, Dict, List, Optional
import re


# RAGEN `ctx_manager.py:158` 对齐：enable_think=True 时的解析正则。
_FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)


def check_format(response: str) -> bool:
    """检查一次 LLM 回复是否符合 `<think>...</think><answer>...</answer>` 约定格式。"""
    return bool(_FORMAT_PATTERN.search(response))


def rollout_one_trajectory(
    env: Any,
    agent: Any,
    *,
    seed: Optional[int],
    system_prompt: str,
    max_turn: int,
    use_format_reward: bool = False,
    format_penalty: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    用给定 agent 在给定 env 里跑一条完整 trajectory。

    Args:
        env:                  继承 BaseEnv 的环境实例，调用方已经构造好
        agent:                实现 `chat_request(messages) -> str` 的 agent
        seed:                 传给 env.reset 的 seed；None 表示让环境自选
        system_prompt:        system message 内容
        max_turn:             turn 级硬截断（LLM chat_request 调用次数上限）。
                              与 env 自身的 max_steps 各自独立，任一触发 → truncate。
                              必须 >= 1。
        use_format_reward:    True 则不符合格式的 turn 会额外扣 `format_penalty`
                              （累加到该步的 `reward`）；False 则完全不扣。
        format_penalty:       格式错误时要叠加的负奖励。仅当 use_format_reward=True 时生效。

    Returns:
        trajectory: List[Dict]，每个 dict 代表一个 turn，包含：
            - obs:             本 turn 开始时的 observation（进入 chat_request 前）
            - messages:        本 turn chat_request 输入的 messages 拷贝（含 system + 所有 user/assistant）
            - response:        LLM 这次的完整回复
            - reward:          本 turn 用于训练的 reward（= env_reward + 可选 format_penalty）
            - env_reward:      本 turn 环境返回的原生 reward（不含 format penalty）
            - format_penalty:  本 turn 累加的 format penalty（use_format_reward=False 时恒为 0）
            - format_ok:       本 turn response 是否符合 `<think>/<answer>` 格式
            - terminated:      BaseEnv.step 返回的 terminated
            - truncated:       BaseEnv.step 返回的 truncated（max_turn 触发时会被改写为 True）
            - info:            BaseEnv.step 合并后的 info dict（含 executed_action_count 等）
            - turn_idx:        这是第几个 turn（从 0 开始）
    """
    if max_turn < 1:
        raise ValueError(f"max_turn must be >= 1 (got {max_turn})")

    obs, _reset_info = env.reset(seed=seed)

    # RAGEN 风格：第一轮 user message 注入环境玩法说明（含多动作协议示例）。
    # 后续 turn 不再注入，避免 context 累积过快（和 StarPOTrainer 原实现一致）。
    env_instruction = env.get_env_instruction() if hasattr(env, "get_env_instruction") else ""
    first_user = ""
    if env_instruction:
        first_user = env_instruction.rstrip() + "\n\n"
    first_user += f"{obs}\n{env.get_valid_actions()}\nPlease reason step by step."

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_user},
    ]

    trajectory: List[Dict[str, Any]] = []
    terminated, truncated = False, False
    turn_idx = 0

    while not (terminated or truncated):
        # ---- Turn-level 硬截断：在进入下一次 chat_request 前判定 ----
        # 如果已经消耗满 max_turn 个 LLM turn 却还没 terminated/truncated，
        # 就把"上一个 entry"标注为被 agent 预算截断，并退出循环。
        if turn_idx >= max_turn:
            if trajectory:
                trajectory[-1]["truncated"] = True
                last_info = trajectory[-1].get("info") or {}
                last_info["truncated_reason"] = "max_turn_reached"
                last_info["max_turn"] = max_turn
                trajectory[-1]["info"] = last_info
            truncated = True
            break

        response = agent.chat_request(messages)

        format_ok = check_format(response)
        step_penalty = 0.0
        if use_format_reward and not format_ok:
            step_penalty = format_penalty

        next_obs, env_reward, terminated, truncated, info = env.step(response)
        total_reward = env_reward + step_penalty

        trajectory.append({
            "obs": obs,
            "messages": list(messages),
            "response": response,
            "reward": total_reward,
            "env_reward": env_reward,
            "format_penalty": step_penalty,
            "format_ok": format_ok,
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
                "content": f"Observation: {obs}\nReward for last step: {env_reward}\nNext action?",
            })

    return trajectory


def judge_success(trajectory: List[Dict[str, Any]]) -> bool:
    """
    严格 RAGEN 口径的成功判定（对齐 `es_manager.get_rollout_states`）：

        success = terminated and (not truncated)

    若环境在 info 里显式提供了 `is_success` / `success` 字段，**以 info 为准**
    （兼容 BanditEnv 用 `success` / MathEnv 用 `is_success` / SokobanEnv 用
    `is_success` 这三种命名），否则回退到 RAGEN 口径。

    之所以 info 优先：Bandit "拉到 safe arm 但 reward=10" 的场景下，环境 terminated=True
    但 RAGEN 语义上应判失败，这只能靠 env 自己在 info["success"] 里明说。
    """
    if not trajectory:
        return False
    last = trajectory[-1]
    info = last.get("info") or {}
    if isinstance(info, dict):
        if "is_success" in info:
            return bool(info["is_success"])
        if "success" in info:
            return bool(info["success"])
    # RAGEN 严格口径兜底
    return bool(last.get("terminated", False)) and not bool(last.get("truncated", False))
