"""
公共 Rollout 工具 / Shared Rollout Utilities
-----------------------------------
把"用一个 agent 在一个 env 里跑一条完整 trajectory"的逻辑集中到这里，供：
- `StarPOTrainer._rollout_one_trajectory`
- `PureRLTrainer._rollout_one_trajectory`
- `scripts/evaluate.py`
三处共享，消除之前三份各自演化、容易漂移的代码重复。

核心语义（与 RAGEN 论文 agent_proxy + ctx_manager 对齐 —— 路径 A）：

1. **System content 由 env 自动拼装**：
   ``env.build_system_content()`` 根据 ``system_prefix + env_instruction +
   grid_vocab + action_lookup`` 输出，**每个环境只描述一次任务**，详见
   ``envs/base_env.py::BaseEnv.build_system_content``。

2. **每个 turn 的 user message 末尾追加 FORMAT_PROMPT + LENGTH_PROMPT**：
   ``env.build_format_prompt(actions_left)``，对齐论文
   ``ctx_manager.py::_build_turn_state_content`` —— 每 turn 都重申一次"输出
   <think>...</think><answer>...</answer>"和"剩多少 action / 多少 token 预算"。
   这是论文比我们旧版本"只在 system 里说一次格式"做得对的地方：高频提醒
   显著提升 small model 的 format_compliance。

3. **第一轮 user message** = State + FORMAT_PROMPT；
   **后续 user message** = "Reward: ... \n State: ... " + FORMAT_PROMPT。
   不再注入 ``env.get_env_instruction()``（任务说明已经在 system 里给了）。
   不再注入 ``env.get_valid_actions()``（动作集合也已经在 system 里给了）。

4. **Turn-level 硬截断**：每次 chat_request 之前先查 turn_idx 是否已 >= max_turn，
   若触发则把末尾 entry 标注为 truncated_by_agent_budget 并 break。这对齐 RAGEN
   `agent_proxy.max_turn`。`max_env_steps` 是另一层独立截断（env 层），BaseEnv.step
   内部会自己维护，任一触发 → truncated=True。

5. **可选 format penalty**：训练时 StarPO 要扣分；纯 RL baseline 不扣；评估时不扣。
   由调用方用 `use_format_reward` / `format_penalty` 开关控制。

6. **格式检查**：`<think>...</think>\s*<answer>...</answer>` 正则（RAGEN `enable_think=True`
   时的默认格式）。评估时虽然不扣分，但仍把每 turn 的 `format_ok` 写入 entry 便于
   事后统计 format_compliance 指标。
"""

from typing import Any, Dict, List, Optional, Union
import re


# RAGEN `ctx_manager.py:158` 对齐：enable_think=True 时的解析正则。
_FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)


def check_format(response: str) -> bool:
    """检查一次 LLM 回复是否符合 `<think>...</think><answer>...</answer>` 约定格式。"""
    return bool(_FORMAT_PATTERN.search(response))


def _build_first_user_content(env: Any, obs: str) -> str:
    """
    第一轮 user message：``"State:\n<obs>\n<format_prompt>"``。

    跟 RAGEN ``ctx_manager.py::_build_turn_state_content`` 对齐：
    - State 标签 + obs 文本（不再带 "Please reason step by step" 这类强表述）
    - 末尾追加 FORMAT_PROMPT + LENGTH_PROMPT（每 turn 都追加，不仅是 system 里说一次）
    """
    fmt = env.build_format_prompt(env.actions_left) if hasattr(env, "build_format_prompt") else ""
    state_block = f"State:\n{obs}"
    return f"{state_block}\n{fmt}".rstrip() if fmt else state_block


def _build_next_user_content(env: Any, obs: str, env_reward: float) -> str:
    """
    第二轮起的 user message：``"Reward: <r>\nState:\n<obs>\n<format_prompt>"``。

    论文 ``ctx_manager`` 实际是把 reward 单独作为一条 user message 加进去（再后面
    跟一条 State message），但等价地把它揉进同一条 user message 信息无丢失，且
    显著缩短 chat_template 渲染长度，对 8GB 显存友好。
    """
    fmt = env.build_format_prompt(env.actions_left) if hasattr(env, "build_format_prompt") else ""
    body = f"Reward: {env_reward}\nState:\n{obs}"
    return f"{body}\n{fmt}".rstrip() if fmt else body


def rollout_one_trajectory(
    env: Any,
    agent: Any,
    *,
    seed: Optional[int],
    max_turn: int,
    use_format_reward: bool = False,
    format_penalty: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    用给定 agent 在给定 env 里跑一条完整 trajectory。

    .. note::
        **System prompt 由 ``env.build_system_content()`` 自动拼装**（路径 A 论文对齐：
        见 ``envs/base_env.py::BaseEnv``）。每个 turn 的 user message 末尾都会通过
        ``env.build_format_prompt(env.actions_left)`` 追加 FORMAT_PROMPT + LENGTH_PROMPT。
        换环境时不需要改 CLI 或调用方代码，prompt 随环境自动切换。

    Args:
        env:                  继承 BaseEnv 的环境实例，调用方已经构造好。
                              必须实现 ``build_system_content()`` / ``build_format_prompt()``
                              （BaseEnv 已给默认实现）。
        agent:                实现 `chat_request(messages) -> str` 的 agent
        seed:                 传给 env.reset 的 seed；None 表示让环境自选
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

    # 路径 A：system 完全由 env.build_system_content() 自动拼装；
    # 第一轮 user message = "State:\n{obs}\n<FORMAT_PROMPT>"。
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": env.build_system_content()},
        {"role": "user", "content": _build_first_user_content(env, obs)},
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
                "content": _build_next_user_content(env, obs, env_reward),
            })

    return trajectory


def batched_rollout_for_prompt(
    envs: List[Any],
    agent: Any,
    *,
    seed: Union[int, List[Optional[int]], None],
    max_turn: int,
    use_format_reward: bool = False,
    format_penalty: float = 0.0,
) -> List[List[Dict[str, Any]]]:
    """
    并行采样 N 条 trajectory，N = len(envs)。

    设计动机：单条 ``rollout_one_trajectory`` + ``HFAgent.chat_request`` 的串行循环
    在小模型 + 单卡场景下属于 memory-bandwidth bound，batch=1 浪费了大量 GPU 带宽。
    本函数把 N 条 traj 在每个 turn 上凑成 ``batched_chat_request`` 的一次 batch generate，
    利用同一次 weights fetch 服务 N 条 sequence，rollout 总时间下降 3~5x。

    ``seed`` 参数支持两种语义：
        - ``int`` / ``None`` → **广播到所有 N 个 env**（训练路径：同 prompt 内 R 条 traj 起点相同，
          只有 LLM 采样不同，跟串行 N 次 ``rollout_one_trajectory(seed=同一个)`` 严格等价）。
        - ``List[Optional[int]]`` → **逐 env 独立 seed**（评估路径：N 个 episode 各自独立的
          初始状态，跟串行 N 次 ``rollout_one_trajectory(seed=不同)`` 严格等价）。
          列表长度必须等于 ``len(envs)``，元素允许为 None（让该 env 自选 seed）。

    与"串行 N 次 ``rollout_one_trajectory``"的等价性保证：
        1. **环境侧**：N 个独立 env 实例，按 ``seed`` 解析后的 per-env seed 分别 reset，
           跟串行 N 次的 reset 行为逐字一致。env state 之间互不污染（独立实例）。
        2. **采样侧**：每个 turn 把所有"还活着"的 traj 凑 batch 调一次 ``batched_chat_request``，
           内部对每条独立 multinomial 采样，**采样分布与单条 generate 严格相同**（见
           ``HFAgent.batched_chat_request`` docstring）。
        3. **格式/奖励**：format check + format penalty 的计算逻辑与 ``rollout_one_trajectory``
           逐字一致（共用 ``check_format``）。
        4. **截断语义**：turn 级硬截断（``max_turn``）+ env 级 terminated/truncated 处理与
           单条版本完全一致；某条 traj 提前结束后立刻退出 batch，不浪费它的 generate 资源。

    Args:
        envs:               已经实例化好的 N 个 **独立** env（调用方负责 close）。长度即 N。
        agent:              必须实现 ``batched_chat_request``（BaseAgent 默认实现是串行 fallback）。
        seed:               int / None / List[Optional[int]]，详见上方语义说明。
        max_turn:           turn 级硬截断（单条 traj 最多 chat_request 调用次数）。必须 >= 1。
        use_format_reward:  True 则不符合格式的 turn 会额外扣 ``format_penalty``。
        format_penalty:     格式错误时叠加的负奖励（仅 use_format_reward=True 时生效）。

    Returns:
        长度 N 的 trajectory 列表，与 ``envs`` 顺序对应。每条 trajectory 的 entry schema
        与 ``rollout_one_trajectory`` 完全一致（obs / messages / response / reward / env_reward /
        format_penalty / format_ok / terminated / truncated / info / turn_idx）。
    """
    if max_turn < 1:
        raise ValueError(f"max_turn must be >= 1 (got {max_turn})")
    if not envs:
        return []

    n = len(envs)

    # ---- 解析 seed：单值广播 / 列表逐个分发 ----
    if isinstance(seed, list):
        if len(seed) != n:
            raise ValueError(
                f"seed list length ({len(seed)}) must match envs length ({n})"
            )
        per_env_seeds: List[Optional[int]] = list(seed)
    else:
        per_env_seeds = [seed] * n

    # ---- 每个 env 独立 reset，组装第一轮 user message ----
    obs_list: List[str] = []
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=per_env_seeds[i])
        obs_list.append(obs)

    # 路径 A：system 完全由 env.build_system_content() 自动拼装；
    # 第一轮 user message = "State:\n{obs}\n<FORMAT_PROMPT>"。
    messages_list: List[List[Dict[str, str]]] = []
    for i in range(n):
        messages_list.append([
            {"role": "system", "content": envs[i].build_system_content()},
            {"role": "user", "content": _build_first_user_content(envs[i], obs_list[i])},
        ])

    trajectories: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    terminated_arr = [False] * n
    truncated_arr = [False] * n
    obs_arr = list(obs_list)
    turn_idx = 0

    while True:
        # 还活着 = 还没 terminated 也没 truncated 的 env
        alive_idx = [i for i in range(n) if not (terminated_arr[i] or truncated_arr[i])]
        if not alive_idx:
            break

        # ---- Turn 级硬截断（与 rollout_one_trajectory 完全对齐）----
        if turn_idx >= max_turn:
            for i in alive_idx:
                if trajectories[i]:
                    trajectories[i][-1]["truncated"] = True
                    last_info = trajectories[i][-1].get("info") or {}
                    last_info["truncated_reason"] = "max_turn_reached"
                    last_info["max_turn"] = max_turn
                    trajectories[i][-1]["info"] = last_info
                truncated_arr[i] = True
            break

        # ---- 凑 batch 调一次 generate ----
        batch_messages = [messages_list[i] for i in alive_idx]
        batch_responses = agent.batched_chat_request(batch_messages)
        assert len(batch_responses) == len(batch_messages), (
            f"batched_chat_request returned {len(batch_responses)} responses for "
            f"{len(batch_messages)} prompts; agent implementation is broken."
        )

        # ---- 把每条 response 分发给各自的 env，更新 trajectory ----
        for sub_idx, i in enumerate(alive_idx):
            response = batch_responses[sub_idx]

            format_ok = check_format(response)
            step_penalty = 0.0
            if use_format_reward and not format_ok:
                step_penalty = format_penalty

            next_obs, env_reward, term, trunc, info = envs[i].step(response)
            total_reward = env_reward + step_penalty

            trajectories[i].append({
                "obs": obs_arr[i],
                "messages": list(messages_list[i]),
                "response": response,
                "reward": total_reward,
                "env_reward": env_reward,
                "format_penalty": step_penalty,
                "format_ok": format_ok,
                "terminated": term,
                "truncated": trunc,
                "info": info,
                "turn_idx": turn_idx,
            })

            terminated_arr[i] = bool(term)
            truncated_arr[i] = bool(trunc)
            obs_arr[i] = next_obs
            messages_list[i].append({"role": "assistant", "content": response})
            if not (term or trunc):
                messages_list[i].append({
                    "role": "user",
                    "content": _build_next_user_content(envs[i], next_obs, env_reward),
                })

        turn_idx += 1

    return trajectories


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
