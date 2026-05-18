"""
环境基类 / Base Environment Class
-----------------------------------
所有自定义环境或 gym 包装环境都必须继承该类。与 LLM 代理（基于文本）的交互解耦。

## RAGEN 对齐：一次 LLM 回复 → 多个原子 env step

为和 RAGEN 论文的 `max_actions_per_traj` + `||` 动作序列语义对齐，
`BaseEnv.step(action_text)` 被提升为"批量动作执行器"：
    1. `_parse_action_sequence(action_text)` → List[atomic_action]
       （默认实现把整段文本当成一个原子动作；支持 `||` 的子类可覆盖它）
    2. 依次调用 `_step_atomic(atomic_action)` 直到：
         - 某一步 `terminated=True`；或
         - `self.current_step >= self.max_steps` → `truncated=True`；或
         - 动作序列执行完毕
    3. 聚合奖励（累加），返回最终 obs / 累计 reward / 标志位 / 合并后的 info。

## 子类义务

- **必须实现** `_step_atomic(atomic_action)`：单个原子动作的执行；**不要**在这里修改 `self.current_step`，由 `BaseEnv.step` 统一维护。
- **必须实现** `reset()`、`get_valid_actions()`、`render()`。
- **可以覆盖** `_parse_action(text)`、`_parse_action_sequence(text)`。

## RAGEN 论文 prompt 结构（路径 A 严格对齐）

System content 不再由子类硬编码长 prompt，而是由基类按下面公式自动拼装：

    system_content = system_prefix + env_instruction
                     + auto-generated "The meaning of each symbol ..." (from grid_vocab)
                     + auto-generated "Your available actions are ..." (from action_lookup)

每个 turn 的 user message 末尾会追加：

    "You have N actions left. Always output: <think> [Your thoughts] </think>
     <answer> [your answer] </answer> with no extra text. Strictly follow this format.
     Max response length: M words (tokens)."

跟 RAGEN ``ragen/llm_agent/ctx_manager.py::_build_system_content`` /
``_build_format_prompt`` / ``_build_turn_state_content`` 的注入策略逐字对齐。

子类需要做的就是把下面 5 个 class attributes 配好：
    env_instruction:       一句话任务描述 + 一个 example，**不要写 heuristic**、**不要泄露 reward 规则**
    grid_vocab:            符号表 dict，会被自动拼成 system 的一部分
    action_lookup:         动作表 dict，会被自动拼成 system 的一部分
    max_actions_per_traj:  论文 `max_actions_per_traj`，决定 system 里 "You can make up to N actions" 的 N
    max_response_tokens:   论文 `max_tokens`，决定每 turn LENGTH_PROMPT 里报告的预算
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List, Optional


class BaseEnv(ABC):

    # ------------------------------------------------------------------
    # RAGEN 论文 prompt 五件套（路径 A 严格对齐）
    # ------------------------------------------------------------------
    # 论文 `ctx_manager.py::_build_system_content` 的固定前缀：
    #   "You're a helpful assistant. " + env_instruction
    # 这一行**不要**改成"专家 agent"之类的强表述，论文就是用通用 helper 角色，
    # 让 reasoning 能力完全靠 RL 学习而不是靠 prompt 灌输。
    system_prefix: str = "You're a helpful assistant."

    # 一句话任务描述 + 至多一个 example（**不超过 80 词**）。
    # 严格禁止：① 任何"如何决策"的人类先验提示（"plan ahead"/"push toward leaning"等）
    #          ② 任何 reward 函数泄露（"reward 1 if reach goal"等）
    #          ③ 任何成功/失败规则（让 RL 通过试错学到）
    # 论文示例（FrozenLake）："You are solving the FrozenLake puzzle. Forbid the hole
    # and go to the target. You may move to the unintended direction due to the
    # slippery ice. Example answer format: To forbid the hole and go to the target,
    # I should go left then go up. <answer>Left || Up</answer>"
    env_instruction: str = ""

    # Grid 符号表。若提供，build_system_content 会自动拼出：
    #   "The meaning of each symbol in the state is:\nP: player, _: empty, ..."
    # None 表示该环境的 obs 不是网格（如 Bandit、Math）。
    grid_vocab: Optional[Dict[str, str]] = None

    # Action 表。若提供，build_system_content 会自动拼出：
    #   "Your available actions are:\nLeft, Down, Right, Up\n
    #    You can make up to N actions, separated by the action separator " || ""
    # None 表示动作集合是动态的（如 Bandit 的 arm 名字每个 episode 不同）。
    action_lookup: Optional[Dict[int, str]] = None

    # 一条 trajectory 内允许的最大原子 action 数。> 1 时会在 system 里告诉
    # 模型可以用 `||` 串多个。论文的 SimpleSokoban / FrozenLake 都是 10。
    max_actions_per_traj: int = 1

    # 单次 LLM 回复的 token 预算上报值（仅用于 LENGTH_PROMPT 文案，**不**用于
    # 实际截断；实际截断由 agent.max_new_tokens 控制）。论文典型值 100。
    max_response_tokens: int = 100

    # 多动作分隔符。论文用 " || "（前后带空格），我们解析时已对带/不带空格
    # 两种形式做了容错。
    action_separator: str = " || "

    def __init__(self, config: Any):
        self.config = config
        self.current_step = 0
        # 必填字段；缺失直接 AttributeError，不再静默兜底。
        self.max_steps = config.max_steps

    # ------------------------------------------------------------------
    # 论文风格 prompt 构建（rollout_utils 直接调）
    # ------------------------------------------------------------------

    def build_system_content(self) -> str:
        """
        按 RAGEN ``_build_system_content`` 的公式拼出 system message 内容：

            "You're a helpful assistant. " + env_instruction
            + "\nThe meaning of each symbol in the state is:\n<vocab>"  (if grid_vocab)
            + "\nYour available actions are:\n<actions>\n
               You can make up to N actions, separated by ' || '"        (if action_lookup)
        """
        parts: List[str] = [self.system_prefix.rstrip()]

        if self.env_instruction:
            parts.append(self.env_instruction.rstrip())

        if self.grid_vocab:
            vocab_str = "The meaning of each symbol in the state is:\n" + ", ".join(
                f"{k}: {v}" for k, v in self.grid_vocab.items()
            )
            parts.append(vocab_str)

        if self.action_lookup:
            actions_str = (
                "Your available actions are:\n"
                + ", ".join(str(v) for v in self.action_lookup.values())
            )
            if self.max_actions_per_traj > 1:
                actions_str += (
                    f"\nYou can make up to {self.max_actions_per_traj} actions, "
                    f"separated by the action separator \"{self.action_separator}\""
                )
            parts.append(actions_str)

        return " ".join(parts)

    def build_format_prompt(self, actions_left: int) -> str:
        """
        按 RAGEN ``_build_format_prompt`` + ``_build_turn_state_content`` 的公式
        生成每个 turn user message 末尾要追加的 FORMAT_PROMPT + LENGTH_PROMPT：

            "You have N actions left. Always output: <think> [Your thoughts] </think>
             <answer> [your answer] </answer> with no extra text.
             Strictly follow this format. Max response length: M words (tokens)."
        """
        return (
            f"You have {actions_left} actions left. "
            f"Always output: <think> [Your thoughts] </think>"
            f"<answer> [your answer] </answer> with no extra text. "
            f"Strictly follow this format. "
            f"Max response length: {self.max_response_tokens} words (tokens)."
        )

    @property
    def actions_left(self) -> int:
        """剩余可执行的原子 action 数（用于 build_format_prompt）。"""
        return max(0, int(self.max_steps) - int(self.current_step))

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict]:
        """
        重置环境状态。子类 **必须**先 `super().reset(seed=...)` 让 `current_step` 清零。

        Returns:
            (文本化的初始 observation, 附加 info)
        """
        self.current_step = 0
        return "", {}  # 子类应覆盖返回值

    def step(self, action_text: str) -> Tuple[str, float, bool, bool, dict]:
        """
        公共 API：接受模型一次完整回复，内部可能触发 **多个** 原子 env step。

        Args:
            action_text: 模型 chat_request 返回的完整字符串（通常含 `<answer>...</answer>`）。

        Returns:
            (final_obs, total_reward, terminated, truncated, info)
        """
        atomic_actions = self._parse_action_sequence(action_text)
        if len(atomic_actions) == 0:
            # parsing 给了空列表 → 视为一个 invalid 原子动作
            atomic_actions = [None]

        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = ""
        last_info: Dict[str, Any] = {}

        executed = 0
        any_invalid = False
        all_effective = True

        for atomic in atomic_actions:
            obs, r, term, trunc, info = self._step_atomic(atomic)
            # 统一在这里推进计步器，子类不要再自己 ++
            self.current_step += 1
            total_reward += float(r)
            terminated = bool(term)
            truncated = bool(trunc)
            last_obs = obs
            last_info = dict(info) if isinstance(info, dict) else {}

            executed += 1
            if not last_info.get("action_is_valid", True):
                any_invalid = True
            if not last_info.get("action_is_effective", True):
                all_effective = False

            # env 层面总步数封顶
            if self.current_step >= self.max_steps:
                truncated = True

            if terminated or truncated:
                break

        last_info["executed_action_count"] = executed
        last_info["requested_action_count"] = len(atomic_actions)
        last_info["any_invalid_in_sequence"] = any_invalid
        last_info["all_effective_in_sequence"] = all_effective
        return last_obs, total_reward, terminated, truncated, last_info

    # ------------------------------------------------------------------
    # 子类需要实现 / 可覆盖
    # ------------------------------------------------------------------

    @abstractmethod
    def _step_atomic(self, atomic_action: Any) -> Tuple[str, float, bool, bool, dict]:
        """
        执行 **一个** 原子动作。`atomic_action` 是 `_parse_action_sequence` 里单个元素的返回值
        （例如 gym 离散动作 int，或者原始字符串）。**不要**在这里 `current_step += 1`。

        `info` 中建议提供 `action_is_valid`、`action_is_effective` 字段（缺省视为 True）。
        """
        raise NotImplementedError

    def _parse_action_sequence(self, action_text: str) -> List[Any]:
        """
        把一次完整回复解析成原子动作 **列表**。

        默认行为（与旧接口兼容）：把 `action_text` 当作单个动作整体解析，返回 `[parsed]`。
        想支持 "Left || Up || Up" 的子类应当覆盖此方法。
        """
        return [self._parse_action(action_text)]

    def _parse_action(self, action_text: str) -> Any:
        """
        单动作（或单 token）解析。默认 **原样透传字符串**，方便 Bandit / Math 这种
        "在 `_step_atomic` 里才做字符串匹配" 的环境直接复用。

        想返回 gym 整数动作的子类应当覆盖此方法。
        """
        return action_text

    @abstractmethod
    def render(self) -> Any:
        """渲染环境（可选）。"""
        raise NotImplementedError

    def get_valid_actions(self) -> str:
        """
        DEPRECATED（路径 A 重构后不再被 rollout_utils 注入到每轮 user message）：
        论文 prompt 把 action 列表通过 `action_lookup` 自动拼到 system 一次到位。
        子类如有需要仍可保留实现，但不再影响实际 rollout prompt。

        默认返回空串以避免子类 NotImplementedError。
        """
        return ""
