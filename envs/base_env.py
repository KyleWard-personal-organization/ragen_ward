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
- **可以覆盖** `_parse_action(text)`、`_parse_action_sequence(text)`、`get_env_instruction()`。

## 单轮任务（Bandit / Math 等）

单轮任务的 `_step_atomic` 自己会立即返回 `terminated=True`，循环会在第一个原子动作后就 break，所以它们不需要关心"多动作"，默认 sequence 长度为 1 的行为与旧接口完全一致。
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, List, Optional


class BaseEnv(ABC):

    # ------------------------------------------------------------------
    # Agent-facing system prompt（环境 → LLM 的"人设 + 格式契约"）
    # ------------------------------------------------------------------
    # System prompt 本质上是"任务/环境相关的 agent 角色设定 + 输出格式约束"，
    # 所以 **应该由环境类持有**，而不是作为 agent/CLI 参数（换一个 env 就
    # 必须同步改 CLI 会非常容易漂移）。
    #
    # - 这里提供的是**兜底默认值**：任何未重写该属性的子类都会继承它。
    # - 子类建议通过 class attribute 形式覆盖即可：
    #       class FooEnv(BaseEnv):
    #           agent_system_prompt = "You are a Foo expert ..."
    # - rollout 工具（ragen_core.rollout_utils.rollout_one_trajectory）会
    #   直接从 `env.agent_system_prompt` 读取，拼到 messages[0] 的 system
    #   message。get_env_instruction()（每条 traj 首个 user message）仍是
    #   另一层正交的"环境玩法说明 + 多动作协议示例"。
    agent_system_prompt: str = "You are a helpful reinforcement learning agent."

    def __init__(self, config: Any):
        self.config = config
        self.current_step = 0
        # 必填字段；缺失直接 AttributeError，不再静默兜底。
        self.max_steps = config.max_steps

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

    def get_env_instruction(self) -> str:
        """
        返回一段给模型看的"环境玩法说明"（RAGEN 风格），会被 trainer 注入到
        第一个 user message 的开头。默认空串表示不注入。
        """
        return ""

    @abstractmethod
    def render(self) -> Any:
        """渲染环境（可选）。"""
        raise NotImplementedError

    @abstractmethod
    def get_valid_actions(self) -> str:
        """当前有效动作的文本描述。每轮 user message 都会带上它。"""
        raise NotImplementedError
