"""
多臂老虎机 环境 / Bandit Environment
-------------------------------------
RAGEN-main 中的探索与利用 (Exploration vs. Exploitation) 文本推理环境。

这是一个"一拉决胜负"的单轮任务：模型选完 arm 就立即 terminated=True。
所以它不需要 `||` 多动作序列，复用 BaseEnv 默认的"sequence 长度 = 1"路径即可。
"""

import random
import re
from typing import Any, Optional, Tuple

from .base_env import BaseEnv


class BanditEnv(BaseEnv):
    """
    路径 A 论文对齐：Bandit 环境的特殊性 ——
    - **arm 名字每个 episode 都不同**（动态生成），所以无法做 class-level 的 action_lookup。
    - 论文 ``config/envs.yaml::Bandit`` 把 ``env_instruction`` 设成 ``""``：
      具体的 prompt（包含 arm 名字）由 ``reset()`` 返回的 obs 直接给出，跟着第一轮
      user message 进 prompt，跟 system 解耦。
    我们这里完全照搬该约定。
    """

    # ---- 路径 A：论文风格 prompt 五件套 ----
    env_instruction = ""  # 论文一致：Bandit 不在 system 里说任务，全在 reset obs 里说
    grid_vocab = None     # 文本 obs，无网格符号
    action_lookup = None  # arm 名字动态生成，无法 class-level 列出
    max_actions_per_traj = 1
    max_response_tokens = 100

    def __init__(self, config: Any):
        super().__init__(config)
        self.name_b: Optional[str] = None
        self.name_a: Optional[str] = None
        self.is_a_high_risk: Optional[bool] = None
        self.lo_arm_score = getattr(config, 'lo_arm_score', 10)
        self.hi_arm_hiscore = getattr(config, 'hi_arm_hiscore', 100)
        self.hi_arm_loscore = getattr(config, 'hi_arm_loscore', 0)
        self.hi_arm_prob = getattr(config, 'hi_arm_hiscore_prob', 0.2)

        self.arm_names = [("Safe&Steady", "HighRisk&HighReward")]

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict]:
        super().reset(seed=seed, **kwargs)
        if seed is not None:
            random.seed(seed)

        self.is_a_high_risk = random.choice([True, False])
        name_safe, name_risk = self.arm_names[0]

        self.name_a = name_risk if self.is_a_high_risk else name_safe
        self.name_b = name_safe if self.is_a_high_risk else name_risk

        obs = (
            f"You are playing a bandit game. Goal: Maximize your total reward by choosing which arm to pull.\n"
            f"Game Rules:\n"
            f"1. There are 2 arms, named '{self.name_a}' and '{self.name_b}'.\n"
            f"2. Each arm has its own reward distribution, related to their names.\n"
            f"3. Analyze the symbolic meaning of each arm's name to guess how their reward distribution might behave.\n"
        )
        return obs, {}

    def _step_atomic(self, atomic_action: Any) -> Tuple[str, float, bool, bool, dict]:
        """
        `atomic_action` 由默认的 `_parse_action` 透传得到，就是完整的 `action_text`。
        内部仍做一次 `<answer>` 抽取。
        """
        action_text = atomic_action if isinstance(atomic_action, str) else ""
        match = re.search(r'<answer>(.*?)</answer>', action_text, re.DOTALL)
        act_str = match.group(1).strip() if match else action_text.strip()

        chose_a = self.name_a in act_str if self.name_a else False
        chose_b = self.name_b in act_str if self.name_b else False

        if not (chose_a ^ chose_b):
            return (
                "Invalid action. Choose exactly one arm.",
                -1.0,
                True,
                False,
                {
                    "error": "Invalid action format.",
                    "action_is_valid": False,
                    "action_is_effective": False,
                },
            )

        chose_high_risk = (chose_a and self.is_a_high_risk) or (chose_b and not self.is_a_high_risk)

        if chose_high_risk:
            reward = self.hi_arm_hiscore if random.random() < self.hi_arm_prob else self.hi_arm_loscore
        else:
            reward = self.lo_arm_score

        chosen_name = self.name_a if chose_a else self.name_b
        obs = f"You pulled {chosen_name} and got {reward} points."
        return obs, float(reward), True, False, {"success": bool(chose_high_risk), "action_is_valid": True}

    def render(self) -> Any:
        return "Bandit Env - Text Based"
