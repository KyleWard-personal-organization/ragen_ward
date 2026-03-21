from typing import Optional, Tuple, Any
from .base_env import BaseEnv


class BanditEnv(BaseEnv):
    """
    多臂老虎机 环境
    RAGEN-main 中的探索与利用 (Exploration vs. Exploitation) 文本推理环境。
    不依赖 gym，直接继承 BaseEnv 实现。
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.name_b = None
        self.name_a = None
        self.is_a_high_risk = None
        self.lo_arm_score = getattr(config, 'lo_arm_score', 10)
        self.hi_arm_hiscore = getattr(config, 'hi_arm_hiscore', 100)
        self.hi_arm_loscore = getattr(config, 'hi_arm_loscore', 0)
        self.hi_arm_prob = getattr(config, 'hi_arm_hiscore_prob', 0.2)

        self.arm_names = [("Safe&Steady", "HighRisk&HighReward")]

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict]:
        super().reset(seed=seed, **kwargs)
        import random
        if seed is not None:
            random.seed(seed)

        # 随机分配机器 A 和 B
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

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        super().step(action)
        import random

        # 解析动作
        import re
        match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
        act_str = match.group(1).strip() if match else action.strip()

        chose_a = self.name_a in act_str
        chose_b = self.name_b in act_str

        if not (chose_a ^ chose_b):
            return "Invalid action. Choose exactly one arm.", -1.0, True, False, {"error": "Invalid action format."}

        chose_high_risk = (chose_a and self.is_a_high_risk) or (chose_b and not self.is_a_high_risk)

        if chose_high_risk:
            reward = self.hi_arm_hiscore if random.random() < self.hi_arm_prob else self.hi_arm_loscore
        else:
            reward = self.lo_arm_score

        chosen_name = self.name_a if chose_a else self.name_b
        obs = f"You pulled {chosen_name} and got {reward} points."

        return obs, float(reward), True, False, {"success": chose_high_risk}

    def render(self) -> Any:
        return "Bandit Env - Text Based"

    def get_valid_actions(self) -> str:
        return f"Based on the symbolic meaning, which arm do you think gives higher expected rewards? Output <answer> {self.name_a} </answer> or <answer> {self.name_b} </answer>."
