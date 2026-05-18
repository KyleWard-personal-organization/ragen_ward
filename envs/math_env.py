"""
Countdown (算 24 点) 环境 / Countdown Game Environment
------------------------------------------------------
单轮问答任务：给几个数字和目标值，模型输出表达式。答对 +1，答错 -1，一步即 terminated=True。

单轮任务不需要 `||` 多动作语义，复用 BaseEnv 默认的"sequence 长度 = 1"路径即可。
"""

import random
import re
from typing import Any, List, Optional, Tuple

from .base_env import BaseEnv


class MathEnv(BaseEnv):
    """
    路径 A 论文对齐：Countdown 单轮一次决策 → max_actions_per_traj=1，
    env_instruction 一句话级别（对齐 RAGEN ``config/envs.yaml::Countdown`` 风格）。
    grid_vocab/action_lookup=None：表达式没有固定符号集合，全靠模型自由生成。
    """

    # ---- 路径 A：论文风格 prompt 五件套 ----
    env_instruction = (
        "You are solving the Countdown puzzle. Combine the given numbers with "
        "+, -, *, /, and parentheses so the expression evaluates exactly to the "
        "target. Use each number exactly once. Output only the final expression "
        "inside <answer>...</answer>."
    )
    grid_vocab = None
    action_lookup = None
    max_actions_per_traj = 1
    max_response_tokens = 100

    def __init__(self, config: Any):
        super().__init__(config)
        self.numbers: List[int] = []
        self.target: int = 0
        self.problem_str = ""

    @staticmethod
    def _generate_solvable_problem() -> Tuple[List[int], int]:
        nums = [random.randint(1, 10) for _ in range(4)]
        ops = [random.choice(['+', '-', '*']) for _ in range(3)]
        # 为避免整数除法分数问题，生成时只用加减乘；agent 回答可用除法
        expr = f"{nums[0]} {ops[0]} {nums[1]} {ops[1]} {nums[2]} {ops[2]} {nums[3]}"
        target = eval(expr)
        random.shuffle(nums)
        return nums, target

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict]:
        super().reset(seed=seed, **kwargs)
        if seed is not None:
            random.seed(seed)

        self.numbers, self.target = self._generate_solvable_problem()

        nums_str = ", ".join(map(str, self.numbers))
        self.problem_str = (
            f"Using the numbers [{nums_str}], create a mathematical expression "
            f"that evaluates to the target value: {self.target}. "
            "You must use all numbers exactly once. You may use +, -, *, /, and parentheses ()."
        )
        return f"Problem: {self.problem_str}", {}

    def _check_answer(self, expr: str) -> Tuple[bool, str]:
        """1) 使用且仅使用给定数字；2) 结果等于 target。"""
        clean_expr = re.sub(r'[^0-9+\-*/(). ]', '', expr)
        if not clean_expr.strip():
            return False, "No valid mathematical expression found."

        used_nums = [int(n) for n in re.findall(r'\d+', clean_expr)]
        if sorted(used_nums) != sorted(self.numbers):
            return False, (
                f"You must use exactly these numbers: {sorted(self.numbers)}. "
                f"You used: {sorted(used_nums)}."
            )

        try:
            result = eval(clean_expr)
            if abs(result - self.target) < 1e-5:
                return True, "You are right!"
            return False, f"Expression evaluates to {result}, but target is {self.target}."
        except Exception as e:
            return False, f"Invalid expression syntax: {str(e)}"

    def _step_atomic(self, atomic_action: Any) -> Tuple[str, float, bool, bool, dict]:
        """
        `atomic_action` 由默认 `_parse_action` 透传得到完整的 `action_text`。
        """
        action_text = atomic_action if isinstance(atomic_action, str) else ""

        answer_match = re.search(r'<answer>(.*?)</answer>', action_text, re.DOTALL)
        if answer_match:
            candidate_expr = answer_match.group(1).strip()
        else:
            # 没有标签 → 回退到最后一行
            candidate_expr = action_text.strip().split('\n')[-1].strip()

        # 去除等号右侧的说明性内容：如 "3 + 4 = 7"
        if "=" in candidate_expr:
            candidate_expr = candidate_expr.split("=")[0]

        is_correct, msg = self._check_answer(candidate_expr)

        if is_correct:
            reward = 1.0
            obs = "Correct! " + msg
        else:
            reward = -1.0
            obs = "Wrong! " + msg

        # 单轮问答：不管对错都 terminated=True
        return (
            obs,
            reward,
            True,
            False,
            {
                "target": self.target,
                "agent_expr": candidate_expr,
                "is_success": bool(is_correct),
                "action_is_valid": True,
            },
        )

    def render(self) -> Any:
        print(self.problem_str)
