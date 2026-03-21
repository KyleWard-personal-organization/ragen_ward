import random
import re
from typing import Optional, Tuple, Any
from .base_env import BaseEnv

class MathEnv(BaseEnv):
    """
    Countdown (算24点) 环境 / Countdown Game Environment
    为了配合论文中的“多步推理(Multi-step reasoning)”和方差过滤，
    普通的20以内四则运算太简单。这里我们将 MathEnv 升级为 Countdown 游戏。
    游戏规则：给出几个数字和一个目标值，模型必须通过组合这些数字（加减乘除）来计算出目标值。
    这种任务具备搜索空间大、需要多步中间推导的特点，非常适合 StarPO 的强化学习场景。
    """
    def __init__(self, config: Any):
        super().__init__(config)
        self.numbers = []
        self.target = 0
        self.problem_str = ""

    @staticmethod
    def _generate_solvable_problem() -> Tuple[list, int]:
        """简单的随机生成可解的算式"""
        # 为保证一定有解，我们随机生成一个表达式，并计算其结果
        nums = [random.randint(1, 10) for _ in range(4)]
        ops = [random.choice(['+', '-', '*']) for _ in range(3)]
        
        # 为了避免除法产生小数，我们在生成环境时只使用加减乘。但Agent回答时可以使用除法。
        expr = f"{nums[0]} {ops[0]} {nums[1]} {ops[1]} {nums[2]} {ops[2]} {nums[3]}"
        target = eval(expr)
        
        # 打乱数字显示顺序，增加难度
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
        """
        验证表达式是否：
        1. 仅使用了给定的数字（且每个数字使用次数一致）
        2. 计算结果等于 target
        """
        # 移除非法字符
        clean_expr = re.sub(r'[^0-9+\-*/(). ]', '', expr)
        if not clean_expr.strip():
            return False, "No valid mathematical expression found."
            
        # 提取使用的所有数字
        used_nums = [int(n) for n in re.findall(r'\d+', clean_expr)]
        if sorted(used_nums) != sorted(self.numbers):
            return False, f"You must use exactly these numbers: {sorted(self.numbers)}. You used: {sorted(used_nums)}."
            
        try:
            # 评估结果
            result = eval(clean_expr)
            # 处理浮点数误差
            if abs(result - self.target) < 1e-5:
                return True, "You are right!"
            else:
                return False, f"Expression evaluates to {result}, but target is {self.target}."
        except Exception as e:
            return False, f"Invalid expression syntax: {str(e)}"

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        super().step(action)
        
        # 模型被期望在 <answer> 标签中输出最终的公式，或者在回答的最后提供公式。
        # 简单起见，我们提取它文本中可能像是等式或表达式的最后一行。
        # 这里为了兼容性，假设 agent 将答案包装在类似 "The answer is: (1+2)*3" 这样的结构中，
        # 或者直接就是表达式。
        
        # 抽取可能包含在 <answer> 标签里或者是最后一行的内容
        answer_match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
        if answer_match:
            candidate_expr = answer_match.group(1).strip()
        else:
            # 如果没有标签，取最后一行
            candidate_expr = action.strip().split('\n')[-1].strip()
            
        # 去除等号前面的解释性文字，例如 "So the expression is 3 + 4 * ..."
        if "=" in candidate_expr:
            candidate_expr = candidate_expr.split("=")[0]
            
        is_correct, msg = self._check_answer(candidate_expr)
        
        if is_correct:
            reward = 1.0
            terminated = True
            obs = "Correct! " + msg
        else:
            reward = -1.0
            terminated = True # 这种单轮问答任务一旦答错就直接结束
            obs = "Wrong! " + msg
            
        truncated = False
        return obs, reward, terminated, truncated, {"target": self.target, "agent_expr": candidate_expr}

    def render(self) -> Any:
        print(self.problem_str)

    def get_valid_actions(self) -> str:
        return "Please output a valid mathematical expression using the given numbers inside an <answer> tag."
