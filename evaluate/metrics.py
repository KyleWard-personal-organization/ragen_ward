import numpy as np
from typing import List, Dict

class EvaluatorMetrics:
    """
    RAGEN 论文中提到的评估指标统计工具。
    主要用于在评估脚本（evaluation）或训练日志中追踪和汇总指标。
    """
    def __init__(self):
        self.success_rates = []
        self.rewards = []
        self.trajectory_lengths = []
        
    def add_episode(self, reward: float, success: bool, length: int):
        self.rewards.append(reward)
        self.success_rates.append(1.0 if success else 0.0)
        self.trajectory_lengths.append(length)
        
    def summary(self) -> Dict[str, float]:
        if not self.rewards:
            return {}
            
        return {
            "eval/success_rate": float(np.mean(self.success_rates)),
            "eval/avg_reward": float(np.mean(self.rewards)),
            "eval/avg_trajectory_length": float(np.mean(self.trajectory_lengths))
        }

def compute_reward_variance(rewards: List[float]) -> float:
    """
    辅助函数：计算奖励方差（StarPO-S 核心过滤指标）
    """
    if not rewards:
        return 0.0
    return float(np.var(rewards))

def check_echo_trap_signs(reward_variances: List[float], entropies: List[float]) -> bool:
    """
    检测是否陷入“回声陷阱”(Echo Trap)：
    当奖励方差骤降且输出熵急剧减少时，往往预示着模型正在陷入局部捷径。
    """
    if len(reward_variances) < 5:
        return False
        
    var_trend = np.mean(reward_variances[-3:]) < np.mean(reward_variances[:3]) * 0.1
    entropy_trend = np.mean(entropies[-3:]) < np.mean(entropies[:3]) * 0.5
    
    return var_trend and entropy_trend
