from .base_algo import BaseRLAlgo
from .ppo import PPO
from .grpo import GRPO

def make_algo(config, agent):
    """
    根据配置中的 algo_name 实例化对应的强化学习算法
    """
    algo_name = config.algo_name.lower()
    if algo_name == "ppo":
        return PPO(config, agent)
    elif algo_name == "grpo":
        return GRPO(config, agent)
    else:
        raise ValueError(f"Unknown algorithm name: {config.algo_name}. Please check your RLAlgoConfig.")

__all__ = [
    "BaseRLAlgo",
    "PPO",
    "GRPO",
    "make_algo"
]
