from .base_env import BaseEnv
from .gym_envs import GymEnvWrapper, CartPoleEnv, FrozenLakeEnv, SokobanEnv
from .bandit_env import BanditEnv
from .math_env import MathEnv

def make_env(config):
    """
    根据配置中的 env_name 实例化对应的环境
    """
    env_name = config.env_name.lower()
    if "cartpole" in env_name:
        return CartPoleEnv(config)
    elif "frozenlake" in env_name:
        return FrozenLakeEnv(config)
    elif "sokoban" in env_name:
        return SokobanEnv(config)
    elif "bandit" in env_name:
        return BanditEnv(config)
    elif "math" in env_name or "countdown" in env_name:
        return MathEnv(config)
    else:
        raise ValueError(f"Unknown environment name: {config.env_name}. Please check your EnvConfig.")

__all__ = [
    "BaseEnv",
    "GymEnvWrapper",
    "CartPoleEnv",
    "FrozenLakeEnv",
    "SokobanEnv",
    "BanditEnv",
    "MathEnv",
    "make_env"
]
