from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class EnvConfig:
    """环境配置类 / Environment Configuration"""
    env_name: str = "FrozenLake-v1"  # 环境名称，如 FrozenLake-v1, CartPole-v1, MathProblem
    max_steps: int = 100             # 每个episode的最大步数
    kwargs: Dict[str, Any] = field(default_factory=dict) # 其他环境相关的参数

@dataclass
class AgentConfig:
    """Agent配置类 / Agent Configuration"""
    agent_type: str = "hf"           # "hf" 为本地HuggingFace模型, "api" 为远程API调用
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct" # 模型路径或名称
    api_key: Optional[str] = None    # API key (仅对api类型有效)
    base_url: Optional[str] = None   # API base url (仅对api类型有效)
    temperature: float = 0.7         # 采样温度
    max_new_tokens: int = 512        # 最大生成token数
    system_prompt: str = "You are a helpful reinforcement learning agent."
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RLAlgoConfig:
    """强化学习算法配置类 / RL Algorithm Configuration"""
    algo_name: str = "ppo"           # 算法名称: ppo, grpo
    learning_rate: float = 1e-5      # 学习率
    gamma: float = 0.99              # 折扣因子
    batch_size: int = 32             # 批次大小
    
    # PPO / GRPO 特有参数
    ppo_epochs: int = 4              # 更新迭代次数
    clip_ratio: float = 0.2          # 截断范围
    vf_coef: float = 0.5             # Value Loss 系数 (PPO专属)
    ent_coef: float = 0.01           # 熵正则化系数
    target_kl: Optional[float] = 0.05# 提前停止的 KL 散度阈值
    
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RagenConfig:
    """RAGEN框架配置类 / RAGEN Framework Configuration"""
    mode: str = "fast"               # "fast" 或 "slow" 等不同模式的实现
    num_rollouts: int = 16           # 每次更新采样的轨迹数量 (StarPO特性)
    use_format_reward: bool = True   # 是否使用格式化奖励 (例如奖励<think>标签)
    format_penalty: float = -0.1     # 格式错误时的惩罚
    variance_filter_ratio: float = 0.25 # 方差过滤比例，只保留高方差样本
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    """整体实验配置类 / Overall Experiment Configuration"""
    exp_name: str = "default_exp"    # 实验名称
    seed: int = 42                   # 随机种子
    total_training_steps: int = 1000 # 总训练步数
    eval_interval: int = 50          # 评估间隔
    env_config: EnvConfig = field(default_factory=EnvConfig)
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    rl_algo_config: RLAlgoConfig = field(default_factory=RLAlgoConfig)
    ragen_config: RagenConfig = field(default_factory=RagenConfig)

@dataclass
class EvalConfig:
    """评估实验配置类 / Evaluation Configuration"""
    exp_name: str = "eval_default"   # 实验名称
    seed: int = 42                   # 随机种子
    episodes: int = 10               # 评估的总回合数
    env_config: EnvConfig = field(default_factory=EnvConfig)
    agent_config: AgentConfig = field(default_factory=AgentConfig)
