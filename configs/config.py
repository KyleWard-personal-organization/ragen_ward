"""
配置模块 / Configuration Module
-----------------------------------
本文件使用 dataclass 定义实验所需的所有配置对象：
    EnvConfig         -> envs/
    AgentConfig       -> agents/
    RLAlgoConfig      -> rl_algos/
    RagenConfig       -> ragen_core/ (StarPO 框架)
    ExperimentConfig  -> 训练入口 scripts/train.py 使用的总配置
    EvalConfig        -> 评估入口 scripts/evaluate.py 使用的总配置

=======================================================================
⚠️ 默认值来源的唯一性约定（Single Source of Truth）
=======================================================================
这里定义的所有字段（除了类型语义上的 Optional 字段）都**不**携带默认值，
强制调用方在构造时显式传入。默认值统一由 ``scripts/train.py`` 中的 argparse
集中管理，再通过 ``build_config`` 一次性注入。

这样带来的保证：
1. 修改超参只改 argparse 一处，不会出现 dataclass / argparse / getattr
   三处同名却不同值的"隐形分叉"。
2. 下游消费（trainer / algo / env）里不再使用 ``getattr(config, 'xxx', default)``，
   而是直接属性访问 ``config.xxx``——如果字段缺失立即报错，避免"参数没按预期
   跑通但跑出看似正常的结果"这种致命的静默 bug。
3. 真正语义上允许"缺省即未启用"的字段（如 ``api_key`` / ``target_kl``），
   保留 ``Optional[X] = None`` 作为显式的"未设置"标记。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class EnvConfig:
    """环境配置类 / Environment Configuration。"""
    env_name: str                                      # FrozenLake-v1 / CartPole-v1 / math / bandit / sokoban 等
    max_steps: int                                     # 每个 episode 的最大步数
    kwargs: Dict[str, Any] = field(default_factory=dict)  # 环境内部动态扩展字段（留给具体子类自己消化）


@dataclass
class AgentConfig:
    """
    Agent 配置类 / Agent Configuration。

    .. note::
        这里**故意不包含** ``system_prompt`` 字段。
        System prompt 本质上是"环境任务的 agent 人设 + 输出格式契约"，属于环境侧
        的责任（见 ``envs/base_env.py::BaseEnv.agent_system_prompt`` 和各子类的
        覆盖）。Agent 本身是"通用 LLM 客户端"，不应与具体任务 prompt 耦合：否则
        每换一个 env 都要同步改 AgentConfig，极易漂移。
        ``rollout_utils.rollout_one_trajectory`` 会直接从 ``env.agent_system_prompt``
        读取该字符串，填到 messages[0] 的 system message。
    """
    agent_type: str                                    # "hf" 本地模型 / "openai" 远程 API
    model_name_or_path: str                            # HF 模型名或本地路径
    temperature: float                                 # 采样温度
    max_new_tokens: int                                # 单次生成的最大 token 数
    api_key: Optional[str] = None                      # 仅 openai agent 使用，缺省即未设置
    base_url: Optional[str] = None                     # 仅 openai agent 使用，缺省即未设置
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLAlgoConfig:
    """
    强化学习算法配置类 / RL Algorithm Configuration。

    PPO / GRPO 共享的全部超参；``bi_level_gae`` + ``high_level_gamma`` 用来对齐
    RAGEN 论文的 bi-level advantage estimation；``target_kl`` 为 Optional，None
    表示不启用提前停止。

    ``use_ref`` 与 RAGEN 原项目 ``actor_rollout_ref.actor.use_ref`` 语义一致：
    True 时构造一份冻结的 ref_model 用于 KL 正则；False 时完全不建 ref_model，
    强制 ``kl_coef=0``，用"放弃 KL 锚"换取约 1GB 的 VRAM。

    ``optimizer`` 控制优化器类型，默认走 ``adamw8bit``（bitsandbytes 8-bit state）
    以节省约 2GB 的 optimizer state 显存；复现论文时可显式切回 ``adamw``。
    """
    algo_name: str                                     # "ppo" / "grpo"
    learning_rate: float                               # AdamW 学习率
    gamma: float                                       # token 级折扣因子
    lam: float                                         # GAE λ
    bi_level_gae: bool                                 # 是否开启 turn 级 + token 级 双层 GAE
    high_level_gamma: float                            # turn 级折扣因子 (仅 bi_level_gae=True 时有效)
    ppo_epochs: int                                    # 每批数据上的 PPO/GRPO 更新轮数
    micro_batch_size: int                              # 每次 forward/backward 的样本数（VRAM 峰值由此决定）
    gradient_accumulation: int                         # 梯度累积步数；1 即关闭累积（等同于 micro_batch 立即 step）。
    #                                                  # 等效的 "mini_batch_size" = micro_batch_size * gradient_accumulation，
    #                                                  # 乘法定义避免除不尽问题；对齐 RAGEN 论文的
    #                                                  # ppo_mini_batch_size = micro_batch_size_per_gpu * grad_accum。
    clip_ratio: float                                  # PPO-Clip 截断范围
    vf_coef: float                                     # Critic loss 系数 (PPO 专属)
    ent_coef: float                                    # 熵正则化系数
    kl_coef: float                                     # KL 散度正则化系数
    max_seq_length: int                                # 拼接整条 trajectory 后的最大 token 长度
    use_ref: bool                                      # 是否创建 ref_model + 计算 KL (False 时强制 kl_coef=0)
    optimizer: str                                     # {"adamw", "adamw8bit", "adafactor"}
    target_kl: Optional[float] = None                  # 提前停止的 KL 阈值；缺省即不启用
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RagenConfig:
    """RAGEN / StarPO 框架配置类 / RAGEN Framework Configuration。"""
    mode: str                                          # "fast" / "slow"，预留不同实现切换
    num_rollouts: int                                  # 每个 prompt 采样的轨迹数（GRPO group size）
    use_format_reward: bool                            # 是否启用 <think>/<answer> 格式奖励/惩罚
    format_penalty: float                              # 格式错误时的额外负奖励
    variance_filter_ratio: float                       # 方差过滤保留比例（StarPO-S 稳定化）
    max_turn: int                                      # 多轮交互最大 turn 数（单轮任务此项无效）
    prompt_batch_size: int                             # 每一步训练用多少个不同 prompt
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """整体实验配置类 / Overall Experiment Configuration（用于训练）。"""
    exp_name: str                                      # 实验名（用于日志文件、checkpoint 命名）
    seed: int                                          # 随机种子
    total_training_steps: int                          # 训练总步数 (每一步 = 一次 rollout + 一次 RL 更新)
    eval_interval: int                                 # 验证间隔步数 (每多少步做一次 evaluate)
    eval_episodes: int                                 # 每次验证跑的 episode 数
    save_interval: int                                 # 每多少步保存一次 checkpoint
    env_config: EnvConfig                              # 环境配置 (必填)
    agent_config: AgentConfig                          # Agent 配置 (必填)
    rl_algo_config: RLAlgoConfig                       # RL 算法配置 (必填)
    ragen_config: RagenConfig                          # RAGEN 框架配置 (必填)


@dataclass
class EvalConfig:
    """评估实验配置类 / Evaluation Configuration（用于 scripts/evaluate.py）。"""
    exp_name: str
    seed: int
    episodes: int
    env_config: EnvConfig
    agent_config: AgentConfig
