# RAGEN-Ward: 本地深度解耦强化推理大模型训练框架

RAGEN-Ward 是基于 [RAGEN](https://github.com/RAGEN-AI/RAGEN) 论文（*RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*）思想的完全本地重写与解耦优化版本。

本项目针对论文中指出的多轮交互强化学习中的 "Echo Trap"（回声陷阱）现象，复现了 **StarPO (State-Thinking-Actions-Reward Policy Optimization)** 轨迹级优化算法，并通过高度工程化的模块拆分，实现了环境、Agent、RL算法和RAGEN核心框架的完全解耦。并且严格区分了**训练（Training）**与**评估（Evaluation）**流程。

## 🎯 设计目标
- **结构优化**：深度优化原有代码的耦合问题，各个模块可自由替换。
- **可扩展性**：通过统一的 Base 类接口，极其方便地引入新环境、新模型或新RL算法。
- **易调试性**：引入了统一的 Config 系统 (Dataclass) 和完善的日志记录 (`utils/logger.py`)。
- **论文对齐**：原生支持论文中的 Variance-based trajectory filtering (方差轨迹过滤)、多轮历史追踪以及 `<think>` 标签的强制约束与格式惩罚。
- **职责分离**：将训练和推理脚本彻底分离，新增独立的评价和简单测试模块，方便科研人员跑实验。

---

## 🏗 项目架构

项目被精心拆解为以下几个高度独立的核心模块：

```text
ragen_ward/
├── configs/               # ⚙️ 配置模块
│   ├── constants.py       # 常量定义（路径、常量等）
│   └── config.py          # 基于 Dataclass 的各类配置（EnvConfig, AgentConfig, ExperimentConfig, EvalConfig）
├── envs/                  # 🌍 环境模块
│   ├── base_env.py        # 环境基类（定义 reset, step, render 接口）
│   ├── gym_envs.py        # 基于 Gymnasium 的游戏环境封装（CartPole, FrozenLake）
│   └── math_env.py        # 自定义 NLP/逻辑推理环境（已升级为 Countdown 24点游戏）
├── agents/                # 🤖 Agent模块
│   ├── base_agent.py      # Agent 基类（定义 chat_request, get_log_probs 接口）
│   ├── hf_agent.py        # 基于 HuggingFace transformers 的本地白盒模型（训练必须）
│   └── openai_agent.py    # 基于 OpenAI API 的远程黑盒测试模型（仅限评估）
├── rl_algos/              # 🧠 RL 算法模块
│   ├── base_algo.py       # 强化学习算法基类
│   ├── ppo.py             # PPO 落地实现（当前主流）
│   └── grpo.py            # GRPO 落地实现（RAGEN 基线算法，节省显存）
├── ragen_core/            # 🚀 核心框架模块 (RAGEN/StarPO)
│   ├── trajectory_buffer.py # 轨迹回放池（实现了论文核心的方差过滤机制）
│   ├── starpo_trainer.py  # StarPO 训练器，包含 Variance-based filtering
│   └── pure_rl_trainer.py # 纯 RL 训练器基线（无格式约束和方差过滤，用作对比评估）
├── evaluate/              # 📊 评估指标模块
│   └── metrics.py         # 存放论文中提到的评价指标（如 回声陷阱预警、奖励方差等）
├── scripts/               # 🏃‍♂️ 执行脚本模块
│   ├── train.py           # 核心训练脚本
│   ├── evaluate.py        # 纯推理/测试脚本
│   ├── test_env.py        # 环境可用性简单测试脚本
│   └── test_agent.py      # Agent可用性简单测试脚本
└── utils/                 # 🛠 工具模块
    └── logger.py          # 高级日志追踪工具
```

---

## 🚀 核心特性详述

### 1. 严格的训练与评估解耦
为了科研中方便控制单一变量，本项目把流程划分为两部分：
- **训练流程 (`scripts/train.py`)**：启动完整的 RAGEN 框架，调用 `StarPOTrainer`。包括轨迹方差过滤、优势函数计算、以及最终向 LLM 反向传播梯度。
- **评估流程 (`scripts/evaluate.py`)**：剥离训练组件，只用大模型作为 Policy 和环境做推理死磕。此阶段可以使用远程黑盒 API (如 GPT-4) 进行能力对齐，并使用 `evaluate/metrics.py` 进行指标打点。

### 2. 统一的环境接口 (BaseEnv)
所有的环境无论是来自 Gym 还是手写的 NLP 任务，在输出给模型时都统一被转化为**纯文本的自然语言描述**。
- `reset()`: 返回环境初始文本描述。
- `step(action)`: 接受模型的自然语言动作，由环境内部做解析映射，然后返回新的文本观测、奖励以及终止状态。
- `get_valid_actions()`: 将合法的动作空间作为 Prompt 的一部分返回，协助模型理解当前可执行选项。

*注：在本项目中，为了更好地研究多步推理和配合方差过滤机制，原本简单的“20以内四则运算”环境已被升级为 **Countdown（算24点）** 游戏。这要求模型不仅得出答案，还要生成有效的中间表达式。*

### 3. StarPO 训练器 (StarPOTrainer)
RAGEN论文提出的核心。区别于普通的单步 RL 更新，这里采用**轨迹级（Trajectory-Level）**的管理方式。
- **强制思考约束**: 如果模型输出中缺失 `<think>...</think><answer>...</answer>` 结构，将在轨迹结算时给予强烈的负面格式惩罚 (Format Penalty)。
- **轨迹不确定性过滤 (Variance-based Filtering)**: 为防止模型很快陷入局部捷径 (Echo Trap)，在缓冲池更新前，框架会计算对同一 prompt 不同 Rollout 轨迹回报的方差，过滤丢弃方差低（信息量低）的样本，只将具有探索潜力的轨迹喂给 PPO 算法，极大提高了训练稳定性。

---

## 🛠 快速上手

### 环境依赖
请确保你的环境安装了如下依赖：
```bash
pip install torch transformers gymnasium loguru openai numpy
```

### 运行组件测试
在开始大规模训练前，可以测试各组件是否正常工作：
```bash
# 测试环境接口是否工作
python scripts/test_env.py

# 测试本地 HF 模型是否工作
python scripts/test_agent.py
```

### 运行实验
所有的主要执行入口都放到了 `scripts/` 下，配合不同的 `argparse` 参数可以非常灵活地调用。

**发起一场训练：**
```bash
# 使用本地HF模型跑简单的数学推理环境
python scripts/train.py --env math --model Qwen/Qwen2.5-0.5B-Instruct --exp_name exp1_math
```

**对某个模型进行推理评估：**
```bash
# 使用远程API模型评估 FrozenLake 迷宫环境 (需要修改代码中的api key)
python scripts/evaluate.py --env frozenlake --agent openai --episodes 10

# 评估本地刚训练好的模型
python scripts/evaluate.py --env math --agent hf --model_source trained --model_name exp1_math
```

---

## 💡 后续开发指南与实验单一变量控制

本框架的最大优势在于高度解耦。如果你想对比两个模型的表现，只需要修改脚本传参或 `config.py` 中的 `model_name_or_path`。

如果你想测试你的新强化学习算法：
1. 在 `rl_algos/` 下新建 `my_algo.py` 并继承 `BaseRLAlgo`。
2. 实现 `train_step(batch)` 的梯度计算与更新逻辑。
3. 在 `scripts/train.py` 中将实例化的 `algo` 替换为你的新算法。

对于新增环境也同理，只要继承自 `envs/base_env.py`，你就可以将任何现实世界的复杂任务快速接入到这套大模型强化学习工作流中来。

