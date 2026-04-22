# RAGEN-Ward: 本地深度解耦强化推理大模型训练框架

RAGEN-Ward 是基于 [RAGEN](https://github.com/RAGEN-AI/RAGEN) 论文（*RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*）思想的完全本地重写与解耦优化版本。

本项目针对论文中指出的多轮交互强化学习中的 "Echo Trap"（回声陷阱）现象，复现了 **StarPO (State-Thinking-Actions-Reward Policy Optimization)** 轨迹级优化算法，并通过高度工程化的模块拆分，实现了环境、Agent、RL 算法和 RAGEN 核心框架的完全解耦。并且严格区分了**训练（Training）**与**评估（Evaluation）**流程。

## 🎯 设计目标
- **结构优化**：深度优化原有代码的耦合问题，各个模块可自由替换。
- **可扩展性**：通过统一的 Base 类接口，极其方便地引入新环境、新模型或新 RL 算法。
- **易调试性**：引入了统一的 Config 系统 (Dataclass) 和完善的日志记录 (`utils/logger.py`)，加入训练指标追踪器 (`utils/tracker.py`) 自动把所有指标写 JSONL 便于事后分析。
- **论文对齐**：原生支持 **bi-level GAE**（turn 级 + token 级双层 advantage 估计）、Variance-based trajectory filtering、多轮历史追踪，以及 `<think>` 标签格式惩罚。
- **职责分离**：将训练和推理脚本彻底分离，新增独立的评价和简单测试模块，方便科研人员跑实验。

---

## 🏗 项目架构

```text
ragen_ward/
├── configs/               # ⚙️ 配置模块
│   ├── constants.py       # 常量定义（路径、常量等）
│   └── config.py          # 基于 Dataclass 的各类配置（EnvConfig, AgentConfig, RLAlgoConfig, RagenConfig, ExperimentConfig, EvalConfig）
├── envs/                  # 🌍 环境模块
│   ├── base_env.py        # 环境基类（定义 reset/step/render 接口）
│   ├── gym_envs.py        # 基于 Gymnasium 的游戏环境封装（CartPole/FrozenLake/Sokoban）
│   ├── bandit_env.py      # 自定义 Bandit 环境
│   └── math_env.py        # 自定义 Countdown（算 24 点）环境
├── agents/                # 🤖 Agent 模块
│   ├── base_agent.py      # Agent 基类（定义 chat_request/get_log_probs 接口）
│   ├── hf_agent.py        # 基于 HuggingFace transformers 的本地白盒模型（训练必须）
│   └── openai_agent.py    # 基于 OpenAI API 的远程黑盒模型（仅限评估）
├── rl_algos/              # 🧠 RL 算法模块
│   ├── base_algo.py       # 强化学习算法基类
│   ├── gae_utils.py       # **GAE / bi-level GAE 工具（对齐 RAGEN-main verl）**
│   ├── trajectory_utils.py# **轨迹 tokenize / collate / 前向 的公共工具**
│   ├── ppo.py             # PPO（带 Critic、bi-level GAE、PPO-Clip、KL）
│   └── grpo.py            # GRPO（actor-only、组相对 advantage、PPO-Clip、KL）
├── ragen_core/            # 🚀 核心框架模块 (RAGEN/StarPO)
│   ├── trajectory_buffer.py # 轨迹回放池（实现论文核心的方差过滤机制）
│   ├── starpo_trainer.py  # StarPO 训练器（含 format reward + variance filter + evaluate + tracker）
│   └── pure_rl_trainer.py # 纯 RL 训练器基线（对照组）
├── evaluation/            # 📊 评估指标模块（改名自 evaluate/ 以避免与 HF 同名包冲突）
│   ├── __init__.py
│   └── metrics.py         # 评估指标（success_rate / avg_reward / echo_trap 预警等）
├── scripts/               # 🏃‍♂️ 执行脚本模块
│   ├── train.py           # 核心训练脚本（完整 argparse 暴露所有核心超参）
│   ├── evaluate.py        # 推理/测试脚本
│   ├── test_env.py        # 环境可用性测试脚本
│   └── test_agent.py      # Agent 可用性测试脚本
└── utils/                 # 🛠 工具模块
    ├── logger.py          # loguru 日志
    ├── tracker.py         # **训练指标追踪器（loguru + JSONL + 可选 wandb）**
    └── basic_utils.py     # 通用辅助函数
```

---

## 🚀 核心特性详述

### 1. 严格的训练与评估解耦
- **训练流程 (`scripts/train.py`)**：完整 StarPO/PureRL 框架 + argparse 暴露所有超参。
- **评估流程 (`scripts/evaluate.py`)**：剥离训练组件，只做在线推理，可使用远程 API 做能力对齐。

### 2. 统一的环境接口 (BaseEnv) + **多动作序列（RAGEN 对齐）**

所有环境对 LLM 都是 **纯文本 in / 纯文本 out**，并按 RAGEN 论文约定支持 **一次回复多个原子动作**：

- `reset()` → 初始文本描述 + info dict
- `step(action_text)` → (final_obs, total_reward, terminated, truncated, info)

`step` 的内部做三件事：
1. `_parse_action_sequence(action_text)` 把一次 LLM 回复切成原子动作列表（例如 `<answer>Right || Down || Down</answer>` → `[Right, Down, Down]`）。
2. 依次调用 `_step_atomic(atom)` 直到 terminated / truncated / 序列跑完；`current_step` 按原子动作计数，与 `--max_env_steps` 做 truncation 比较。
3. reward 累加、obs 取最后一步、info 合并（附带 `executed_action_count` / `requested_action_count` 便于诊断）。

这与 RAGEN-main `config/envs.yaml` 的 `max_actions_per_traj=10` + `env_instruction` 示例（如 `<answer>Left || Up || Up</answer>`）**在语义与协议层完全对齐**。

单轮任务（Bandit / Math）继承默认行为 —— 一次回复解析成 1 个 token，`_step_atomic` 立即 `terminated=True`，循环一轮就退出。

每个环境另有一个 `get_env_instruction()`，返回 RAGEN 风格的"玩法 + 多动作答题示例"。trainer **只在第一轮 user message** 注入它一次（避免 context 膨胀）。

### 3. StarPO 训练器 (StarPOTrainer)
区别于单步 RL，本训练器采用**轨迹级（Trajectory-Level）**的管理方式：
- **格式惩罚 (Format Penalty)**：缺失 `<think>…</think><answer>…</answer>` 结构时对每一步施加负奖励。
- **方差过滤 (Variance-based Filtering)**：对同一 prompt 的多条 rollout 计算回报方差，只保留 top-k 高方差组喂给 PPO/GRPO，缓解 Echo Trap。
- **自动评估**：每 `--eval_interval` 步跑 `--eval_episodes` 轮在线推理，记录 avg_reward/success_rate/reward_variance。
- **Echo Trap 预警**：基于 reward_variance 和 entropy 的滑动窗口检测。

### 4. bi-level GAE（复现 RAGEN-main 论文核心）
多轮 agent 场景下，每条 trajectory 会被拼成一条长 token 序列：
```text
[system prompt] [user obs_0] [assistant response_0] [user obs_1] [assistant response_1] ...
```
其中 `token_level_rewards` 只在每个 response 末尾的那个 token 上给值，其余为 0。
`bi-level GAE` 分两阶段计算 advantage：
1. **Turn 级 GAE**：只在 response 末尾位置上用 `high_level_gamma=0.95` 做一次 GAE。
2. **Token 级 GAE**：把每个 turn 的 `advantages + values` 填回作为新 reward，再用 `gamma=1.0` 在整条序列上做 token 级 GAE（跨 turn 边界时 `lastgaelam` 重置）。

对比普通 GAE 的优势：明确区分 turn 间信用分配和 turn 内 token 级信用分配。用 `--bi_level_gae` 开关切换。

实现位于 `rl_algos/gae_utils.py:compute_bi_level_gae_advantage_return`，函数签名与 `RAGEN-main/ragen/trainer/core_algos.py` 保持一致。

### 5. 训练指标追踪器
`utils/tracker.py::TrainingTracker` 自动把每一步的指标同时打到：
- loguru 终端日志（`key=value`格式，便于 grep）
- JSONL 文件 `logs/<exp_name>_metrics.jsonl`（便于 pandas 读取作图）
- (可选) wandb（在代码里把 `use_wandb=True` 即可）

常见指标：
- 训练：`train/actor_loss`、`train/critic_loss`、`train/entropy`、`train/kl_penalty`、`train/approx_kl`、`train/clip_frac`、`train/raw_reward_mean`、`train/raw_reward_var`、`train/echo_trap_sign`
- 评估：`eval/success_rate`、`eval/avg_reward`、`eval/avg_trajectory_length`、`eval/reward_variance`
- 时序：`timing/rollout_sec`、`timing/update_sec`

---

## 🛠 快速上手

### 环境依赖
请使用项目根目录下的 `requirements.txt` 安装：
```bash
pip install -r requirements.txt
```
（核心依赖：`torch`、`transformers`、`gymnasium`、`numpy`、`loguru`、`python-dotenv`、`huggingface_hub`；sokoban / openai / wandb 等为可选依赖，详见 `requirements.txt`。）

### 运行组件测试
```bash
python scripts/test_env.py
python scripts/test_agent.py
```

### 训练实验

> **Windows / PowerShell 用户请注意**：PowerShell 不支持 Linux 的反斜杠 `\` 作为换行续行符，会直接把命令截断。下面的命令全部写成 **单行**，直接复制到终端即可；如果确实想换行，请使用反引号 `` ` ``（PowerShell 续行符）。

**零参数一键跑通（默认配置 = FrozenLake + StarPO + PPO + Qwen2.5-0.5B）：**
```bash
python scripts/train.py
```

**StarPO + PPO + bi-level GAE（FrozenLake，显式覆盖重点超参）：**
```bash
python scripts/train.py --env frozenlake --trainer starpo --algo ppo --exp_name frozenlake_starpo_ppo_bilevel --total_training_steps 200 --eval_interval 20 --num_rollouts 8 --bi_level_gae --high_level_gamma 0.95
```

**StarPO + GRPO（无 Critic 省显存，Bandit 任务）：**
```bash
python scripts/train.py --env bandit --trainer starpo --algo grpo --exp_name bandit_starpo_grpo --total_training_steps 200 --eval_interval 20 --num_rollouts 8
```

**对照组 PureRL + PPO（无 format penalty、无 variance filter）：**
```bash
python scripts/train.py --env frozenlake --trainer pure --algo ppo --exp_name frozenlake_pure_ppo --total_training_steps 200 --eval_interval 20
```

**（可选）PowerShell 下用反引号换行写法：**
```powershell
python scripts/train.py `
    --env frozenlake --trainer starpo --algo ppo `
    --total_training_steps 200 --eval_interval 20 `
    --num_rollouts 8 --bi_level_gae
```

### 关键 CLI 参数一览

> 本表的默认值**就是**全项目唯一的默认值来源（`scripts/train.py::parse_args`）。`configs/config.py` 的 dataclass 里不再重复定义默认值，下游消费（trainer / algo / env）也全部改成 `config.xxx` 直接属性访问——传入什么就用什么，缺字段直接 `AttributeError`，不再静默兜底。

| 类别 | 参数 | 默认值 | 说明 |
|---|---|---|---|
| 运行 | `--exp_name` | `train_default` | 实验名，用于日志 / checkpoint 目录命名 |
| 运行 | `--seed` | `42` | 随机种子（random / numpy / torch 全 seed 同步设置） |
| 环境 | `--env` | `frozenlake` | math / cartpole / frozenlake / sokoban / bandit |
| 环境 | `--max_env_steps` | `10` | 每个 episode 的最大**原子 env step** 数（对齐 RAGEN `max_actions_per_traj=10`；模型一次回复若含 `A \|\| B \|\| C` 就消耗 3 个 step） |
| Agent | `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | 模型路径或 HF repo 名 |
| Agent | `--temperature` | `1.0` | 采样温度（与 RAGEN 训练配置对齐） |
| Agent | `--max_new_tokens` | `256` | 单次生成 token 上限 |
| Agent | _system prompt_ | 由 `envs/*::agent_system_prompt` 持有 | **不是 CLI 参数**。每个环境自带任务相关的"人设 + 格式契约"，`rollout_utils` 自动读 `env.agent_system_prompt`。想改 prompt 直接改对应 env 类即可 |
| 训练 | `--trainer` | `starpo` | starpo / pure |
| 训练 | `--algo` | `ppo` | ppo / grpo |
| 训练 | `--total_training_steps` | `200` | **每步 = 一次 rollout + 一次更新** |
| 训练 | `--eval_interval` | `20` | 每多少步做一次 evaluate |
| 训练 | `--eval_episodes` | `8` | 每次 evaluate 跑多少个 episode |
| 训练 | `--save_interval` | `100` | 每多少步保存 checkpoint |
| RAGEN | `--mode` | `fast` | 预留 fast / slow 实现切换 |
| RAGEN | `--num_rollouts` | `8` | 每个 prompt 采样的轨迹数（GRPO group size） |
| RAGEN | `--prompt_batch_size` | `1` | 每步用多少不同 prompt |
| RAGEN | `--variance_filter_ratio` | `0.25` | StarPO 方差过滤保留比例 |
| RAGEN | `--use_format_reward / --no_format_reward` | on | 是否启用格式惩罚 |
| RAGEN | `--format_penalty` | `-0.1` | 格式错误的负奖励 |
| RAGEN | `--max_turn` | `3` | 每条 trajectory 最多调用 LLM 的次数（RAGEN `agent_proxy.max_turn`）。与 `--max_env_steps` **两层独立截断**，任一触发即 truncate。RAGEN 主力 FrozenLake/Sokoban 用 `1`（0.5B 小模型 + 1600 rollout 配置下过于严苛，默认 3 给模型几次试错空间）。单轮任务（Bandit/Math）第一个 atomic step 就 terminated，不受影响。 |
| RL | `--learning_rate` | `1e-6` | AdamW 学习率 |
| RL | `--ppo_epochs` | `1` | 每批数据更新轮数 |
| RL | `--micro_batch_size` | `2` | 单次 forward/backward 的样本数（VRAM 峰值由此决定；对齐 RAGEN `ppo_micro_batch_size_per_gpu`） |
| RL | `--gradient_accumulation` | `4` | 梯度累积步数；`1` 即关闭累积（等价于 micro-batch 立即 step）。逻辑 mini-batch = `micro_batch_size × gradient_accumulation`，对齐 RAGEN `ppo_mini_batch_size`。**Gradient Checkpointing 已硬编码启用，无开关参数**（rollout 阶段不受影响）|
| RL | `--clip_ratio` | `0.2` | PPO-Clip 阈值 |
| RL | `--kl_coef` | `0.001` | KL 系数；`0.001` 对齐 RAGEN 主流实验，`0.0` 对应 `ppo-nokl` ablation |
| RL | `--ent_coef` | `0.001` | 熵正则化系数 |
| RL | `--vf_coef` | `0.5` | Critic loss 系数（PPO 专属） |
| RL | `--target_kl` | `None` | 提前停止的 KL 阈值（None 表示不启用） |
| RL | `--max_seq_length` | `4096` | 拼接长序列的最大长度 |
| RL | `--use_ref / --no_use_ref` | on | 是否构造 ref_model 并施加 KL；关闭会强制 `kl_coef=0`，省约 1GB VRAM |
| RL | `--optimizer` | `adamw8bit` | `adamw` / `adamw8bit` / `adafactor`；默认 8-bit 节省 ~2GB optimizer state |
| GAE | `--gamma` | `1.0` | token 级折扣因子 |
| GAE | `--lam` | `1.0` | GAE λ |
| GAE | `--bi_level_gae` | off | **开启 bi-level GAE** |
| GAE | `--high_level_gamma` | `0.95` | turn 级折扣因子（仅 `--bi_level_gae` 时生效） |

---

## 💡 开发指南与扩展

### 接新环境
1. `envs/` 下新建 `my_env.py`，继承 `BaseEnv`。
2. 必须实现：
   - `reset(seed, **kwargs)`：记得开头调 `super().reset(seed=seed)` 把 `current_step` 清零。
   - `_step_atomic(atomic_action)`：**只管一个原子动作**，不要自己 `current_step += 1`（`BaseEnv.step` 会统一维护）。
   - `get_valid_actions()`：每轮都会提示给模型的动作描述。
3. 视需要覆盖：
   - `_parse_action(text)`：单 token 解析（默认透传字符串）。
   - `_parse_action_sequence(text)`：**支持 `||` 多动作就在这里处理**。可参考 `FrozenLakeEnv` / `SokobanEnv` 的实现，复用 `_split_action_tokens` 工具函数。
   - `get_env_instruction()`：给模型看的玩法说明（含多动作答题示例），只在 trajectory 第一轮 user message 注入。
4. 在 `envs/__init__.py` 的 `make_env` 注册。

### 接新 RL 算法
1. `rl_algos/` 下新建 `my_algo.py`，继承 `BaseRLAlgo`。
2. 复用 `trajectory_utils.tokenize_trajectory / collate_fn / forward_logprobs_and_entropy` 把 trajectory 打平成长序列。
3. 在 `rl_algos/__init__.py` 的 `make_algo` 注册。

### 接新训练器
1. `ragen_core/` 下新建 `my_trainer.py`（可参考 StarPOTrainer）。
2. 接入 `TrainingTracker` 后，所有指标自动进入 JSONL。

---

## 📊 事后分析

所有指标都被追加到 `logs/<exp_name>_metrics.jsonl`，一行一个 JSON。可以用 pandas 快速作图：

```python
import pandas as pd
df = pd.read_json("logs/frozenlake_starpo_ppo_bilevel_metrics.jsonl", lines=True)
df.plot(x="step", y=["train/raw_reward_mean", "eval/avg_reward"])
```
