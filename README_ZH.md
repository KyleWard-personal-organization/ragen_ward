# RAGEN-Ward

RAGEN-Ward 是一个完全本地、模块解耦的多轮强化学习推理框架，灵感来自 RAGEN 论文 *RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*。当前代码重点围绕文本环境中的 LLM agent rollout、StarPO / PureRL 训练框架、PPO / GRPO 更新，以及 RAGEN 风格的 prompt / multi-action 协议展开。

这个 README 按当前代码状态整理。训练 CLI 的默认值以 `scripts/train.py::parse_args` 为准，评估 CLI 的默认值以 `scripts/evaluate.py::parse_args` 为准。

## 当前能力

- 环境、Agent、RL 算法、训练器彼此解耦。
- 支持 HuggingFace 本地模型训练；OpenAI 风格 API agent 仅用于无梯度评估。
- 支持 `FrozenLake`、`Sokoban`、`CartPole`、`Bandit`、`Math / Countdown` 环境。
- `FrozenLake` 和 `Sokoban` 支持一次 LLM 回复执行多个原子动作，例如 `<answer>Right || Down || Down</answer>`。
- 训练器包含 `StarPOTrainer` 和 `PureRLTrainer`。
- RL 算法包含 PPO 和 GRPO。
- PPO 支持 critic value head、reference model KL、PPO clip、bi-level GAE。
- GRPO 是 actor-only，使用组内 reward z-score 作为 token advantage。
- rollout、训练、评估指标写入 JSONL，便于 pandas / matplotlib 后处理。
- `results/` 下包含用于最终报告的作图脚本和已生成图表。

## 项目结构

```text
ragen_ward/
├── agents/
│   ├── base_agent.py          # Agent 抽象接口：chat_request / batched_chat_request / get_log_probs
│   ├── hf_agent.py            # HuggingFace 本地模型 agent，训练使用
│   └── openai_agent.py        # OpenAI 风格 API agent，仅适合评估
├── configs/
│   ├── config.py              # Dataclass 配置对象；默认值不在这里定义
│   └── constants.py           # PROJECT_ROOT / LOG_DIR / MODELS_DIR / CKPT_DIR 等路径
├── envs/
│   ├── base_env.py            # 文本环境基类、RAGEN 风格 prompt 构造、多动作 step
│   ├── gym_envs.py            # CartPole / FrozenLake / Sokoban 包装
│   ├── bandit_env.py          # 单轮 bandit 环境
│   └── math_env.py            # 单轮 Countdown 数学环境
├── evaluation/
│   └── metrics.py             # success_rate、avg_reward、action_valid_rate 等评估指标
├── ragen_core/
│   ├── rollout_utils.py       # 单条 / batch rollout 的公共实现
│   ├── starpo_trainer.py      # StarPO：format penalty + variance filtering
│   ├── pure_rl_trainer.py     # PureRL baseline：无 format penalty、无 variance filtering
│   └── trajectory_buffer.py   # trajectory 存储、return 统计、方差过滤
├── rl_algos/
│   ├── gae_utils.py           # flat GAE 与 bi-level GAE
│   ├── grpo.py                # actor-only GRPO
│   ├── optimizer_utils.py     # adamw / adamw8bit / adafactor 构造
│   ├── ppo.py                 # PPO + critic + KL + GAE
│   └── trajectory_utils.py    # trajectory tokenize / collate / logprob + entropy forward
├── scripts/
│   ├── train.py               # 训练入口
│   ├── evaluate.py            # 独立评估入口
│   ├── test_env.py            # 环境 smoke test
│   ├── test_agent.py          # HF agent smoke test
│   └── predict_tokens_left.py # 辅助脚本
├── results/
│   ├── data_loader.py
│   ├── make_figures.py
│   ├── plot_styles.py
│   └── figures/*.png
├── tests/
│   └── test_batched_rollout.py
├── utils/
│   ├── basic_utils.py
│   ├── logger.py
│   ├── stdout_tee.py
│   └── tracker.py
├── final_report.md
├── final_report.html
├── ppo_analysis.md
└── requirements.txt
```

## 安装

建议先创建并激活自己的 Python / conda 环境，再安装依赖。

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 已列出训练主路径需要的核心依赖：

- `torch`
- `transformers`
- `huggingface_hub`
- `gymnasium`
- `numpy`
- `loguru`
- `python-dotenv`
- `bitsandbytes`

注意：

- `scripts/train.py` 默认使用 `--optimizer adamw8bit`，因此默认训练路径依赖 `bitsandbytes`。如果本机不适合安装 bitsandbytes，可以训练时显式传 `--optimizer adamw` 或 `--optimizer adafactor`。
- 多个训练 / 评估文件直接 import `tqdm`。如果你的环境没有预装 tqdm，请额外执行 `pip install tqdm`。
- `Sokoban` 需要可选依赖 `gym-sokoban`，默认在 `requirements.txt` 中是注释状态。
- OpenAI 风格 API 评估需要可选依赖 `openai`，默认在 `requirements.txt` 中是注释状态。
- `results/make_figures.py` 需要 `matplotlib` 和 `pandas`，默认在 `requirements.txt` 中也是注释状态。
- PyTorch 的 CUDA wheel 建议按你的 CUDA 版本从 PyTorch 官方 index 安装；`requirements.txt` 只声明最低版本。

## 快速检查

```bash
python scripts/test_env.py
python scripts/test_agent.py
```

`scripts/test_env.py` 默认只测 `frozenlake`，也可以指定：

```bash
python scripts/test_env.py --env all
python scripts/test_env.py --env math
python scripts/test_env.py --env bandit
python scripts/test_env.py --env sokoban
```

`scripts/test_agent.py` 会尝试加载 `Qwen/Qwen2.5-0.5B-Instruct`。如果本地 `models/Qwen_Qwen2.5-0.5B-Instruct` 不存在，`HFAgent` 会通过 `huggingface_hub.snapshot_download` 下载到 `models/` 下。

## 训练入口

默认训练命令：

```bash
python scripts/train.py
```

当前默认配置是：

- env: `frozenlake`
- trainer: `starpo`
- algo: `ppo`
- model: `Qwen/Qwen2.5-0.5B-Instruct`
- total training steps: `200`
- eval interval: `20`
- eval episodes: `200`
- num rollouts per prompt: `16`
- prompt batch size: `8`
- optimizer: `adamw8bit`
- bi-level GAE: 默认开启
- variance filter ratio: `1.0`，即 StarPO 的方差过滤默认不丢弃任何 group

示例：

```bash
python scripts/train.py --env frozenlake --trainer starpo --algo ppo --exp_name frozenlake_starpo_ppo --total_training_steps 200 --eval_interval 20
```

```bash
python scripts/train.py --env bandit --trainer starpo --algo grpo --exp_name bandit_starpo_grpo --total_training_steps 200 --eval_interval 20 --num_rollouts 16
```

```bash
python scripts/train.py --env frozenlake --trainer pure --algo ppo --exp_name frozenlake_pure_ppo --total_training_steps 200 --eval_interval 20
```

关闭 bi-level GAE：

```bash
python scripts/train.py --no_bi_level_gae
```

关闭 reference model 和 KL anchor：

```bash
python scripts/train.py --no_use_ref
```

关闭 StarPO format penalty：

```bash
python scripts/train.py --no_format_reward
```

降低显存压力的常见组合：

```bash
python scripts/train.py --optimizer adamw --micro_batch_size 1 --gradient_accumulation 32 --max_seq_length 2048 --no_use_ref
```

Windows PowerShell 多行命令请使用反引号而不是反斜杠：

```powershell
python scripts/train.py `
    --env frozenlake --trainer starpo --algo ppo `
    --total_training_steps 200 --eval_interval 20
```

### 训练 CLI 默认值

| 类别 | 参数 | 当前默认值 | 说明 |
|---|---|---:|---|
| 运行 | `--exp_name` | `ragen_baseline_0.5B_sparse_nofilter` | 实验名；影响日志、metrics、checkpoint 命名 |
| 运行 | `--seed` | `42` | random / numpy / torch seed |
| 环境 | `--env` | `frozenlake` | choices: `math`, `cartpole`, `frozenlake`, `sokoban`, `bandit` |
| 环境 | `--max_env_steps` | `10` | 每个 episode 的最大原子环境步数 |
| Agent | `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | HF repo id 或本地模型路径名 |
| Agent | `--temperature` | `1.0` | rollout 采样温度 |
| Agent | `--max_new_tokens` | `256` | 单次回复最大生成 token 数 |
| 训练器 | `--trainer` | `starpo` | choices: `starpo`, `pure` |
| 算法 | `--algo` | `ppo` | choices: `ppo`, `grpo` |
| 循环 | `--total_training_steps` | `200` | 每步 = rollout + RL update |
| 循环 | `--eval_interval` | `20` | 每多少训练步评估一次 |
| 循环 | `--eval_episodes` | `200` | 每次评估 episode 数 |
| 循环 | `--save_interval` | `50` | 每多少训练步保存 checkpoint |
| RAGEN | `--mode` | `fast` | 预留 choices: `fast`, `slow` |
| RAGEN | `--num_rollouts` | `16` | 每个 prompt 采样的 trajectory 数，也是训练侧 batch rollout 的组大小 |
| RAGEN | `--prompt_batch_size` | `8` | 每个训练 step 使用多少个不同 seed / prompt |
| RAGEN | `--no_format_reward` | 未传时开启 | 传入后关闭 format penalty |
| RAGEN | `--format_penalty` | `-0.1` | 格式错误时叠加到该 turn reward 的负奖励 |
| RAGEN | `--variance_filter_ratio` | `1.0` | StarPO 方差过滤保留比例；`1.0` 表示不实际过滤 |
| RAGEN | `--max_turn` | `5` | 每条 trajectory 最多 LLM 调用次数；与 `--max_env_steps` 独立 |
| RL | `--learning_rate` | `1e-6` | actor learning rate |
| RL | `--critic_learning_rate` | `1e-5` | PPO critic head learning rate；GRPO 忽略 |
| RL | `--ppo_epochs` | `1` | 每批数据更新轮数 |
| RL | `--micro_batch_size` | `1` | 每次 forward / backward 的样本数 |
| RL | `--gradient_accumulation` | `32` | 梯度累积步数；有效 mini-batch = micro batch x accumulation |
| RL | `--clip_ratio` | `0.2` | PPO / GRPO clip ratio |
| RL | `--vf_coef` | `1.0` | PPO critic loss 系数 |
| RL | `--ent_coef` | `0.001` | entropy bonus 系数 |
| RL | `--kl_coef` | `0.001` | reference KL 系数 |
| RL | `--target_kl` | `None` | KL early stop 阈值，未设置则关闭 |
| RL | `--max_seq_length` | `2048` | trajectory 拼成长序列后的最大长度 |
| RL | `--no_use_ref` | 未传时开启 ref | 传入后不创建 ref model，并强制 KL 失效 |
| RL | `--optimizer` | `adamw8bit` | choices: `adamw`, `adamw8bit`, `adafactor` |
| GAE | `--gamma` | `1.0` | token-level discount |
| GAE | `--lam` | `1.0` | GAE lambda |
| GAE | `--no_bi_level_gae` | 未传时开启 bi-level GAE | 传入后关闭 bi-level GAE，改用 flat GAE |
| GAE | `--high_level_gamma` | `0.95` | bi-level GAE 的 turn-level discount |

## 独立评估入口

默认评估命令：

```bash
python scripts/evaluate.py
```

评估入口复用 `ragen_core.rollout_utils.batched_rollout_for_prompt`，与训练器内部 evaluate 使用同一套 rollout / prompt / success 判定逻辑。评估阶段 `use_format_reward=False`，只统计格式合规率，不叠加格式惩罚。

评估 base HF 模型：

```bash
python scripts/evaluate.py --agent hf --model_source base --model_name Qwen/Qwen2.5-0.5B-Instruct --episodes 50
```

评估训练后的 checkpoint：

```bash
python scripts/evaluate.py --agent hf --model_source trained --model_name ragen_baseline_0.5B_sparse_nofilter_final --episodes 50
```

使用 OpenAI 风格 API 评估：

```bash
python scripts/evaluate.py --agent openai --model_name gpt-4o-mini --api_key YOUR_KEY
```

### 评估 CLI 默认值

| 参数 | 当前默认值 | 说明 |
|---|---:|---|
| `--exp_name` | `eval_trained_30step ` | 当前代码默认值末尾包含一个空格 |
| `--episodes` | `50` | 评估 episode 数 |
| `--seed` | `42` | 用于随机生成 episode seed 的 RNG seed |
| `--fixed_seed_base` | `None` | 设置后使用 `base + ep` 的确定性环境 seed |
| `--env` | `frozenlake` | choices: `math`, `cartpole`, `frozenlake`, `sokoban`, `bandit` |
| `--max_env_steps` | `10` | 每个 episode 最大原子环境步数 |
| `--max_turn` | `5` | 每条 trajectory 最大 LLM turn 数 |
| `--agent` | `hf` | choices: `hf`, `openai` |
| `--model_source` | `base` | choices: `base`, `trained` |
| `--model_name` | `Qwen_Qwen2.5-1.5B-Instruct` | base 模式下是模型名；trained 模式下是 `checkpoints/` 子目录 |
| `--temperature` | `0.5` | 评估采样温度 |
| `--max_new_tokens` | `256` | 单次回复最大生成 token 数 |
| `--eval_batch_size` | `8` | batch rollout 的并行 episode 数 |
| `--api_key` | `None` | OpenAI agent 使用 |
| `--base_url` | `None` | OpenAI agent 使用 |

## Prompt 与环境协议

当前项目使用 `BaseEnv` 统一构造 RAGEN 风格 prompt。

System message 由 `env.build_system_content()` 生成：

```text
system_prefix + env_instruction + grid_vocab + action_lookup
```

每个 user turn 会追加 `env.build_format_prompt(env.actions_left)`，形如：

```text
You have N actions left. Always output: <think> [Your thoughts] </think><answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: M words (tokens).
```

第一轮 user message 是：

```text
State:
<observation>
<format prompt>
```

后续 user message 是：

```text
Reward: <previous env reward>
State:
<observation>
<format prompt>
```

当前 rollout 不再注入 `get_env_instruction()` 或 `get_valid_actions()`。`get_valid_actions()` 在 `BaseEnv` 中保留为 deprecated 兼容方法，默认返回空字符串。

## 环境说明

### FrozenLake

实现位于 `envs/gym_envs.py::FrozenLakeEnv`。

- 底层使用 `gymnasium` 的 `FrozenLake-v1`。
- 内部始终将 gym env 设置为 deterministic，然后在 wrapper 层按 `0.8 / 0.1 / 0.1` 自己实现 slippery action 重采样。
- 当前 class defaults:
  - `is_slippery=True`
  - `use_shaped_reward=False`
  - `randomize_map=True`
  - `random_map_size=4`
  - `random_map_frozen_p=0.9`
- 支持 `<answer>Left || Down || Right</answer>` 多动作序列。
- 动作词映射：`Left`, `Down`, `Right`, `Up`。

### Sokoban

实现位于 `envs/gym_envs.py::SokobanEnv`。

- 需要 `gym-sokoban`。
- 当前固定为 `dim_room=(6, 6)`、`num_boxes=1`。
- 支持 `<answer>Right || Right || Up</answer>` 多动作序列。
- 动作词映射：`Up`, `Down`, `Left`, `Right`。

### CartPole

实现位于 `envs/gym_envs.py::CartPoleEnv`。

- 使用 `CartPole-v1`。
- 不支持多动作序列；每次回复只执行一个动作。
- 成功语义和目标型环境不同：活到 time limit truncation 才算 success，杆倒导致 terminated 视为失败。

### Bandit

实现位于 `envs/bandit_env.py::BanditEnv`。

- 单轮任务，选完 arm 后立即 `terminated=True`。
- arm 名称动态写在 observation 中。
- 不使用 class-level `action_lookup`。

### Math / Countdown

实现位于 `envs/math_env.py::MathEnv`。

- 单轮任务。
- 随机生成 4 个数和一个可解目标值。
- 模型需要在 `<answer>...</answer>` 中输出表达式。
- 答对 reward `+1`，答错 reward `-1`。

## Rollout 与训练流程

`ragen_core/rollout_utils.py` 是训练和评估共享的 rollout 实现。

单条 trajectory 记录的主要字段：

- `obs`
- `messages`
- `response`
- `reward`
- `env_reward`
- `format_penalty`
- `format_ok`
- `terminated`
- `truncated`
- `info`
- `turn_idx`

`BaseAgent.batched_chat_request` 默认退化为串行调用；`HFAgent` 重写为真正的 batch generate。`StarPOTrainer`、`PureRLTrainer` 和 `scripts/evaluate.py` 默认都走 batch rollout。

`max_turn` 是 LLM 调用次数预算，`max_env_steps` 是原子环境步数预算。任意一层触发都会结束 episode；若达到 `max_turn`，最后一条 trajectory entry 会被标记 `truncated=True`，并在 `info` 中写入 `truncated_reason="max_turn_reached"`。

## StarPO 与 PureRL

`StarPOTrainer` 每个训练 step 的流程：

1. 采样 `prompt_batch_size` 个 seed。
2. 每个 seed 采样 `num_rollouts` 条 trajectory。
3. 如果启用 format reward，格式不符合 `<think>...</think><answer>...</answer>` 的 turn 加 `format_penalty`。
4. 按同 prompt group 的 return variance 做方差过滤。
5. 将剩余 trajectories 交给 PPO 或 GRPO 更新。
6. 按 `eval_interval` 做在线评估。
7. 按 `save_interval` 保存 checkpoint。

当前默认 `variance_filter_ratio=1.0`，因此 StarPO 默认不会实际丢弃 group，但相关逻辑仍会执行。

`PureRLTrainer` 与 StarPOTrainer 保持同样的循环结构，但：

- 不使用 format penalty。
- 不使用 variance filtering。
- tracker 的 `exp_name` 会追加 `_pureRL`。

## PPO 与 GRPO

### PPO

`rl_algos/ppo.py` 将每条多轮 trajectory tokenized 成一条长序列。`loss_mask` 只覆盖 assistant response tokens。每个 turn 的 reward 只放在该 turn response 的最后一个 token 上。

PPO 包含：

- actor: `HFAgent.model`
- critic: 共享 backbone 之上的 `nn.Linear(hidden_size, 1)` value head
- optional ref model: `copy.deepcopy(actor)` 后冻结
- PPO clipped policy loss
- value loss
- entropy term
- KL penalty
- gradient accumulation
- gradient norm logging
- 默认开启 gradient checkpointing，并把 `actor.config.use_cache` 恢复为 True 以保持 rollout generate 速度

如果 `--no_use_ref`，PPO 不创建 ref model，并强制 KL 分布不可用；代码会将 `kl_coef` 视为 0。

### GRPO

`rl_algos/grpo.py` 是 actor-only。

- 不使用 critic。
- 不使用 GAE。
- 对同一批 trajectory 的 total reward 做组内 z-score。
- 将每条 trajectory 的组相对 advantage 标量铺到其 response token 上。
- 同样支持 optional ref model、PPO clip、entropy、KL、gradient accumulation。

## 指标和输出文件

### 日志与 metrics

`utils/logger.py` 将 loguru 日志写到：

```text
logs/<exp_name>.log
```

`utils/tracker.py::TrainingTracker` 将 metrics 写到：

```text
logs/<exp_name>_metrics.jsonl
```

每次 tracker 初始化都会以 append 模式打开 JSONL，并写入 `run_start`；结束时写入 `run_end`。

`scripts/train.py` 会通过 `utils/stdout_tee.py` 将 stdout / stderr tee 到：

```text
train_stdout.txt
```

`scripts/evaluate.py` 会 tee 到：

```text
eval_stdout.txt
```

### 训练指标

常见训练指标包括：

- `train/raw_reward_mean`
- `train/raw_reward_var`
- `train/in_group_reward_std`
- `train/actor_loss`
- `train/critic_loss`，PPO 有，GRPO 没有
- `train/entropy`
- `train/kl_penalty`
- `train/approx_kl`
- `train/clip_frac`
- `train/grad_norm`
- `train/grad_norm_max`
- `train/n_grad_steps`
- `train/group_adv_mean`，GRPO 有
- `train/group_adv_std`，GRPO 有
- `timing/rollout_sec`
- `timing/update_sec`

当前代码没有实现在线的 `echo_trap_sign` 布尔判定或滑动窗口报警器。Echo trap 相关分析主要通过 `train/in_group_reward_std`、reward、entropy、gradient norm、format compliance 等指标事后观察。

### 评估指标

评估 summary 常见字段：

- `eval/success_rate`
- `eval/avg_reward`
- `eval/avg_trajectory_length`
- `eval/avg_num_actions`
- `eval/action_valid_rate`
- `eval/action_effective_rate`
- `eval/format_compliance`
- `eval/reward_variance`
- `eval/total_episodes`，独立 `scripts/evaluate.py` 写 summary 时包含

success 判定由 `ragen_core.rollout_utils.judge_success` 负责。若最后一步 `info` 中有 `is_success` 或 `success`，优先使用环境显式字段；否则使用 `terminated and not truncated`。

## Checkpoint 和模型缓存

HuggingFace 模型缓存目录：

```text
models/<safe_model_name>
```

其中 `/` 会被替换为 `_`，例如：

```text
models/Qwen_Qwen2.5-0.5B-Instruct
```

训练中间 checkpoint：

```text
checkpoints/<exp_name>_step<step>
checkpoints/<exp_name>_pureRL_step<step>
```

训练结束后的 final checkpoint：

```text
checkpoints/<exp_name>_final
```

`scripts/train.py` 保存 final checkpoint 前，如果目标目录已存在，会先删除旧目录再保存。

## 作图与报告

生成 `results/figures/*.png`：

```bash
python results/make_figures.py
```

列出可生成的图：

```bash
python results/make_figures.py --list
```

只生成部分图：

```bash
python results/make_figures.py --only fig01,fig05
```

作图脚本依赖 `matplotlib`、`pandas`、`numpy`。当前仓库中还包含：

- `final_report.md`
- `final_report.html`
- `ppo_analysis.md`
- `results/figures/*.png`

## 扩展指南

### 新增环境

1. 在 `envs/` 下新增环境文件，继承 `BaseEnv`。
2. 必须实现：
   - `reset(seed=None, **kwargs)`
   - `_step_atomic(atomic_action)`
   - `render()`
3. `reset` 开头应调用 `super().reset(seed=seed)`，以重置 `current_step`。
4. `_step_atomic` 只执行一个原子动作，不要自己修改 `current_step`。
5. 如果是固定动作集合，设置 class-level `action_lookup`。
6. 如果是网格环境，设置 class-level `grid_vocab`。
7. 设置 `env_instruction`、`max_actions_per_traj`、`max_response_tokens`。
8. 如需支持 `A || B || C`，覆盖 `_parse_action_sequence`。
9. 在 `envs/__init__.py::make_env` 注册新环境。

当前 prompt 入口是 `build_system_content()` 和 `build_format_prompt()`，不是 `get_env_instruction()`。

### 新增 RL 算法

1. 在 `rl_algos/` 下新增算法文件，继承 `BaseRLAlgo`。
2. 实现 `train_step`、`get_action`、`save`、`load`。
3. 推荐复用 `trajectory_utils.tokenize_trajectory`、`collate_fn`、`forward_logprobs_and_entropy`。
4. 在 `rl_algos/__init__.py::make_algo` 注册新算法。

### 新增训练器

1. 在 `ragen_core/` 下新增 trainer。
2. 推荐复用 `rollout_utils.rollout_one_trajectory` 或 `batched_rollout_for_prompt`。
3. 使用 `TrainingTracker` 写 metrics。
4. 保持 trajectory schema 与现有 PPO / GRPO 兼容。

## 当前需要注意的代码细节

- `configs/config.py` 的 dataclass 大多数字段没有默认值；默认值集中在 CLI parser 中。
- `scripts/train.py` 只构造 `HFAgent`，训练路径不支持 OpenAI agent。
- `scripts/evaluate.py` 支持 `hf` 和 `openai` 两种 agent。
- `BaseEnv.get_valid_actions()` 当前是 deprecated compatibility method，实际 rollout prompt 不使用它。
- `requirements.txt` 目前没有列出 `tqdm`，但代码中使用了它。
- `scripts/evaluate.py` 的 `--exp_name` 当前默认值末尾带空格。
- `PureRLTrainer` 的 metrics 文件会使用 `<exp_name>_pureRL_metrics.jsonl`。
