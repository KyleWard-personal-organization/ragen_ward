# RAGEN-Ward

RAGEN-Ward is a local, modular reinforcement-learning framework for multi-turn LLM agents. It is inspired by the RAGEN paper, *RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*, and focuses on RAGEN-style text environments, multi-action rollouts, StarPO / PureRL training, and PPO / GRPO policy updates.

This README reflects the current codebase. Training defaults come from `scripts/train.py::parse_args`; evaluation defaults come from `scripts/evaluate.py::parse_args`.

Chinese documentation is available in [README_ZH.md](README_ZH.md).

## Features

- Decoupled environments, agents, RL algorithms, and trainers.
- HuggingFace local model training through `HFAgent`.
- OpenAI-compatible API agent for evaluation-only use.
- Environments: `FrozenLake`, `Sokoban`, `CartPole`, `Bandit`, and `Math / Countdown`.
- Multi-action responses for `FrozenLake` and `Sokoban`, for example `<answer>Right || Down || Down</answer>`.
- Trainers: `StarPOTrainer` and `PureRLTrainer`.
- RL algorithms: PPO and GRPO.
- PPO supports a critic value head, reference-model KL, PPO clipping, and bi-level GAE.
- GRPO is actor-only and uses group-relative reward normalization.
- Training and evaluation metrics are written as JSONL for later analysis.
- `results/` includes plotting scripts and generated report figures.

## Repository Layout

```text
ragen_ward/
├── agents/
│   ├── base_agent.py          # Agent interface
│   ├── hf_agent.py            # HuggingFace local model agent
│   └── openai_agent.py        # OpenAI-compatible API agent for evaluation
├── configs/
│   ├── config.py              # Dataclass config objects; most defaults live in CLI parsers
│   └── constants.py           # Project paths
├── envs/
│   ├── base_env.py            # Text environment base class and RAGEN-style prompt builder
│   ├── gym_envs.py            # CartPole / FrozenLake / Sokoban wrappers
│   ├── bandit_env.py          # One-step bandit environment
│   └── math_env.py            # One-step Countdown math environment
├── evaluation/
│   └── metrics.py             # Evaluation metrics
├── ragen_core/
│   ├── rollout_utils.py       # Shared rollout implementation
│   ├── starpo_trainer.py      # StarPO trainer
│   ├── pure_rl_trainer.py     # PureRL baseline trainer
│   └── trajectory_buffer.py   # Trajectory storage and variance filtering
├── rl_algos/
│   ├── gae_utils.py
│   ├── grpo.py
│   ├── optimizer_utils.py
│   ├── ppo.py
│   └── trajectory_utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── test_env.py
│   ├── test_agent.py
│   └── predict_tokens_left.py
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

## Installation

Create and activate your Python or conda environment, then install the listed dependencies:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` lists the main training dependencies:

- `torch`
- `transformers`
- `huggingface_hub`
- `gymnasium`
- `numpy`
- `loguru`
- `python-dotenv`
- `bitsandbytes`

Notes:

- The default training optimizer is `--optimizer adamw8bit`, so the default path depends on `bitsandbytes`. If that is not suitable on your machine, pass `--optimizer adamw` or `--optimizer adafactor`.
- Several training and evaluation files import `tqdm`. If your environment does not already provide it, install it with `pip install tqdm`.
- `Sokoban` requires the optional package `gym-sokoban`, which is commented out in `requirements.txt`.
- OpenAI-compatible evaluation requires the optional package `openai`, also commented out in `requirements.txt`.
- `results/make_figures.py` requires `matplotlib` and `pandas`, both commented out in `requirements.txt`.
- For CUDA, install the PyTorch wheel that matches your CUDA version from the official PyTorch index.

## Smoke Tests

```bash
python scripts/test_env.py
python scripts/test_agent.py
```

`scripts/test_env.py` defaults to `frozenlake`. You can choose another environment:

```bash
python scripts/test_env.py --env all
python scripts/test_env.py --env math
python scripts/test_env.py --env bandit
python scripts/test_env.py --env sokoban
```

`scripts/test_agent.py` loads `Qwen/Qwen2.5-0.5B-Instruct`. If `models/Qwen_Qwen2.5-0.5B-Instruct` does not exist, `HFAgent` downloads the model through `huggingface_hub.snapshot_download`.

## Training

Default command:

```bash
python scripts/train.py
```

Current default training setup:

- env: `frozenlake`
- trainer: `starpo`
- algo: `ppo`
- model: `Qwen/Qwen2.5-0.5B-Instruct`
- total training steps: `200`
- eval interval: `20`
- eval episodes: `200`
- rollouts per prompt: `16`
- prompt batch size: `8`
- optimizer: `adamw8bit`
- bi-level GAE: enabled by default
- variance filter ratio: `1.0`, so StarPO does not drop groups by default

Examples:

```bash
python scripts/train.py --env frozenlake --trainer starpo --algo ppo --exp_name frozenlake_starpo_ppo --total_training_steps 200 --eval_interval 20
```

```bash
python scripts/train.py --env bandit --trainer starpo --algo grpo --exp_name bandit_starpo_grpo --total_training_steps 200 --eval_interval 20 --num_rollouts 16
```

```bash
python scripts/train.py --env frozenlake --trainer pure --algo ppo --exp_name frozenlake_pure_ppo --total_training_steps 200 --eval_interval 20
```

Disable bi-level GAE:

```bash
python scripts/train.py --no_bi_level_gae
```

Disable the reference model:

```bash
python scripts/train.py --no_use_ref
```

Disable the StarPO format penalty:

```bash
python scripts/train.py --no_format_reward
```

Lower-memory example:

```bash
python scripts/train.py --optimizer adamw --micro_batch_size 1 --gradient_accumulation 32 --max_seq_length 2048 --no_use_ref
```

On Windows PowerShell, use backticks for line continuation:

```powershell
python scripts/train.py `
    --env frozenlake --trainer starpo --algo ppo `
    --total_training_steps 200 --eval_interval 20
```

### Training CLI Defaults

| Category | Argument | Current default | Description |
|---|---|---:|---|
| Run | `--exp_name` | `ragen_baseline_0.5B_sparse_nofilter` | Experiment name for logs, metrics, and checkpoints |
| Run | `--seed` | `42` | random / numpy / torch seed |
| Env | `--env` | `frozenlake` | choices: `math`, `cartpole`, `frozenlake`, `sokoban`, `bandit` |
| Env | `--max_env_steps` | `10` | Maximum atomic environment steps per episode |
| Agent | `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | HF repo id or local model path |
| Agent | `--temperature` | `1.0` | Rollout sampling temperature |
| Agent | `--max_new_tokens` | `256` | Maximum tokens generated per response |
| Trainer | `--trainer` | `starpo` | choices: `starpo`, `pure` |
| Algo | `--algo` | `ppo` | choices: `ppo`, `grpo` |
| Loop | `--total_training_steps` | `200` | One step means rollout plus RL update |
| Loop | `--eval_interval` | `20` | Evaluation frequency |
| Loop | `--eval_episodes` | `200` | Episodes per evaluation |
| Loop | `--save_interval` | `50` | Checkpoint frequency |
| RAGEN | `--mode` | `fast` | reserved; choices: `fast`, `slow` |
| RAGEN | `--num_rollouts` | `16` | Trajectories sampled per prompt |
| RAGEN | `--prompt_batch_size` | `8` | Prompt seeds sampled per training step |
| RAGEN | `--no_format_reward` | not passed | Passing it disables format penalty |
| RAGEN | `--format_penalty` | `-0.1` | Reward added when response format is invalid |
| RAGEN | `--variance_filter_ratio` | `1.0` | StarPO variance-filter keep ratio |
| RAGEN | `--max_turn` | `5` | Maximum LLM calls per trajectory |
| RL | `--learning_rate` | `1e-6` | Actor learning rate |
| RL | `--critic_learning_rate` | `1e-5` | PPO critic head learning rate |
| RL | `--ppo_epochs` | `1` | Optimization epochs per collected batch |
| RL | `--micro_batch_size` | `1` | Samples per forward / backward |
| RL | `--gradient_accumulation` | `32` | Effective mini-batch = micro batch x accumulation |
| RL | `--clip_ratio` | `0.2` | PPO / GRPO clipping ratio |
| RL | `--vf_coef` | `1.0` | PPO value loss coefficient |
| RL | `--ent_coef` | `0.001` | Entropy coefficient |
| RL | `--kl_coef` | `0.001` | Reference KL coefficient |
| RL | `--target_kl` | `None` | KL early-stop threshold |
| RL | `--max_seq_length` | `2048` | Maximum tokenized trajectory length |
| RL | `--no_use_ref` | not passed | Passing it disables the reference model |
| RL | `--optimizer` | `adamw8bit` | choices: `adamw`, `adamw8bit`, `adafactor` |
| GAE | `--gamma` | `1.0` | token-level discount |
| GAE | `--lam` | `1.0` | GAE lambda |
| GAE | `--no_bi_level_gae` | not passed | Passing it disables bi-level GAE |
| GAE | `--high_level_gamma` | `0.95` | turn-level discount for bi-level GAE |

## Standalone Evaluation

Default command:

```bash
python scripts/evaluate.py
```

The evaluation script reuses `ragen_core.rollout_utils.batched_rollout_for_prompt`, the same rollout path used by trainer-side evaluation. Evaluation does not apply format penalty; it only records format compliance.

Evaluate a base HF model:

```bash
python scripts/evaluate.py --agent hf --model_source base --model_name Qwen/Qwen2.5-0.5B-Instruct --episodes 50
```

Evaluate a trained checkpoint:

```bash
python scripts/evaluate.py --agent hf --model_source trained --model_name ragen_baseline_0.5B_sparse_nofilter_final --episodes 50
```

Evaluate through an OpenAI-compatible API:

```bash
python scripts/evaluate.py --agent openai --model_name gpt-4o-mini --api_key YOUR_KEY
```

### Evaluation CLI Defaults

| Argument | Current default | Description |
|---|---:|---|
| `--exp_name` | `eval_trained_30step ` | Current parser default includes a trailing space |
| `--episodes` | `50` | Number of evaluation episodes |
| `--seed` | `42` | RNG seed for drawing episode seeds |
| `--fixed_seed_base` | `None` | If set, env seeds are `base + ep` |
| `--env` | `frozenlake` | choices: `math`, `cartpole`, `frozenlake`, `sokoban`, `bandit` |
| `--max_env_steps` | `10` | Maximum atomic env steps per episode |
| `--max_turn` | `5` | Maximum LLM turns per trajectory |
| `--agent` | `hf` | choices: `hf`, `openai` |
| `--model_source` | `base` | choices: `base`, `trained` |
| `--model_name` | `Qwen_Qwen2.5-1.5B-Instruct` | Model id in base mode; checkpoint folder in trained mode |
| `--temperature` | `0.5` | Evaluation sampling temperature |
| `--max_new_tokens` | `256` | Maximum generated tokens per response |
| `--eval_batch_size` | `8` | Parallel episodes per batched rollout |
| `--api_key` | `None` | OpenAI agent only |
| `--base_url` | `None` | OpenAI agent only |

## Prompt and Environment Protocol

The current code uses `BaseEnv` to build RAGEN-style prompts.

The system message is generated by `env.build_system_content()`:

```text
system_prefix + env_instruction + grid_vocab + action_lookup
```

Every user turn appends `env.build_format_prompt(env.actions_left)`, roughly:

```text
You have N actions left. Always output: <think> [Your thoughts] </think><answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: M words (tokens).
```

The first user message is:

```text
State:
<observation>
<format prompt>
```

Later user messages are:

```text
Reward: <previous env reward>
State:
<observation>
<format prompt>
```

The current rollout path no longer injects `get_env_instruction()` or `get_valid_actions()`. `BaseEnv.get_valid_actions()` is retained only as a deprecated compatibility method and returns an empty string by default.

## Environments

### FrozenLake

Implemented by `envs/gym_envs.py::FrozenLakeEnv`.

- Uses Gymnasium `FrozenLake-v1`.
- The underlying gym environment is deterministic; the wrapper implements the RAGEN-style slippery transition with probabilities `0.8 / 0.1 / 0.1`.
- Current class defaults:
  - `is_slippery=True`
  - `use_shaped_reward=False`
  - `randomize_map=True`
  - `random_map_size=4`
  - `random_map_frozen_p=0.9`
- Supports multi-action responses such as `<answer>Left || Down || Right</answer>`.
- Actions: `Left`, `Down`, `Right`, `Up`.

### Sokoban

Implemented by `envs/gym_envs.py::SokobanEnv`.

- Requires `gym-sokoban`.
- Current setup is `dim_room=(6, 6)` and `num_boxes=1`.
- Supports multi-action responses such as `<answer>Right || Right || Up</answer>`.
- Actions: `Up`, `Down`, `Left`, `Right`.

### CartPole

Implemented by `envs/gym_envs.py::CartPoleEnv`.

- Uses `CartPole-v1`.
- Does not support multi-action sequences.
- Success semantics are special: surviving until time-limit truncation counts as success, while pole fall / out-of-bounds termination counts as failure.

### Bandit

Implemented by `envs/bandit_env.py::BanditEnv`.

- One-step task.
- Arm names are included dynamically in the observation.
- No class-level `action_lookup`.

### Math / Countdown

Implemented by `envs/math_env.py::MathEnv`.

- One-step task.
- Generates four numbers and a solvable target.
- The model should put the expression inside `<answer>...</answer>`.
- Correct reward is `+1`; incorrect reward is `-1`.

## Rollout and Training Flow

`ragen_core/rollout_utils.py` is shared by training and evaluation.

A trajectory entry contains:

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

`BaseAgent.batched_chat_request` defaults to serial calls. `HFAgent` overrides it with real batched generation. `StarPOTrainer`, `PureRLTrainer`, and `scripts/evaluate.py` all use batched rollout by default.

`max_turn` is the LLM-call budget. `max_env_steps` is the atomic environment-step budget. Either one can end an episode. If `max_turn` is reached, the final trajectory entry is marked `truncated=True`, and `info["truncated_reason"]` is set to `"max_turn_reached"`.

## StarPO and PureRL

`StarPOTrainer` runs:

1. Sample `prompt_batch_size` seeds.
2. Collect `num_rollouts` trajectories per seed.
3. Apply format penalty when enabled.
4. Filter prompt groups by return variance.
5. Train PPO or GRPO on the remaining trajectories.
6. Evaluate every `eval_interval` steps.
7. Save checkpoints every `save_interval` steps.

The current default `variance_filter_ratio=1.0` means no groups are actually dropped by default.

`PureRLTrainer` uses the same loop structure, but:

- disables format penalty;
- disables variance filtering;
- appends `_pureRL` to the tracker experiment name.

## PPO and GRPO

### PPO

`rl_algos/ppo.py` tokenizes each multi-turn trajectory into one long sequence. `loss_mask` covers only assistant response tokens. Each turn reward is placed on the last token of that turn's response.

PPO includes:

- actor: `HFAgent.model`
- critic: `nn.Linear(hidden_size, 1)` value head on top of the shared actor backbone
- optional frozen reference model
- PPO clipped policy loss
- value loss
- entropy term
- KL penalty
- gradient accumulation
- gradient norm logging
- gradient checkpointing enabled by default, with `actor.config.use_cache` restored to True for rollout generation speed

If `--no_use_ref` is passed, PPO does not create a reference model and KL is unavailable; the code forces the KL coefficient to behave as zero.

### GRPO

`rl_algos/grpo.py` is actor-only.

- No critic.
- No GAE.
- Computes group-level z-score normalization over trajectory total rewards.
- Applies each trajectory's scalar group-relative advantage to its response tokens.
- Supports optional reference model, PPO-style clipping, entropy, KL, and gradient accumulation.

## Metrics and Outputs

### Logs and Metrics

`utils/logger.py` writes loguru logs to:

```text
logs/<exp_name>.log
```

`utils/tracker.py::TrainingTracker` writes metrics to:

```text
logs/<exp_name>_metrics.jsonl
```

The tracker opens the JSONL file in append mode, writes `run_start` on initialization, and writes `run_end` on close.

`scripts/train.py` tees stdout and stderr to:

```text
train_stdout.txt
```

`scripts/evaluate.py` tees stdout and stderr to:

```text
eval_stdout.txt
```

### Training Metrics

Common training metrics:

- `train/raw_reward_mean`
- `train/raw_reward_var`
- `train/in_group_reward_std`
- `train/actor_loss`
- `train/critic_loss`, PPO only
- `train/entropy`
- `train/kl_penalty`
- `train/approx_kl`
- `train/clip_frac`
- `train/grad_norm`
- `train/grad_norm_max`
- `train/n_grad_steps`
- `train/group_adv_mean`, GRPO only
- `train/group_adv_std`, GRPO only
- `timing/rollout_sec`
- `timing/update_sec`

The current code does not implement an online `echo_trap_sign` boolean or a sliding-window alert. Echo-trap analysis is done after the fact through metrics such as `train/in_group_reward_std`, reward, entropy, gradient norm, and format compliance.

### Evaluation Metrics

Common evaluation summary metrics:

- `eval/success_rate`
- `eval/avg_reward`
- `eval/avg_trajectory_length`
- `eval/avg_num_actions`
- `eval/action_valid_rate`
- `eval/action_effective_rate`
- `eval/format_compliance`
- `eval/reward_variance`
- `eval/total_episodes`, written by standalone `scripts/evaluate.py`

Success is determined by `ragen_core.rollout_utils.judge_success`. If the final step `info` contains `is_success` or `success`, that explicit environment field is used. Otherwise, success is `terminated and not truncated`.

## Checkpoints and Model Cache

HuggingFace models are cached under:

```text
models/<safe_model_name>
```

Slashes are replaced with underscores, for example:

```text
models/Qwen_Qwen2.5-0.5B-Instruct
```

Intermediate checkpoints:

```text
checkpoints/<exp_name>_step<step>
checkpoints/<exp_name>_pureRL_step<step>
```

Final checkpoint:

```text
checkpoints/<exp_name>_final
```

Before saving the final checkpoint, `scripts/train.py` deletes an existing target directory with the same name.

## Figures and Reports

Generate all figures:

```bash
python results/make_figures.py
```

List available figures:

```bash
python results/make_figures.py --list
```

Generate selected figures:

```bash
python results/make_figures.py --only fig01,fig05
```

The plotting script depends on `matplotlib`, `pandas`, and `numpy`.

The repository currently also includes:

- `final_report.md`
- `final_report.html`
- `ppo_analysis.md`
- `results/figures/*.png`

## Extending the Project

### Add an Environment

1. Create a new file under `envs/` and subclass `BaseEnv`.
2. Implement:
   - `reset(seed=None, **kwargs)`
   - `_step_atomic(atomic_action)`
   - `render()`
3. Call `super().reset(seed=seed)` at the start of `reset`.
4. Do not update `current_step` inside `_step_atomic`; `BaseEnv.step` owns that counter.
5. For fixed action sets, define class-level `action_lookup`.
6. For grid environments, define class-level `grid_vocab`.
7. Define `env_instruction`, `max_actions_per_traj`, and `max_response_tokens`.
8. Override `_parse_action_sequence` if the environment should support `A || B || C`.
9. Register the environment in `envs/__init__.py::make_env`.

The current prompt hooks are `build_system_content()` and `build_format_prompt()`, not `get_env_instruction()`.

### Add an RL Algorithm

1. Add a new file under `rl_algos/` and subclass `BaseRLAlgo`.
2. Implement `train_step`, `get_action`, `save`, and `load`.
3. Prefer reusing `trajectory_utils.tokenize_trajectory`, `collate_fn`, and `forward_logprobs_and_entropy`.
4. Register the algorithm in `rl_algos/__init__.py::make_algo`.

### Add a Trainer

1. Add a trainer under `ragen_core/`.
2. Prefer reusing `rollout_utils.rollout_one_trajectory` or `batched_rollout_for_prompt`.
3. Use `TrainingTracker` for metrics.
4. Keep the trajectory schema compatible with PPO and GRPO.

## Current Code Notes

- Most dataclass fields in `configs/config.py` have no defaults; CLI parsers are the source of defaults.
- `scripts/train.py` always constructs `HFAgent`; training does not support OpenAI API agents.
- `scripts/evaluate.py` supports both `hf` and `openai` agents.
- `BaseEnv.get_valid_actions()` is a deprecated compatibility method and is not used by rollout prompts.
- `requirements.txt` currently does not list `tqdm`, although code imports it.
- `scripts/evaluate.py` currently has a trailing space in the default `--exp_name`.
- `PureRLTrainer` writes metrics as `<exp_name>_pureRL_metrics.jsonl`.
