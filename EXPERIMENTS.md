# RAGEN-Ward 实验运行说明（Experiment Protocol）

> **文档定位**：独立于 `README.md` 之外，专门记录**本地硬件条件下如何系统性地复现 + 探索 RAGEN 论文**的实验计划、推进节奏和各阶段的通过标准。  
> **目标受众**：本人（日后回看 / 写报告）、未来可能的协作者。  
> **状态**：Demo / Living Document。Stage 1 已落地数据；Stage 2 有完整方案待执行；Stage 3 仍为占位骨架，后续补充。

---

## 0. 为什么要写这份文档

`README.md` 讲的是"**这个代码库是什么**" —— 架构、模块、CLI 参数。

本文档讲的是"**我怎么用这个代码库做出一篇可交付的报告**" —— 实验分阶段、每阶段做什么、拿到什么、下一阶段靠什么推进。README 的参数表到了这里是"输入"；本文档关心的是"产出 + 决策"。

两者分离是为了：
- README 保持稳定，作为工具手册；
- 本文档随实验推进持续更新，作为项目的"研究日记纲要"。

---

## 1. 总体流程俯瞰

```
Stage 1:  Base Baseline Preview  (已完成)
          └─ 目的: 在"论文参数"下粗看 0.5B/1.5B base 的真实 baseline
          └─ 产出: 两个 base 模型的 8 项 RAGEN 指标 snapshot
          └─ 耗时: 约 10 min / 模型 × 10 episodes

Stage 2:  本地参数调优  (即将开始)
          ├─ Phase A: 冒烟  (10 training_steps, ~30-40 min)
          │         └─ 目的: 确认训练管线能端到端跑通, 不 OOM, loss 不 NaN
          ├─ Phase B: 控制变量调参  (50 training_steps × N 组, 每组 ~3h)
          │         └─ 目的: 固定 "baseline", 一次只改 1 个变量, 找本地最优组合
          └─ Phase C: 冲刺训练  (200 training_steps, ~10-13h)
                    └─ 目的: 用 Phase B 赢家跑完整训练, 得到最终 checkpoint

Stage 3:  正式评估实验  (Stage 2 产出 checkpoint 后开始)
          └─ 目的: 系统性对比 {base / trained} × {PPO / GRPO} × {StarPO / PureRL}
                  × {0.5B / 1.5B} × 其他维度(待补充)
          └─ 产出: 最终报告所需的全部数字表格 + 曲线图
```

阶段之间的**依赖关系**：Stage 2 Phase B 的结论决定 Phase C 参数；Phase C 产出的 checkpoint 是 Stage 3 的直接输入。Stage 1 本身不改变 Stage 2 的任何决策，只是作为"未训练时的下界"存档，供 Stage 3 对比。

---

## 2. 硬件与环境基线

| 项目 | 规格 / 版本 |
|---|---|
| GPU | 单卡，**8GB VRAM** |
| RAM | 64GB |
| 训练模型 | Qwen2.5-0.5B-Instruct（主），Qwen2.5-1.5B-Instruct（base 对照） |
| 精度 | bfloat16 权重 + `adamw8bit` optimizer state |
| 核心依赖 | 见 `requirements.txt` |

**硬件带来的关键让步**（这些值偏离论文，但在 8GB VRAM 下属于"能跑通的妥协"）：

| 参数 | 论文值 | 本地值 | 妥协了什么 |
|---|---|---|---|
| `prompt_batch_size` | 8 | **2**（Safe Baseline）| effective batch = `num_rollouts × prompt_batch` 从 128 砍到 8；见 §5.3 关于 StarPO variance filter 的关键发现 |
| `micro_batch_size × gradient_accumulation` | 4 × 8 = 32~64 | **2 × 4 = 8** | 梯度估计更不稳；已通过 **强制 Gradient Checkpointing + Accumulation** 把 effective mini-batch 从最初的 2 推到 8 |
| `optimizer` | AdamW fp32 | adamw8bit | 省约 6GB VRAM，收敛略慢（论文也用过 8bit） |
| `max_seq_length` | 4096 | **2048** | 省激活显存；FrozenLake 实测单条 trajectory 不超过 2k token |

**与论文完全对齐的核心参数**（不动）：  
`learning_rate=1e-6`, `clip_ratio=0.2`, `kl_coef=0.001`, `format_penalty=-0.1`, `variance_filter_ratio=0.25`, `gamma=1.0`, `lam=1.0`, `max_env_steps=10`, `temperature=1.0` (train) / `0.5` (eval)。

---

## 3. 评估框架对齐声明（一次性声明，下文各阶段都引用）

本项目的 `scripts/evaluate.py` 与 RAGEN 论文 `RAGEN-main/ragen/eval.py` + `es_manager.get_rollout_states` 的对齐情况：

| 维度 | 本项目做法 | 与 RAGEN 对齐 |
|---|---|---|
| rollout 构造 | `ragen_core.rollout_one_trajectory` 统一构造（含 `env_instruction` 首轮注入、`max_turn` 硬截断、`terminated/truncated` 双停止） | ✅ |
| 成功判定 | `info["is_success"]` 优先，fallback `terminated and not truncated`（见 `ragen_core.rollout_utils.judge_success`） | ✅ |
| 格式检查正则 | `<think>.*?</think>\s*<answer>.*?</answer>` | ✅（与 RAGEN `ctx_manager.py:158` 一致）|
| 核心指标集 | success_rate / avg_reward / avg_trajectory_length / avg_num_actions / action_valid_rate / action_effective_rate / format_compliance / reward_variance | ✅ 8 项全对齐 |
| seed 策略 | 默认 `Random(seed=42)` 决定论，`--fixed_seed_base` 可选 debug | ⚠️ 个性化，**优于**论文的全随机（支持公平横比） |
| 持久化 | `logs/eval_<exp_name>_metrics.jsonl` + `stdout.txt` | ⚠️ 个性化（论文用 wandb） |

**已知偏差（不阻塞核心 claim）**：
- `action_effective_rate` 是 per-turn 粒度（整 turn 内任一 atomic 不 effective 即记 0），RAGEN 可能是 per-atomic。对 `||` 多步策略略偏严。**选择不改**，全实验统一用 per-turn 口径，横比仍然公平。
- 论文中的 `pass@k` / `entropy` / `BoN` 未实现，不是核心 claim 所需。

> Stage 1/2/3 的所有评估都使用这套对齐的流程。只要本声明不改，横比就是公平的。

---

## 4. Stage 1：Base Baseline Preview（已完成）

### 4.1 目的

在"尽量接近论文的评估参数"下，对 0.5B / 1.5B 两个 base 模型各跑 10 episodes，拿到**未训练时**的真实 baseline。本阶段**不追求绝对数字的 publication quality**，追求：
1. 验证评估管线 bug-free；
2. 记录"**下界**"供 Stage 3 的 trained 模型对比；
3. 粗看两个模型规模的差异。

### 4.2 参数（已用）

```bash
python scripts/evaluate.py --env frozenlake --episodes 10 --max_turn 5 --max_env_steps 20 --temperature 0.5 --model_name Qwen/Qwen2.5-0.5B-Instruct
python scripts/evaluate.py --env frozenlake --episodes 10 --max_turn 5 --max_env_steps 20 --temperature 0.5 --model_name Qwen/Qwen2.5-1.5B-Instruct
```

> 注意：10 episodes 的 95% Wilson CI 宽度 ≈ ±25%，success_rate 不能做强断言。本阶段只用于"定性粗看"。

### 4.3 实测结果（2026-04-18）

| 指标 | 0.5B base | 1.5B base | 备注 |
|---|---|---|---|
| `eval/success_rate` | **0.00** | **0.00** | 10/10 失败；都没学过 FrozenLake |
| `eval/avg_reward` | −0.30 | −0.21 | 1.5B 少吞 format penalty |
| `eval/avg_trajectory_length` | 3.4 turn | 3.8 turn | 近似 |
| `eval/avg_num_actions` | 6.0 | 6.7 | 近似 |
| `eval/action_valid_rate` | 0.72 | 0.71 | **两者接近，说明 parse 能力差不多** |
| `eval/action_effective_rate` | 0.45 | 0.35 | 1.5B **反而更低**（多步规划时撞墙概率叠加，per-turn AND 口径偏严）|
| `eval/format_compliance` | **0.00** | **0.00** | 两者都不会自发输出 `<think>/<answer>` 双 tag |
| `eval/reward_variance` | 0.31 | 0.10 | 0.5B 有一个 "seed=1051802512 吐 20 个垃圾 token" 的离群 ep 拉高方差 |

### 4.4 关键发现

1. **两个 base 的 `success_rate` 都是 0.0**：完全符合预期，base 模型不可能凭空解出 FrozenLake。
2. **`format_compliance=0`**：RL 训练的首要目标之一就是把这个拉起来，Stage 2 的 `--use_format_reward` 就是为此。
3. **seed 决定论生效**：两次评估用了完全相同的 10 个 env_seed（因为 `Random(seed=42)` 产生同一批），所以 0.5B vs 1.5B 是**同关卡正面比较**，差异更可信。
4. **1.5B 的"正常化"效应**：在 seed=1051802512 这个 0.5B 的灾难 ep（reward=−1.9）上，1.5B 正常跑完 3 turn（reward=0.0），表明 1.5B 对格式和空间的理解更稳。

### 4.5 产物

- `logs/eval_default_metrics.jsonl` 内的 `_event: run_start` ... `run_end` 片段（两次 run 都在此文件，append 模式）
- `stdout.txt`（只保留最近一次）

---

## 5. Stage 2：本地参数调优

### 5.1 总体思路

**RL 和 SFT 不一样**，小 step 的信号可能被稀疏奖励吃掉（FrozenLake 4x4 的 `p_success` 对 base 约 0%，`num_rollouts=8` 每步期望 0.4 次成功，至少要 2-3 步才能见到第 1 次正反馈）。所以我把 Stage 2 拆成**三个 Phase**，每个 Phase 的目标、信号来源、通过标准都不同：

| Phase | steps | 目的 | 主要信号 | 不看 |
|---|---|---|---|---|
| A (冒烟) | 10 | 管线能端到端 | 无 OOM / loss 不 NaN / tracker 正常落盘 | reward/success 提升 |
| B (调参) | 50 × N 组 | 找本地最优参数组合 | reward / format_compliance 曲线**斜率**、KL、entropy | 最终 success_rate 的数字大小 |
| C (冲刺) | 200 | 产出最终 checkpoint | `eval/success_rate` 单调上升、checkpoint 成功保存 | — |

> 核心方法论：**小规模探索 + 控制变量 + 大规模冲刺**。每次 Phase B 对比中只改 1 个变量；Phase B 的所有组必须在同一个 `--seed` 下跑以保证公平。

### 5.2 Phase A：冒烟（Smoke Run）

#### 目的
验证"**当前 Blocker 修完后**（`scripts/train.py:254` 的 `assert False` 已删），训练能不能从 step 1 跑到 step 10 不崩溃"。

#### 推荐参数

```bash
python scripts/train.py --exp_name smoke_a1 --env frozenlake --trainer starpo --algo ppo --total_training_steps 10 --eval_interval 5 --eval_episodes 5 --save_interval 10
```

其余参数用 `parse_args` 默认值即可。

#### 关注指标

看 `stdout.txt` + `logs/smoke_a1_metrics.jsonl`：
- `train/actor_loss` / `train/critic_loss`：有限浮点、不变 NaN；
- `train/kl_penalty` / `train/approx_kl`：不爆炸（< 1.0 量级）；
- `train/raw_reward_mean`：不强求提升，但不应恒为 0；
- `timing/rollout_sec` / `timing/update_sec`：有记录即可，用来估算 Phase B 预算。

#### 通过标准

| ✓ / ✗ | 标准 |
|---|---|
| ✓ | 10 个 training step 全部跑完 |
| ✓ | 无 OOM（`CUDA out of memory`） |
| ✓ | 无 NaN / inf 出现在 loss |
| ✓ | `logs/smoke_a1_metrics.jsonl` 至少 10 行 `step` 条目 + 2 行 `_event: summary`（对应 step=5, step=10 的 eval）|
| ✓ | `checkpoints/smoke_a1_final/` 存在 |

任一失败 → 先修管线，不要进 Phase B。

#### 时间预算

~30-40 分钟。

---

### 5.3 Phase B：控制变量调参

#### 目的

在 50-step 的预算内，用**一组最小化的对照实验**找出：
1. 本地硬件下的**最大可行 `prompt_batch_size`**（直接决定 effective batch 和 StarPO variance filter 的有效性）；
2. 适合稀疏奖励 FrozenLake 的 **`max_turn`**；
3. 是否开启 `bi_level_gae`；
4. （可选）`format_penalty` / `num_rollouts` / `variance_filter_ratio` 的调整必要性。

#### ⚠️ 关键发现：`prompt_batch_size=1` 会让 StarPO 退化

在进 Phase B 之前先明确一个**容易踩的坑**。`ragen_core/trajectory_buffer.py::filter_by_variance()` 的工作原理是：

```python
num_groups = len(self.trajectories) // group_size   # group_size = num_rollouts
num_retain = max(1, int(num_groups * retain_ratio)) # retain_ratio=0.25
```

而一次训练 step 内 `len(trajectories) = prompt_batch_size × num_rollouts`，因此：

- `num_groups = prompt_batch_size`
- 当 `prompt_batch_size = 1` 时，`num_groups = 1`，`num_retain = max(1, int(1 * 0.25)) = 1`

**结论**：只要 `prompt_batch_size = 1`，无论 `variance_filter_ratio` 设成什么，**variance filter 永远是 no-op**（100% 保留）。此时 StarPO 退化为 **"PureRL + format penalty"**，论文的核心创新点完全没生效。

对照 RAGEN 论文 §6.2 "Prompt Diversity" 消融：论文在固定总 rollout 数（如 128）的前提下扫描了 `env_groups × group_size` 的比例，最终推荐 **`env_groups=8, group_size=16`**（本仓库对应 `prompt_batch_size=8, num_rollouts=16`）。本地 8GB VRAM 显然跑不了，但**至少要把 `prompt_batch_size ≥ 2`** 才能让 filter 起作用。

**重要观察**：`prompt_batch_size` 增大**不会显著增加 rollout 阶段的 VRAM 峰值**（rollouts 是串行采样的，actor + ref 常驻占用恒定），真正吃 VRAM 的是 update 阶段的 `micro_batch_size × max_seq_length`。所以**在 `micro_batch_size` 不变的前提下，`prompt_batch_size` 从 1 升到 2~4 的硬件成本几乎为零**。

基于此，本文档的 Safe Baseline 已将 `prompt_batch_size` 从 1 提升到 **2**，相应把 `num_rollouts` 从 8 降到 **4**（总 rollout 预算 = 8 不变，时间预算基本持平）。

#### ⚙️ 工程优化：Gradient Checkpointing + Gradient Accumulation（硬编码启用）

两个**独立的**显存优化技术，为了让本地 8GB VRAM 能跑出接近论文的 effective batch size，**本项目直接硬编码启用两者**，不提供开关：

| 技术 | 作用 | 代码位置 |
|---|---|---|
| **Gradient Checkpointing** | forward 只保留 √L 层激活，backward 时重算。激活显存 ×~0.5，wall-clock +~30%。 | `rl_algos/ppo.py` / `rl_algos/grpo.py` 在 `self.actor = ...` 后立即 `gradient_checkpointing_enable(use_reentrant=False)`。仅 `training=True` 时生效，rollout 阶段（`eval()` + `generate()`）不受影响、KV cache 正常工作。|
| **Gradient Accumulation** | N 个 micro_batch 的梯度累加、N 步才 `optimizer.step()` 一次。数学上等价于"batch × N"，显存峰值不变，wall-clock × N。| 由 `--gradient_accumulation` 控制；`=1` 即回退成"每个 micro_batch 立即 step"。训练循环内按 `accum_counter % accum == 0 or is_last` 判断 step 时机（参考 HF Accelerate）。|

**参数命名变更**（本次工程改造）：
- 删除了 `--mini_batch_size`（旧设计里它同时是"forward 尺寸"+"step 粒度"，概念混合）
- 新增 `--micro_batch_size`（对齐 RAGEN `ppo_micro_batch_size_per_gpu`）
- 新增 `--gradient_accumulation`（对齐 RAGEN `ppo_mini_batch_size // micro_batch_size`）
- 等效 **`mini_batch_size` = `micro_batch_size × gradient_accumulation`**（算法内部仅作日志用，乘法派生，避免除不尽问题）

**典型组合对应关系**（8GB VRAM 下可达）：

| `--micro_batch_size` | `--gradient_accumulation` | 等效 mini_batch | 适用阶段 |
|---|---|---|---|
| 2 | 1 | 2 | 极保守（等同于未改造前的旧行为）|
| 2 | **4** | **8** | **Safe Baseline**（本文档推荐）|
| 2 | 8 | 16 | Phase C 冲刺可尝试 |
| 4 | 4 | 16 | 需要 Checkpointing + `max_seq_length=1536` + `--no_use_ref` 组合才能跑 |

> ⚠️ **注意事项**：`micro_batch_size × gradient_accumulation` 的乘积不应超过 update 阶段实际轨迹数 `filtered_batch = max(1, int(P × 0.25)) × R`；超过也不会崩，但每 epoch 可能只产出一次 step（等价 full-batch）。详见下方"起跑线"下的 B5 行备注。

#### 起跑线（"Safe Baseline"）

**以下参数是 Phase B 各组的公共起点**（只在每组变更其中 1 个）：

| 参数 | 值 | 理由 |
|---|---|---|
| `--env` | `frozenlake` | 主力环境 |
| `--trainer` | `starpo` | 主算法 |
| `--algo` | `ppo` | Critic 提供更稳的 advantage，小规模调参阶段更可读 |
| `--total_training_steps` | 50 | Phase B 固定预算 |
| `--eval_interval` | 10 | 每 10 步 eval 一次，共 5 个 eval 点 |
| `--eval_episodes` | 5 | 趋势感知足够，不追求精度 |
| `--save_interval` | 50 | 只存最终 |
| `--max_turn` | **5** | **改回论文值**（当前默认 2 偏短）|
| `--num_rollouts` | **4** | 给 `prompt_batch_size` 腾出 total-rollout 预算（原 8，见上方「关键发现」）|
| `--prompt_batch_size` | **2** | ≥2 才能让 StarPO 的 variance filter 真正工作；P=1 会让 StarPO 退化 |
| `--micro_batch_size` | 2 | 单次 forward/backward 的样本数（VRAM 峰值驱动因子）|
| `--gradient_accumulation` | **4** | 梯度累积步数，等效逻辑 mini_batch = 2 × 4 = 8；对齐 RAGEN 的 "micro + accum" 习惯；`=1` 即关闭累积 |
| `--max_new_tokens` | 256 | FrozenLake 单 turn 够用 |
| `--max_seq_length` | 2048 | 从 4096 降下来省 VRAM（实际 FrozenLake 用不到 4096） |
| `--use_format_reward` | on | 推动格式学习 |
| `--format_penalty` | −0.1 | 论文值 |
| `--variance_filter_ratio` | 0.25 | 论文值 |
| `--bi_level_gae` | **on** | 论文默认，零 VRAM 成本 |
| `--use_ref` | on | 论文默认 |
| `--optimizer` | `adamw8bit` | 必须 |
| 其他 RL 超参 | `learning_rate=1e-6, kl_coef=0.001, clip_ratio=0.2, vf_coef=0.5` | 全对齐论文 |

#### 对照组设计（**推荐第一批**，可扩展）

> **命名规则**：`phaseB_<variable>_<value>`

| 组号 | exp_name | 改了什么 | 对比谁 | 要回答的问题 |
|---|---|---|---|---|
| **核心组（必跑）** ||||
| B1 | `phaseB_baseline` | 无（Safe Baseline = P=2, R=4）| — | 新基准曲线（filter 生效版）|
| B2 | `phaseB_maxturn3` | `--max_turn 3` | B1 | 多 1 轮试错机会是否值得多 50% rollout 时间？ |
| B3 | `phaseB_maxturn2` | `--max_turn 2` | B1, B2 | 当前默认 2 是否严重限制学习？（很可能是）|
| B4 | `phaseB_PR_1x8_nofilter` | `--prompt_batch_size 1 --num_rollouts 8` | B1 | **回到原 1×8，验证"filter 失效"到底有没有影响**。若 B4 ≈ B1，说明 filter 可忽略；若 B1 显著优于 B4，说明 StarPO 的设计确实带来增益 —— 这是本次复现**最值得写进报告的核心问题**之一 |
| B5 | `phaseB_PR_4x2` | `--prompt_batch_size 4 --num_rollouts 2` | B1 | P 进一步增大、R 减半：filter 强度翻倍（25% × 4 groups），但 GRPO 的 group baseline 只用 2 条 rollout 估计方差，噪声大 —— **仅建议配 `--algo ppo` 跑** |
| B6 | `phaseB_bilevel_off` | 不加 `--bi_level_gae` | B1 | bi-level GAE 的实际收益 |
| **扩展候选（视 B1-B6 结果 + VRAM 富余决定是否跑）** ||||
| B7 | `phaseB_PR_4x4_bigger` | `--prompt_batch_size 4 --num_rollouts 4`（总数 16，翻倍）| B1 | 最接近论文比例的配置；filter 强过滤 + R=4 方差估计稳定 |
| B8 | `phaseB_fmtpenalty_strong` | `--format_penalty -0.2` | B1 | 加强格式惩罚能否更快拉起 `format_compliance`？ |
| B9 | `phaseB_novariancefilter` | `--variance_filter_ratio 1.0` | B1 | 关掉 filter、与 B1 对比 filter 的真实贡献（**必须在 P≥2 下做才有意义**；P=1 下这组等价 B1 没意义）|

#### 命令模板

```bash
# B1: baseline（P=2, R=4；filter 生效）
python scripts/train.py --exp_name phaseB_baseline --env frozenlake --trainer starpo --algo ppo --total_training_steps 50 --eval_interval 10 --eval_episodes 5 --save_interval 50 --max_turn 5 --prompt_batch_size 2 --num_rollouts 4 --micro_batch_size 2 --gradient_accumulation 4 --max_new_tokens 256 --max_seq_length 2048 --format_penalty -0.1 --variance_filter_ratio 0.25 --bi_level_gae --learning_rate 1e-6 --kl_coef 0.001

# B2: max_turn=3
python scripts/train.py --exp_name phaseB_maxturn3 --env frozenlake --trainer starpo --algo ppo --total_training_steps 50 --eval_interval 10 --eval_episodes 5 --save_interval 50 --max_turn 3 --prompt_batch_size 2 --num_rollouts 4 --micro_batch_size 2 --gradient_accumulation 4 --max_new_tokens 256 --max_seq_length 2048 --format_penalty -0.1 --variance_filter_ratio 0.25 --bi_level_gae --learning_rate 1e-6 --kl_coef 0.001

# B3: max_turn=2
python scripts/train.py --exp_name phaseB_maxturn2 --env frozenlake --trainer starpo --algo ppo --total_training_steps 50 --eval_interval 10 --eval_episodes 5 --save_interval 50 --max_turn 2 --prompt_batch_size 2 --num_rollouts 4 --micro_batch_size 2 --gradient_accumulation 4 --max_new_tokens 256 --max_seq_length 2048 --format_penalty -0.1 --variance_filter_ratio 0.25 --bi_level_gae --learning_rate 1e-6 --kl_coef 0.001

# B4: P=1, R=8（回到原组合，对照 filter 失效的实际影响）
python scripts/train.py --exp_name phaseB_PR_1x8_nofilter --env frozenlake --trainer starpo --algo ppo --total_training_steps 50 --eval_interval 10 --eval_episodes 5 --save_interval 50 --max_turn 5 --prompt_batch_size 1 --num_rollouts 8 --micro_batch_size 2 --gradient_accumulation 4 --max_new_tokens 256 --max_seq_length 2048 --format_penalty -0.1 --variance_filter_ratio 0.25 --bi_level_gae --learning_rate 1e-6 --kl_coef 0.001

# B5: P=4, R=2（PPO 专属；GRPO 勿用）
python scripts/train.py --exp_name phaseB_PR_4x2 --env frozenlake --trainer starpo --algo ppo --total_training_steps 50 --eval_interval 10 --eval_episodes 5 --save_interval 50 --max_turn 5 --prompt_batch_size 4 --num_rollouts 2 --micro_batch_size 2 --gradient_accumulation 4 --max_new_tokens 256 --max_seq_length 2048 --format_penalty -0.1 --variance_filter_ratio 0.25 --bi_level_gae --learning_rate 1e-6 --kl_coef 0.001
```

#### 关注指标（按**优先级**从高到低）

**一级信号（决定谁赢）**：
- `train/raw_reward_mean` 的**斜率**（回归直线斜率 > 0）
- `eval/format_compliance` 的 **5 个 eval 点**是否单调上升
- `eval/reward_variance` **上升**意味着"有些 rollout 开始能拿到奖励"，这是好事（并非越低越好）
- `train/entropy` 不能**急剧塌缩**（`echo_trap_sign` 应为 False）

**二级信号（作为一级的辅助）**：
- `train/approx_kl` < 0.5 即正常
- `train/clip_frac` 在 0.05-0.3 之间
- `timing/rollout_sec` + `timing/update_sec`：累计用来估算 Phase C 的 200 步时间

**三级信号（不看绝对值，只看是否出现）**：
- `eval/success_rate`：50 步预算下很可能还是 0，**不作为 Phase B 决策依据**

#### 通过标准 / 对比方法

每组跑完后，用 pandas 读 `logs/phaseB_*_metrics.jsonl`，画 3 张图：
1. `train/raw_reward_mean` vs `step`
2. `eval/format_compliance` vs `step`
3. `eval/reward_variance` vs `step`

**赢家判定规则**（按优先级）：
1. `format_compliance` 终值更高者胜；
2. 并列时 `raw_reward_mean` 终值更高者胜；
3. 仍并列时 `reward_variance` **更高**者胜（意味着"至少见过正 reward"）；
4. 若所有信号都持平，则选**时间更短**的组（Phase C 资源有限）。

**最坏情况**（所有组所有指标都没改善）：
- 检查是否 `learning_rate` 过小（本地 `num_rollouts × prompt_batch × ppo_epochs` 组合下可能需要轻微上调到 `2e-6` 或 `3e-6`），作为 **Phase B.5 补跑**。
- 检查 `format_penalty` 是否太轻，试 `-0.2`。
- 若仍无信号，回退到 Stage 1 定性结论，考虑任务难度（改 `--env bandit` 这种单步环境先跑通学习信号，再回 FrozenLake）。

#### 时间预算

- 单组 50 步 ≈ 2.5-3 小时（按单 step 3-4 分钟估）
- 核心组 B1-B6 共 6 组 ≈ 15-18 小时（可分 2-3 天跑）
- 若预算紧张，**最优先跑 B1 / B3 / B4**：分别对应"新基准 (P=2×R=4)" / "默认 max_turn=2 是否过短" / "回到原 P=1×R=8 对照 filter 实际贡献" —— 这三组直接回答报告的核心问题（max_turn 选型 + StarPO 的 filter 到底有没有用）

---

### 5.4 Phase C：冲刺训练

#### 目的

用 Phase B 选出的最优参数组合，跑完整的 200-step 训练，产出**可用于 Stage 3 评估**的最终 checkpoint。

#### 参数

= Phase B 赢家的全部参数，**只改 3 个**：
- `--total_training_steps 200`
- `--eval_interval 20`
- `--save_interval 50`（多存几个中间 checkpoint 便于对比训练曲线）

建议同时多产出一份 ablation：
- **主跑**：`phaseC_final_ppo_starpo`（最优组合）
- **配对**：`phaseC_final_grpo_starpo`（只把 `--algo` 改成 `grpo`，给 Stage 3 做 PPO vs GRPO 对比用）
- **对照**：`phaseC_final_ppo_pure`（`--trainer pure`，给 Stage 3 做 StarPO vs PureRL 对比用）

#### 关注指标

与 Phase B 相同，但**现在 `eval/success_rate` 必须纳入观察**。预期：
- `eval/success_rate` 在 step=50~100 附近开始 > 0；
- `eval/format_compliance` 到 step=200 时应 > 0.5；
- `train/raw_reward_mean` 呈整体上升，允许中途波动。

#### 通过标准

| ✓ / ✗ | 标准 |
|---|---|
| ✓ | 200 步全跑完 |
| ✓ | `checkpoints/phaseC_final_*/` 至少保存 1 个 checkpoint |
| ✓ | `eval/format_compliance` (final) > Stage 1 base 对应值（Stage 1 是 0.0，几乎必定 ✓） |
| ⭐ | `eval/success_rate` (final) > 0.1（期望目标，不是必达）|
| ⭐ | `train/entropy` 未出现 Echo Trap（`echo_trap_sign=False` 全程） |

#### 时间预算

~10-13 小时 × 3 组（PPO-StarPO / GRPO-StarPO / PPO-PureRL） ≈ 30-40 小时。建议分 2-3 天分批跑。

---

## 6. Stage 3：正式评估实验（demo / 占位）

> ⚠️ **本章节为 demo 骨架**。Phase C 产出 checkpoint 之后，根据实际情况补充对比矩阵、复现命令、结果表格。

### 6.1 目的

用 Stage 2 训练出的 checkpoint，在**与 Stage 1 完全一致的评估配置下**跑正式评估（更大 `--episodes`），得到可以写进报告的数字。

### 6.2 对比维度（骨架，待扩充）

| 维度 | 取值 | 说明 |
|---|---|---|
| 模型规模 | 0.5B / 1.5B | 主力 0.5B，1.5B 仅做 base 对照（不训练，硬件不允许） |
| 训练状态 | base / trained | base = HF 原模型；trained = Phase C 产出 |
| RL 算法 | PPO / GRPO | Phase C 配对跑出 |
| 训练器 | StarPO / PureRL | Phase C 对照跑出 |
| 评估环境 | frozenlake / (待定) | 主报告主力 frozenlake；若时间允许加 bandit / math |
| 评估种子 | `seed=42` (默认) | seed 决定论保证跨模型同关卡 |
| **TODO: 补充维度** | | 例如 format 单开 / 关 ablation、max_turn eval-time 扫描等 |

### 6.3 推荐对比矩阵（骨架）

> 所有评估都用：`--episodes 50 --max_turn 5 --max_env_steps 20 --temperature 0.5`。50 ep 的 95% Wilson CI ≈ ±14%，足够支撑大部分对比。

| 实验编号 | 模型 | source | trainer/algo | 对比对象 | 核心问题 |
|---|---|---|---|---|---|
| E1 | 0.5B | base | — | E2, E3, E4 | Stage 1 已跑，作为下界（复跑 50ep 得高置信版本）|
| E2 | 0.5B | trained | StarPO + PPO | E1 | **论文复现核心 claim**：训练是否有效 |
| E3 | 0.5B | trained | StarPO + GRPO | E2 | PPO vs GRPO 在稀疏奖励下谁更稳 |
| E4 | 0.5B | trained | PureRL + PPO | E2 | StarPO 的 variance filter + format penalty 是否真的有帮助 |
| E5 | 1.5B | base | — | E2 | 0.5B trained 能否 **超过** 1.5B base（RAGEN 论文核心卖点之一）|
| **TODO** | | | | | 更多 ablation / cross-env / 中间 checkpoint 曲线 |

### 6.4 命令模板（骨架）

```bash
# E1: 0.5B base （50 ep 高置信版本）
python scripts/evaluate.py --exp_name eval_E1_0p5b_base --env frozenlake --episodes 50 --max_turn 5 --max_env_steps 20 --temperature 0.5 --model_source base --model_name Qwen/Qwen2.5-0.5B-Instruct

# E2: 0.5B trained (StarPO + PPO, Phase C 产出)
python scripts/evaluate.py --exp_name eval_E2_0p5b_starpo_ppo --env frozenlake --episodes 50 --max_turn 5 --max_env_steps 20 --temperature 0.5 --model_source trained --model_name phaseC_final_ppo_starpo_final

# TODO: E3, E4, E5 同构
```

### 6.5 预期产出

1. **主表**（报告核心 Table）：E1-E5 的 8 项 RAGEN 指标 × 5 行，直接复制到报告。
2. **训练曲线图**（Phase C 的 tracker JSONL 重画）：`eval/success_rate` / `eval/format_compliance` / `train/raw_reward_mean` vs step，StarPO 和 PureRL 同图对比。
3. **轨迹案例**（定性分析章节用）：挑 3-5 条典型 trajectory，展示 trained 模型的 `<think>` 推理质量。当前 tracker 不存 trajectory 原文，需要 Stage 3 执行时在 `evaluate.py` 里临时加 `--save_trajectories` flag（**待实现**）。

### 6.6 TODO 清单（Stage 3 开工前补齐）

- [ ] 补充更多 ablation 维度（e.g. `kl_coef=0` 的 `ppo-nokl` 对照）
- [ ] 决定是否加 CartPole / Bandit 作为跨环境泛化证据
- [ ] 决定最终报告的对比基准（只对 0.5B base？也对 1.5B base？）
- [ ] 设计统计显著性测试方案（Fisher exact / bootstrap CI）
- [ ] 轨迹保存功能（给报告的定性分析章节用）

---

## 7. 数据 / 日志 / 产物目录约定

```
ragen_ward/
├── stdout.txt                                # 每次 train/evaluate 覆盖写入，只保留最近一次
├── logs/
│   ├── <exp_name>.log                        # loguru 分级日志
│   └── <exp_name>_metrics.jsonl              # 每步 metric + _event: run_start/summary/run_end
│                                             # ⚠️ append 模式，如果 exp_name 重用会累积
├── checkpoints/
│   └── <exp_name>_final/                     # algo.save() 产物，供 --model_source trained 读取
│                                             # 多 checkpoint（save_interval<total_steps 时）在此同目录
└── EXPERIMENTS.md                            # 本文档
```

**强规则**：
- **每个 Phase / 每组对照组 必须用不同 `--exp_name`**，否则 `eval_<name>_metrics.jsonl` 会 append 混在一起，事后非常难切分。
- Phase B 的命名约定：`phaseB_<variable>_<value>`
- Phase C 的命名约定：`phaseC_final_<algo>_<trainer>`
- Stage 3 评估的命名约定：`eval_<实验编号>_<简短描述>`

---

## 8. 风险 / 常见问题清单（持续补充）

| 现象 | 可能原因 | 应对 |
|---|---|---|
| Phase A 直接 OOM | `max_seq_length=4096` 太大 / `use_ref=True` 占 ~1GB / `micro_batch_size` 过大 | 顺序尝试：①`--max_seq_length 1536`；②`--micro_batch_size 1 --gradient_accumulation 8`（等效 batch 不变但峰值减半）；③`--no_use_ref`（强制 kl_coef=0，影响稳定性，兜底）|
| Phase B 某组 loss NaN | `learning_rate` 与 `optimizer` 不匹配 / 极端梯度 | 降 lr 到 `5e-7`；或加 `--target_kl 0.5` 早停 |
| Phase B 所有组都看不到学习信号 | 稀疏奖励 + 小 batch 噪声太大 | 先跑 `--env bandit`（单步任务，信号致密）验证管线是否有效；或 lr 上调到 `3e-6` 重试 |
| `format_compliance` 始终 0 | format penalty 不够强 / lr 太小 | 试 `--format_penalty -0.2` + `--learning_rate 2e-6` |
| `checkpoints/<name>_final/` 加载失败 | save/load 接口对 optimizer state 处理 | 报告里注明仅加载 model weights，evaluate 不需要 optimizer state |
| 跨 Phase 同名 exp_name 导致 JSONL 混合 | 命名约定未严格遵守 | 事后用 `_event: run_start` 切分；下次避免复用 name |
| `prompt_batch_size=1` 下 StarPO 退化为 PureRL | `num_groups = P = 1`，`num_retain = max(1, 0.25)=1`，variance filter 永远保留 100% | **Phase B 起保证 `prompt_batch_size ≥ 2`**；Safe Baseline 已默认 P=2, R=4（见 §5.3 关键发现）|
| **TODO**：跑到 Phase C 时再补 | | |

---

## 9. 变更日志

| 日期 | Stage | 事项 |
|---|---|---|
| 2026-04-18 | Stage 1 | 0.5B / 1.5B base 各跑 10 ep 完成，数据见 §4.3 |
| 2026-04-18 | Stage 1 | 修复 FrozenLake/CartPole 的 `is_success` / `action_effective` 口径（P0 + P1）|
| 2026-04-18 | — | 本文档初稿，覆盖 Stage 1 实测 + Stage 2 完整方案 + Stage 3 骨架 |
| 2026-04-18 | Stage 2 | 发现 `prompt_batch_size=1` 下 StarPO variance filter 失效，Safe Baseline 调整为 P=2 × R=4；Phase B 对照组重排为 B1-B6 核心 + B7-B9 扩展，新增 B4 = 原 P=1×R=8 作为 "filter 失效"对照组 |
| 2026-04-18 | 工程 | **硬编码启用 Gradient Checkpointing + Gradient Accumulation**；CLI 参数 `--mini_batch_size` → `--micro_batch_size + --gradient_accumulation`（乘法派生等效 mini_batch，对齐 RAGEN `ppo_micro_batch_size_per_gpu × accum`）；Safe Baseline 默认 `2 × 4 = 8`，可把 effective batch 推向论文级 |
| **TODO** | Stage 2 Phase A | 冒烟结果 |
| **TODO** | Stage 2 Phase B | 各组对比结果 + 赢家判定 |
| **TODO** | Stage 2 Phase C | 最终 checkpoint + 训练曲线 |
| **TODO** | Stage 3 | 评估矩阵结果 |

---

## 10. 快速参考：命令速查

```bash
# Stage 1 复跑（如需）
python scripts/evaluate.py --env frozenlake --episodes 10 --max_turn 5 --max_env_steps 20 --temperature 0.5 --exp_name eval_S1_0p5b_base --model_name Qwen/Qwen2.5-0.5B-Instruct

# Stage 2 Phase A: 冒烟
python scripts/train.py --exp_name smoke_a1 --total_training_steps 10 --eval_interval 5 --eval_episodes 5 --save_interval 10

# Stage 2 Phase B: 对照组（见 §5.3 完整命令）
python scripts/train.py --exp_name phaseB_<name> --total_training_steps 50 --eval_interval 10 --eval_episodes 5 --save_interval 50 [其他参数]

# Stage 2 Phase C: 冲刺
python scripts/train.py --exp_name phaseC_final_ppo_starpo --total_training_steps 200 --eval_interval 20 --save_interval 50 [Phase B 赢家参数]

# Stage 3: 正式评估
python scripts/evaluate.py --exp_name eval_E<N>_<desc> --episodes 50 [与 Stage 1 一致的其他参数] --model_source trained --model_name phaseC_final_<...>_final
```

---

*Last updated: 2026-04-18. Living document — 请在每个 Phase 结束后更新 §9 变更日志。*
