"""
评估入口 / Evaluation Entry
-----------------------------------
用法（一条命令跑默认配置）：

    python scripts/evaluate.py

所有默认值都集中在本文件的 ``parse_args`` 里。与 ``scripts/train.py`` 一样，
此脚本不再依赖任何 dataclass 默认值或 ``getattr`` fallback——缺字段就直接
AttributeError，避免静默跑出错误结果。

与 RAGEN 原论文 eval pipeline 的对齐（以 ``RAGEN-main/ragen/eval.py`` +
``es_manager.get_rollout_states`` 为参照）：

* **rollout 口径统一**：复用 ``ragen_core.rollout_one_trajectory``，和训练器
  ``StarPOTrainer.evaluate`` / ``PureRLTrainer.evaluate`` 完全等价的上下文
  构造（env_instruction 注入、max_turn 硬截断、terminated/truncated 双停止条件）。
* **成功判定严格 RAGEN 口径**：``terminated and not truncated``（见
  ``ragen_core.rollout_utils.judge_success``）；若 env 在 info 里显式写了
  ``is_success`` / ``success`` 字段以 info 为准（兼容 Bandit/Math/Sokoban）。
  不再用 ``ep_reward > 0`` 这种会把 CartPole 全部算成功、把 Bandit safe-arm
  也算成功的错误口径。
* **指标集对齐 RAGEN**：success_rate / avg_reward / avg_trajectory_length /
  avg_num_actions / action_valid_rate / action_effective_rate /
  format_compliance / reward_variance。
* **结果持久化**：通过 ``TrainingTracker`` 把 per-episode + summary 写入
  ``logs/eval_<exp_name>_metrics.jsonl``，方便 pandas 事后对比不同 ckpt。
"""

import argparse
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 控制台 tee —— 必须先于 loguru / transformers 等会缓存 sys.stderr 的库 import。
# 让本次运行的所有 stdout + stderr 同步写入 <PROJECT_ROOT>/stdout.txt。
# mode="w": 每次启动清空文件，只保留"最近一次"运行（与 scripts/train.py 共用此文件）。
from utils.stdout_tee import setup_stdout_tee  # noqa: E402
setup_stdout_tee("eval_stdout.txt", mode="w")

from configs.config import EnvConfig, AgentConfig
from configs.constants import CKPT_DIR
from envs import make_env
from agents.hf_agent import HFAgent
from agents.openai_agent import OpenAIAgent
from evaluation.metrics import EvaluatorMetrics, compute_reward_variance
from ragen_core.rollout_utils import judge_success, rollout_one_trajectory
from utils.logger import setup_logger, logger
from utils.tracker import TrainingTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RAGEN-Ward framework")
    # 运行总控
    p.add_argument("--exp_name", type=str, default="eval_trained_30step ")
    p.add_argument("--episodes", type=int, default=50,
                   help="Number of episodes to evaluate (RAGEN paper uses 256-512; "
                        "50 is a practical balance for single-GPU HF models)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed for the random generator that draws per-episode env seeds.")
    p.add_argument("--fixed_seed_base", type=int, default=None,
                   help="If set, use deterministic env seeds as `base + ep` (for reproducible "
                        "debugging). If left None (default), each episode uses a random seed "
                        "— this matches StarPOTrainer.evaluate / RAGEN paper conventions.")

    # 环境
    p.add_argument("--env", type=str, default="frozenlake",
                   choices=["math", "cartpole", "frozenlake", "sokoban", "bandit"])
    p.add_argument("--max_env_steps", type=int, default=10,
                   help="Max atomic env steps per episode (environment-level truncation)")
    p.add_argument("--max_turn", type=int, default=5,
                   help="Max LLM turns (chat_request calls) per trajectory. Aligns with "
                        "RAGEN `agent_proxy.max_turn=5` in config/eval.yaml. Independent "
                        "from --max_env_steps; whichever hits first truncates the episode.")

    # Agent
    p.add_argument("--agent", type=str, default="hf", choices=["hf", "openai"])
    p.add_argument("--model_source", type=str, default="base", choices=["base", "trained"],
                   help="base = HF repo id / cached weight under models/; "
                        "trained = checkpoint dir under checkpoints/")
    p.add_argument("--model_name", type=str, default="Qwen_Qwen2.5-1.5B-Instruct",
                   help="HF repo id (base mode) or checkpoint folder name (trained mode)")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="0.0 = greedy decoding (aligns with RAGEN API eval); "
                        "RAGEN local val uses 0.5 if you want light exploration.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    # 注：system prompt 由环境类持有（envs/base_env.py::BaseEnv.agent_system_prompt
    # + 各子类覆盖），不再做成 CLI 参数，保证 train/eval 两侧口径自动一致。

    # OpenAI 专用（可选）
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--base_url", type=str, default=None)
    return p.parse_args()


def _make_agent(args: argparse.Namespace) -> tuple[AgentConfig, object]:
    """解析模型路径 + 构造 AgentConfig + 实例化 agent。"""
    if args.model_source == "trained":
        model_path = os.path.join(CKPT_DIR, args.model_name)
        if not os.path.exists(model_path):
            logger.warning(
                f"Trained checkpoint directory {model_path} not found. Attempting to load anyway."
            )
    else:
        model_path = args.model_name

    agent_cfg = AgentConfig(
        agent_type=args.agent,
        model_name_or_path=model_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    agent = HFAgent(agent_cfg) if args.agent == "hf" else OpenAIAgent(agent_cfg)
    return agent_cfg, agent


def main() -> int:
    args = parse_args()
    setup_logger(level="INFO", log_file=f"{args.exp_name}.log")
    logger.info(
        f"Starting Evaluation | env={args.env} agent={args.agent} "
        f"source={args.model_source} model={args.model_name} "
        f"episodes={args.episodes} max_turn={args.max_turn} "
        f"max_env_steps={args.max_env_steps} temperature={args.temperature}"
    )

    # 1) env + agent
    env_cfg = EnvConfig(
        env_name=args.env,
        max_steps=args.max_env_steps,
    )
    env = make_env(env_cfg)
    agent_cfg, agent = _make_agent(args)

    # 2) 指标聚合器 + 持久化 tracker
    em = EvaluatorMetrics()
    eval_rewards: list[float] = []
    tracker = TrainingTracker(exp_name=args.exp_name, use_wandb=False)

    # 3) seed 策略：默认随机（和 StarPOTrainer.evaluate 对齐），fixed_seed_base 仅用于复现 debug
    rng = random.Random(args.seed)

    # 4) 评估循环：每一轮一条完整 trajectory，由 ragen_core.rollout_one_trajectory 统一构造
    #    —— 与 StarPO/PureRL 训练器内评估逻辑同源，口径完全一致。
    for ep in range(args.episodes):
        if args.fixed_seed_base is not None:
            env_seed = args.fixed_seed_base + ep
        else:
            env_seed = rng.randint(0, 2**31 - 1)

        trajectory = rollout_one_trajectory(
            env=env,
            agent=agent,
            seed=env_seed,
            max_turn=args.max_turn,
            use_format_reward=False,  # 评估阶段只看原生环境 reward，不叠加 format penalty
            format_penalty=0.0,
        )
        if not trajectory:
            logger.warning(f"Episode {ep+1}/{args.episodes} produced an empty trajectory, skipped.")
            continue

        ep_reward = float(sum(step["env_reward"] for step in trajectory))
        success = judge_success(trajectory)

        em.add_episode_from_trajectory(trajectory, success=success)
        eval_rewards.append(ep_reward)

        # 写一行 per-episode JSONL（便于事后按 episode 粒度画图 / 对比不同 ckpt）
        # executed_action_count 是 BaseEnv.step 每 turn 覆盖写入 last_info 的字段，
        # 所以必须对整条 trajectory 累加，才得到 episode 级的原子动作数；否则只能拿到
        # "最后一个 turn 执行了几步" ——  和 summary 里 avg_num_actions（正确累加口径）不一致。
        episode_num_actions = int(sum(
            (step.get("info") or {}).get("executed_action_count", 1)
            for step in trajectory
        ))
        tracker.log(
            {
                "eval/episode_reward": ep_reward,
                "eval/episode_success": int(success),
                "eval/episode_length": len(trajectory),
                "eval/episode_num_actions": episode_num_actions,
                "eval/episode_seed": env_seed,
            },
            step=ep + 1,
        )
        logger.info(
            f"Episode {ep+1}/{args.episodes} | seed={env_seed} "
            f"reward={ep_reward:.4f} success={success} length={len(trajectory)}"
        )

    # 5) 汇总：对齐 RAGEN es_manager.get_rollout_states 的字段
    summary = em.summary()
    summary["eval/reward_variance"] = compute_reward_variance(eval_rewards)
    summary["eval/total_episodes"] = len(eval_rewards)

    logger.info("--- Evaluation Results (RAGEN-aligned) ---")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"  {k:38s} = {v:.4f}")
        else:
            logger.info(f"  {k:38s} = {v}")

    # summary 单独写一行，step 用 args.episodes 使整个 JSONL 的 step 列单调递增
    tracker.log({**summary, "_event": "summary"}, step=args.episodes)
    tracker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
