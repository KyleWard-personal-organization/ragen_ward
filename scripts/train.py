"""
训练入口 / Training Entry
-----------------------------------
Usage（一条命令跑默认配置，什么都不传）：

    python scripts/train.py

想改任何超参，直接在命令行传对应参数即可。**所有默认值都集中在本文件的
``parse_args`` 里**，是全项目唯一的默认值来源（Single Source of Truth）。

PowerShell 多行续行用反引号 `` ` ``，不是 Linux 的反斜杠 ``\``，示例：

    python scripts/train.py `
        --env frozenlake --trainer starpo --algo ppo `
        --total_training_steps 200

或者直接把所有参数写成一行最省事（下面 README 里的示例都用单行）。
"""

import argparse
import os
import sys
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 控制台 tee —— 必须先于 loguru / transformers 等会缓存 sys.stderr 的库 import。
# 让本次运行的所有 stdout + stderr 同步写入 <PROJECT_ROOT>/stdout.txt。
# mode="w": 每次启动清空文件，只保留"最近一次"运行（train/evaluate 共用此文件）。
from utils.stdout_tee import setup_stdout_tee  # noqa: E402
setup_stdout_tee("train_stdout.txt", mode="w")

from configs.config import (
    ExperimentConfig,
    EnvConfig,
    AgentConfig,
    RLAlgoConfig,
    RagenConfig,
)
from configs.constants import CKPT_DIR
from envs import make_env
from agents.hf_agent import HFAgent
from rl_algos import make_algo
from ragen_core.starpo_trainer import StarPOTrainer
from ragen_core.pure_rl_trainer import PureRLTrainer
from utils.logger import setup_logger, logger


# =============================================================================
# argparse：全项目唯一的默认值来源
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train RAGEN-Ward framework (all defaults defined here; "
                    "no defaults in dataclasses or getattr fallbacks)."
    )

    # ---- 运行总控 ----
    p.add_argument("--exp_name", type=str, default="ragen_baseline_0.5B_sparse_nofilter",
                   help="Experiment name used for log file / checkpoint dir naming")
    p.add_argument("--seed", type=int, default=42)

    # ---- 环境 ----
    p.add_argument("--env", type=str, default="frozenlake",
                   choices=["math", "cartpole", "frozenlake", "sokoban", "bandit"],
                   help="Environment to train on")
    p.add_argument("--max_env_steps", type=int, default=10,
                   help="Max atomic env steps per episode (environment-level truncation). "
                        "Aligns with RAGEN's max_actions_per_traj=10. Note: this counts atomic "
                        "env steps, not LLM turns — the model can emit '<answer>A || B || C</answer>' "
                        "to consume multiple steps in one turn.")

    # ---- Agent ----
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model path or repo id (local path preferred to skip download)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    # 注：system prompt 由环境类持有（envs/base_env.py::BaseEnv.agent_system_prompt
    # + 各子类覆盖），不再做成 CLI 参数，避免换 env 时忘记同步改 CLI。

    # ---- 训练器 / 算法选择 ----
    p.add_argument("--trainer", type=str, default="starpo", choices=["starpo", "pure"],
                   help="starpo = RAGEN StarPO; pure = baseline RL w/o variance filter / format penalty")
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "grpo"])

    # ---- 训练循环 ----
    p.add_argument("--total_training_steps", type=int, default=200,
                   help="Each step = 1 rollout phase + 1 RL update. "
                        "RAGEN paper main experiments use 200 steps; echo trap "
                        "typically emerges at step ~120-180 under the vanilla StarPO baseline, "
                        "so running the full 200 is required to reproduce it.")
    p.add_argument("--eval_interval", type=int, default=20,
                   help="Run evaluation every N training steps. Denser intervals (10-15) help "
                        "catch echo-trap onset (typically step 30-80 under PureRL baseline).")
    p.add_argument("--eval_episodes", type=int, default=200,
                   help="Number of episodes per evaluation")
    p.add_argument("--save_interval", type=int, default=50,
                   help="Save checkpoint every N training steps")

    # ---- RAGEN / StarPO 超参 ----
    p.add_argument("--mode", type=str, default="fast", choices=["fast", "slow"],
                   help="Reserved for future fast/slow implementation switching")
    p.add_argument("--num_rollouts", type=int, default=16,
                   help="Number of trajectories sampled per prompt (GRPO group size). "
                        "RAGEN paper uses 8; effective batch = prompt_batch_size * num_rollouts.")
    p.add_argument("--prompt_batch_size", type=int, default=8,
                   help="Number of different prompts per training step "
                        "(effective batch = prompt_batch_size * num_rollouts)")
    p.add_argument("--no_format_reward", dest="use_format_reward", action="store_false")
    p.add_argument("--format_penalty", type=float, default=-0.1)
    p.add_argument("--variance_filter_ratio", type=float, default=1.0,
                   help="Keep top-k ratio by group variance (StarPO-S; only used when trainer=starpo)")
    p.add_argument("--max_turn", type=int, default=5,
                   help="Max LLM turns (chat_request calls) per trajectory. Aligns with RAGEN "
                        "`agent_proxy.max_turn`. This is the *turn*-level budget, independent from "
                        "--max_env_steps (the atomic env-step budget): whichever hits first "
                        "truncates the episode. RAGEN main FrozenLake/Sokoban configs use 1 "
                        "(one-shot full-plan), but small models struggle under such sparse reward; "
                        "default=3 gives the agent a few retries while still capping rollout cost. "
                        "Single-turn envs (Bandit/Math) terminate after step 1 and are unaffected. "
                        "Must be >= 1.")

    # ---- RL 算法通用超参 ----
    p.add_argument("--learning_rate", type=float, default=1e-6,
                   help="Actor lr. RAGEN paper 0.5B baseline uses 1e-6.")
    p.add_argument("--critic_learning_rate", type=float, default=1e-5,
                   help="Critic lr. RAGEN paper 0.5B baseline uses 1e-5 (independent from actor lr; "
                        "verl framework runs critic as a separate backbone with its own optimizer). "
                        "Our PPO uses a Linear(hidden,1) value head sharing actor backbone, so we "
                        "enforce two-lr behavior via PyTorch param_groups: actor_params @ learning_rate, "
                        "critic_head @ critic_learning_rate. GRPO is critic-free and ignores this.")
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--micro_batch_size", type=int, default=1,
                   help="Per-backward sample count (VRAM peak driver). "
                        "Equivalent to RAGEN's ppo_micro_batch_size_per_gpu. "
                        "Setting >1 (vs 1) halves the number of forward+backward calls per step, "
                        "cutting update_sec roughly proportionally — at the cost of VRAM peak "
                        "scaling linearly. Tune down if OOM, up if VRAM has headroom.")
    p.add_argument("--gradient_accumulation", type=int, default=32,
                   help="Gradient accumulation steps; 1 disables accumulation (immediate step per micro-batch). "
                        "Logical mini-batch = micro_batch_size * gradient_accumulation, "
                        "matching RAGEN's ppo_mini_batch_size convention. "
                        "RAGEN paper uses ppo_mini_batch_size=32 for 0.5B FrozenLake/Sokoban — "
                        "we hit it via micro=2 × accum=16 = 32. Effect: with 64 traj/step (P=8 × R=8) "
                        "the number of optimizer steps drops from 8 → 2, giving lower-variance "
                        "gradients and a tighter PPO trust region.")
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=1.0,
                   help="Critic loss coefficient (PPO only). RAGEN/verl default = 1.0; "
                        "value < 1 effectively shrinks critic gradient relative to actor and "
                        "starves the value head — a previous default of 0.001 made critic "
                        "untrained, degrading advantage estimation to plain reward.")
    p.add_argument("--ent_coef", type=float, default=0.001)
    p.add_argument("--kl_coef", type=float, default=0.001,
                   help="KL divergence coefficient; 0.001 aligns with RAGEN main experiments, "
                        "0.0 corresponds to the ppo-nokl ablation setting")
    p.add_argument("--target_kl", type=float, default=None,
                   help="Early stopping KL threshold; leave unset to disable")
    p.add_argument("--max_seq_length", type=int, default=2048,
                   help="Max total token length after concatenating whole trajectory")

    p.add_argument("--no_use_ref", dest="use_ref", action="store_false",
                   help="Disable reference model entirely; forces kl_coef=0 and saves ~1GB VRAM, "
                        "at the cost of losing the KL anchor (risk of reward hacking / policy collapse).")
    p.add_argument("--optimizer", type=str, default="adamw8bit",
                   choices=["adamw", "adamw8bit", "adafactor"],
                   help="adamw = fp32 state (~8B/param, high VRAM); "
                        "adamw8bit = bitsandbytes 8-bit state (~2B/param, recommended on <=16GB GPUs; "
                        "auto applies 32-bit override on embedding/lm_head for stability); "
                        "adafactor = factored second moment (approximate AdamW, minimal state).")

    # ---- GAE 超参 ----
    p.add_argument("--gamma", type=float, default=1.0,
                   help="token-level discount")
    p.add_argument("--lam", type=float, default=1.0,
                   help="GAE lambda")
    p.add_argument("--no_bi_level_gae", action="store_false", dest="bi_level_gae",
                   help="Enable bi-level GAE (turn-level + token-level)")
    p.add_argument("--high_level_gamma", type=float, default=0.95,
                   help="Turn-level discount; only used when bi_level_gae is enabled")

    return p.parse_args()


# =============================================================================
# 把 argparse.Namespace 一次性翻译成 ExperimentConfig
# =============================================================================

def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """将 argparse 结果显式地注入到 dataclass——不依赖 dataclass 自身的默认值。"""
    # --- 最小合法性检查（argparse 不原生支持区间验证） ---
    if args.max_turn < 1:
        raise ValueError(
            f"--max_turn must be >= 1 (got {args.max_turn}); "
            "a trajectory with 0 LLM calls would produce no data."
        )
    if args.max_env_steps < 1:
        raise ValueError(f"--max_env_steps must be >= 1 (got {args.max_env_steps}).")

    env_cfg = EnvConfig(
        env_name=args.env,
        max_steps=args.max_env_steps,
    )
    agent_cfg = AgentConfig(
        agent_type="hf",
        model_name_or_path=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    algo_cfg = RLAlgoConfig(
        algo_name=args.algo,
        learning_rate=args.learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        gamma=args.gamma,
        lam=args.lam,
        bi_level_gae=args.bi_level_gae,
        high_level_gamma=args.high_level_gamma,
        ppo_epochs=args.ppo_epochs,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        clip_ratio=args.clip_ratio,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        kl_coef=args.kl_coef,
        max_seq_length=args.max_seq_length,
        use_ref=args.use_ref,
        optimizer=args.optimizer,
        target_kl=args.target_kl,
    )
    ragen_cfg = RagenConfig(
        mode=args.mode,
        num_rollouts=args.num_rollouts,
        use_format_reward=args.use_format_reward,
        format_penalty=args.format_penalty,
        variance_filter_ratio=args.variance_filter_ratio,
        max_turn=args.max_turn,
        prompt_batch_size=args.prompt_batch_size,
    )
    return ExperimentConfig(
        exp_name=args.exp_name,
        seed=args.seed,
        total_training_steps=args.total_training_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        env_config=env_cfg,
        agent_config=agent_cfg,
        rl_algo_config=algo_cfg,
        ragen_config=ragen_cfg,
    )


def _set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    setup_logger(level="INFO", log_file=f"{args.exp_name}.log")
    logger.info(f"args = {vars(args)}")

    _set_seed(args.seed)

    config = build_config(args)
    env = make_env(config.env_config)
    agent = HFAgent(config.agent_config)

    algo = make_algo(config.rl_algo_config, agent)

    if args.trainer == "starpo":
        trainer = StarPOTrainer(config, env, agent, algo)
    else:
        trainer = PureRLTrainer(config, env, agent, algo)

    logger.info(f"Starting {args.trainer.upper()} training with algo={args.algo.upper()}")
    try:
        trainer.run()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C). Saving latest model.")
    except Exception:
        logger.error("Training failed with exception:\n" + traceback.format_exc())
        raise

    save_path = os.path.join(CKPT_DIR, f"{args.exp_name}_final")
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
        logger.info(f"Detect existed model at {save_path}, and delete the previous folder.")
    try:
        algo.save(save_path)
        logger.info(f"Final model saved to {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save final model: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
