"""
评估入口 / Evaluation Entry
-----------------------------------
用法（一条命令跑默认配置）：

    python scripts/evaluate.py

所有默认值都集中在本文件的 ``parse_args`` 里。与 ``scripts/train.py`` 一样，
此脚本不再依赖任何 dataclass 默认值或 ``getattr`` fallback——缺字段就直接
AttributeError，避免静默跑出错误结果。
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import EnvConfig, AgentConfig
from configs.constants import CKPT_DIR
from envs import make_env
from agents.hf_agent import HFAgent
from agents.openai_agent import OpenAIAgent
from utils.logger import setup_logger, logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RAGEN-Ward framework")
    # 运行总控
    p.add_argument("--exp_name", type=str, default="eval_default")
    p.add_argument("--episodes", type=int, default=10,
                   help="Number of episodes to evaluate")

    # 环境
    p.add_argument("--env", type=str, default="frozenlake",
                   choices=["math", "cartpole", "frozenlake", "sokoban", "bandit"])
    p.add_argument("--max_env_steps", type=int, default=20,
                   help="Max steps per episode (environment-level truncation)")

    # Agent
    p.add_argument("--agent", type=str, default="hf", choices=["hf", "openai"])
    p.add_argument("--model_source", type=str, default="base", choices=["base", "trained"],
                   help="base = HF repo id / cached weight under models/; "
                        "trained = checkpoint dir under checkpoints/")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HF repo id (base mode) or checkpoint folder name (trained mode)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy decoding")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--system_prompt", type=str,
                   default="You are a helpful reinforcement learning agent.")

    # OpenAI 专用（可选）
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--base_url", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logger(level="INFO", log_file=f"{args.exp_name}.log")
    logger.info(
        f"Starting Evaluation | env={args.env} agent={args.agent} "
        f"source={args.model_source} model={args.model_name}"
    )

    # 1) 解析模型路径
    if args.model_source == "trained":
        model_path = os.path.join(CKPT_DIR, args.model_name)
        if not os.path.exists(model_path):
            logger.warning(f"Trained checkpoint directory {model_path} not found. "
                           f"Attempting to load anyway.")
    else:
        model_path = args.model_name

    # 2) 构造 config（所有字段显式传入）
    env_cfg = EnvConfig(
        env_name=args.env,
        max_steps=args.max_env_steps,
    )
    agent_cfg = AgentConfig(
        agent_type=args.agent,
        model_name_or_path=model_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # 3) 实例化
    env = make_env(env_cfg)
    agent = HFAgent(agent_cfg) if args.agent == "hf" else OpenAIAgent(agent_cfg)

    # 4) 评估循环
    success_count = 0
    total_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        messages = [
            {"role": "system", "content": agent_cfg.system_prompt},
            {"role": "user", "content": f"{obs}\n{env.get_valid_actions()}\nPlease reason step by step."},
        ]

        terminated, truncated = False, False
        ep_reward = 0.0
        while not (terminated or truncated):
            response = agent.chat_request(messages)
            next_obs, reward, terminated, truncated, info = env.step(response)
            ep_reward += reward

            obs = next_obs
            messages.append({"role": "assistant", "content": response})
            if not terminated:
                messages.append({
                    "role": "user",
                    "content": f"Observation: {obs}\nReward for last step: {reward}\nNext action?",
                })

        total_rewards.append(ep_reward)
        if ep_reward > 0:  # 假设正奖励代表成功，不同环境可按需改
            success_count += 1
        logger.info(f"Episode {ep+1}/{args.episodes} finished | reward={ep_reward}")

    # 5) 汇总
    success_rate = success_count / max(1, args.episodes)
    avg_reward = sum(total_rewards) / max(1, args.episodes)
    logger.info("--- Evaluation Results ---")
    logger.info(f"Success Rate: {success_rate * 100:.2f}%")
    logger.info(f"Average Reward: {avg_reward:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
