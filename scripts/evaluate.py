import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import EnvConfig, AgentConfig
from configs.constants import CKPT_DIR, MODELS_DIR
from envs import make_env
from agents.hf_agent import HFAgent
from agents.openai_agent import OpenAIAgent
from utils.logger import setup_logger, logger

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAGEN-Ward framework")
    parser.add_argument("--env", type=str, default="math", choices=["math", "cartpole", "frozenlake"],
                        help="Environment to evaluate on")
    parser.add_argument("--agent", type=str, default="hf", choices=["hf", "openai"],
                        help="Agent type (hf or openai)")
    parser.add_argument("--model_source", type=str, default="base", choices=["base", "trained"],
                        help="Source of the model: 'base' for pre-trained from models dir, 'trained' for RL checkpoints")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model name (e.g. Qwen/Qwen2.5-0.5B-Instruct) or checkpoint name (e.g. train_default)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--exp_name", type=str, default="eval_default",
                        help="Experiment name for logging")
    return parser.parse_args()

def main():
    args = parse_args()

    setup_logger(level="INFO", log_file=f"{args.exp_name}.log")
    logger.info(f"Starting Evaluation: Env={args.env}, Agent={args.agent}, Source={args.model_source}, Model={args.model_name}")

    # 1. 确定模型路径
    if args.model_source == "trained":
        model_path = os.path.join(CKPT_DIR, args.model_name)
        if not os.path.exists(model_path):
            logger.warning(f"Trained checkpoint directory {model_path} not found. Attempting to load anyway.")
    else:
        model_path = args.model_name

    # 2. 基础配置 (评估阶段不需要RLAlgo和Ragen(训练)配置)
    env_cfg = EnvConfig(env_name=args.env)
    
    # 推理时可以关掉温度等随机性，采用贪婪策略
    agent_cfg = AgentConfig(
        agent_type=args.agent, 
        model_name_or_path=model_path,
        temperature=0.0 # 评估时使用贪婪解码
    )
    
    # 3. 实例化环境
    env = make_env(env_cfg)

    # 4. 实例化代理
    if args.agent == "hf":
        agent = HFAgent(agent_cfg)
    else:
        agent = OpenAIAgent(agent_cfg)

    # 4. 开始评估循环
    success_count = 0
    total_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep) # 固定种子方便复现
        messages = [
            {"role": "system", "content": agent_cfg.system_prompt},
            {"role": "user", "content": f"{obs}\n{env.get_valid_actions()}\nPlease reason step by step."}
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
                messages.append({"role": "user", "content": f"Observation: {obs}\nReward for last step: {reward}\nNext action?"})
                
        total_rewards.append(ep_reward)
        if ep_reward > 0: # 假设正奖励代表成功，可根据具体环境调整
            success_count += 1
            
        logger.info(f"Episode {ep+1}/{args.episodes} finished. Reward: {ep_reward}")

    # 5. 汇总指标
    success_rate = success_count / args.episodes
    avg_reward = sum(total_rewards) / args.episodes
    logger.info(f"--- Evaluation Results ---")
    logger.info(f"Success Rate: {success_rate * 100:.2f}%")
    logger.info(f"Average Reward: {avg_reward:.4f}")

if __name__ == "__main__":
    main()
