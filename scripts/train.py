import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import ExperimentConfig, EnvConfig, AgentConfig, RLAlgoConfig, RagenConfig
from configs.constants import CKPT_DIR
from envs import make_env
from agents.hf_agent import HFAgent
from rl_algos import make_algo
from ragen_core.starpo_trainer import StarPOTrainer
from ragen_core.pure_rl_trainer import PureRLTrainer
from utils.logger import setup_logger, logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train RAGEN-Ward framework")
    parser.add_argument("--env", type=str, default="math", choices=["math", "cartpole", "frozenlake"],
                        help="Environment to train on")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model path or name")
    parser.add_argument("--trainer", type=str, default="starpo", choices=["starpo", "pure"],
                        help="Trainer to use: starpo (RAGEN framework) or pure (Baseline RL)")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "grpo"],
                        help="RL Algorithm to use")
    parser.add_argument("--exp_name", type=str, default="train_default",
                        help="Experiment name for logging and saving")
    return parser.parse_args()

def main():
    args = parse_args()

    # 初始化日志
    setup_logger(level="INFO", log_file=f"{args.exp_name}.log")
    logger.info(f"Starting Training: Env={args.env}, Model={args.model}")

    # 1. 配置 / Configuration
    env_cfg = EnvConfig(env_name=args.env)
    
    # 训练阶段强制使用本地HF模型（需要梯度）
    agent_cfg = AgentConfig(agent_type="hf", model_name_or_path=args.model)
    algo_cfg = RLAlgoConfig(algo_name=args.algo, learning_rate=1e-5)
    
    # 启用Variance-based filtering等训练特性
    ragen_cfg = RagenConfig(
        num_rollouts=16, 
        use_format_reward=True, 
        variance_filter_ratio=0.25 # 保留前25%高方差的轨迹
    )
    
    config = ExperimentConfig(
        exp_name=args.exp_name,
        env_config=env_cfg,
        agent_config=agent_cfg,
        rl_algo_config=algo_cfg,
        ragen_config=ragen_cfg
    )

    # 2. 实例化环境
    env = make_env(config.env_config)

    # 3. 实例化代理 (训练时只用本地HFAgent)
    agent = HFAgent(config.agent_config)

    # 4. 实例化RL算法
    algo = make_algo(config.rl_algo_config, agent)

    # 5. 实例化训练器
    if args.trainer == "starpo":
        trainer = StarPOTrainer(config, env, agent, algo)
    else:
        trainer = PureRLTrainer(config, env, agent, algo)

    # 6. 开始训练
    logger.info("Initializing run...")
    try:
        trainer.run()
        # 训练结束后保存模型
        save_path = os.path.join(CKPT_DIR, args.exp_name)
        os.makedirs(save_path, exist_ok=True)
        algo.save(save_path)
        logger.info(f"Training completed successfully. Model saved to {save_path}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")

if __name__ == "__main__":
    main()
