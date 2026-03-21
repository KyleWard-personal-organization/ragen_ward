import time
from typing import Any
from .trajectory_buffer import TrajectoryBuffer
from utils.logger import logger

class PureRLTrainer:
    """
    纯强化学习训练器 (Pure RL Trainer)
    作为 Baseline，它不包含 RAGEN 的核心特性（如 Variance-based filtering, Format penalty）。
    它仅仅是一个标准的多轮交互收集与 RL 算法更新的循环，用于对比评估 StarPO 框架带来的收益。
    """
    def __init__(self, config: Any, env: Any, agent: Any, rl_algo: Any):
        self.config = config
        self.env = env
        self.agent = agent
        self.rl_algo = rl_algo
        
        self.buffer = TrajectoryBuffer()
        
        # 即使在纯 RL 中，为了公平对比也采样多次 rollout，但不过滤
        self.num_rollouts = getattr(config.ragen_config, 'num_rollouts', 16)
        
        logger.info("Initialized Pure RL Trainer (Baseline without RAGEN features)")

    def collect_rollouts(self, prompt_states: list):
        """标准的数据收集过程，不进行任何格式惩罚。"""
        for state_info in prompt_states:
            # 提取随机种子以保证同一组 Rollout 生成完全相同的题目
            seed = state_info.get("seed") if isinstance(state_info, dict) else None
            
            for _ in range(self.num_rollouts):
                trajectory = []
                obs, info = self.env.reset(seed=seed)
                
                messages = [
                    {"role": "system", "content": getattr(self.agent.config, "system_prompt", "You are a reasoning agent.")},
                    {"role": "user", "content": f"{obs}\n{self.env.get_valid_actions()}\nPlease reason step by step."}
                ]
                
                terminated, truncated = False, False
                while not (terminated or truncated):
                    response = self.agent.chat_request(messages)
                    
                    # 纯 RL：没有对 <think> 标签的检查和 Format Penalty
                    next_obs, reward, terminated, truncated, info = self.env.step(response)
                    
                    trajectory.append({
                        "obs": obs,
                        "messages": list(messages),
                        "response": response,
                        "reward": reward, # 只有环境原始 reward
                        "terminated": terminated
                    })
                    
                    obs = next_obs
                    messages.append({"role": "assistant", "content": response})
                    if not terminated:
                        messages.append({"role": "user", "content": f"Observation: {obs}\nReward for last step: {reward}\nNext action?"})
                        
                self.buffer.add_trajectory(trajectory)

    def train_iteration(self, prompt_states: list):
        """纯 RL 训练迭代"""
        self.buffer.clear()
        
        # 1. 轨迹采样
        logger.info(f"[Pure RL] Collecting rollouts for {len(prompt_states)} states...")
        self.collect_rollouts(prompt_states)
        
        # 2. 不进行 Variance-based filtering，全部保留！
        logger.info(f"[Pure RL] Skipping variance filtering. Passing all {len(self.buffer.trajectories)} trajectories to RL algo.")
        
        # 3. 交由 RL 算法更新
        batch_data = self.buffer.get_all_data()
        if len(batch_data) > 0:
            metrics = self.rl_algo.train_step(batch_data)
            logger.info(f"[Pure RL] Training metrics: {metrics}")
        else:
            logger.warning("No trajectories collected.")

    def run(self):
        """主执行循环"""
        import random
        num_iterations = self.config.total_training_steps // self.config.eval_interval
        logger.info(f"Starting Pure RL training for {num_iterations} iterations.")
        
        for i in range(num_iterations):
            # 为每个 batch 生成带有随机种子的状态字典
            dummy_states = []
            for _ in range(self.config.rl_algo_config.batch_size):
                seed = random.randint(0, 2**31 - 1)
                dummy_states.append({"seed": seed})
                
            self.train_iteration(dummy_states)
