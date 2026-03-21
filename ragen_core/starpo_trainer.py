from typing import Any
import re
from .trajectory_buffer import TrajectoryBuffer
from utils.logger import logger

class StarPOTrainer:
    """
    StarPO (State-Thinking-Actions-Reward Policy Optimization) 框架
    对多轮、随机环境进行轨迹级强化学习优化，旨在缓解模型崩溃与推理退化。
    """
    def __init__(self, config: Any, env: Any, agent: Any, rl_algo: Any):
        self.config = config
        self.env = env
        self.agent = agent
        self.rl_algo = rl_algo
        
        self.buffer = TrajectoryBuffer()
        
        self.num_rollouts = getattr(config.ragen_config, 'num_rollouts', 16)
        self.use_format_reward = getattr(config.ragen_config, 'use_format_reward', True)
        self.format_penalty = getattr(config.ragen_config, 'format_penalty', -0.1)
        self.variance_filter_ratio = getattr(config.ragen_config, 'variance_filter_ratio', 0.25)
        
        logger.info("Initialized StarPO Trainer (RAGEN Framework)")

    @staticmethod
    def _check_format(response: str) -> bool:
        """检查模型输出是否符合 <think>...</think><answer>...</answer> 格式"""
        pattern = r'<think>.*?</think>\s*<answer>.*?</answer>'
        return bool(re.search(pattern, response, re.DOTALL))

    def collect_rollouts(self, prompt_states: list):
        """
        采样收集多轮交互的轨迹。
        同一个初始状态 (prompt) 采样多个回答以用于组相对优化 (GRPO) 或 PPO 方差计算。
        """
        for state_info in prompt_states:
            # 解析状态以获取随机种子，确保环境对于同一组采样生成相同的初始状态（题目）
            seed = state_info.get("seed") if isinstance(state_info, dict) else None
            
            for _ in range(self.num_rollouts):
                trajectory = []
                obs, info = self.env.reset(seed=seed)
                
                # 初始化对话历史，第一轮把环境的描述和有效动作放进去
                messages = [
                    {"role": "system", "content": getattr(self.agent.config, "system_prompt", "You are a reasoning agent.")},
                    {"role": "user", "content": f"{obs}\n{self.env.get_valid_actions()}\nPlease reason step by step."}
                ]
                
                terminated, truncated = False, False
                while not (terminated or truncated):
                    # 获取Agent回复
                    response = self.agent.chat_request(messages)
                    
                    # RAGEN特征：检查输出格式，如果没有 <think> 标签则给予惩罚
                    step_penalty = 0.0
                    if self.use_format_reward and not self._check_format(response):
                        step_penalty = self.format_penalty
                    
                    # 与环境交互。这里我们简单把 agent 的全量回复丢给环境，由环境内部去做解析
                    next_obs, reward, terminated, truncated, info = self.env.step(response)
                    
                    # 组合真实奖励和格式惩罚
                    total_reward = reward + step_penalty
                    
                    # 记录轨迹
                    trajectory.append({
                        "obs": obs,
                        "messages": list(messages),
                        "response": response,
                        "reward": total_reward,
                        "terminated": terminated
                    })
                    
                    # 更新状态并准备下一轮消息
                    obs = next_obs
                    messages.append({"role": "assistant", "content": response})
                    if not terminated:
                        messages.append({"role": "user", "content": f"Observation: {obs}\nReward for last step: {reward}\nNext action?"})
                        
                self.buffer.add_trajectory(trajectory)

    def train_iteration(self, prompt_states: list):
        """
        一次完整的 RAGEN 训练迭代：
        1. 收集交互轨迹 (Rollout Stage)
        2. 基于轨迹级方差过滤，避免 Echo Trap
        3. 组装数据，交由底层 RL 算法更新
        """
        self.buffer.clear()
        
        # 1. 轨迹采样 (Online Data Collection)
        logger.info(f"Collecting rollouts for {len(prompt_states)} states...")
        self.collect_rollouts(prompt_states)
        
        # 2. 轨迹过滤 (StarPO-S 的稳定化改进)
        original_size = len(self.buffer.trajectories)
        self.buffer.filter_by_variance(group_size=self.num_rollouts, retain_ratio=self.variance_filter_ratio)
        filtered_size = len(self.buffer.trajectories)
        logger.info(f"Variance filtering: retained {filtered_size} / {original_size} trajectories.")
        
        # 3. 把轨迹数据组装，传给 RL 算法进行参数更新
        # 这里需要将轨迹序列数据进行一定的 padding/masking 以适应 PPO/GRPO，
        # 在这个架构演示中，我们直接把 buffer 数据丢给 train_step
        batch_data = self.buffer.get_all_data()
        if len(batch_data) > 0:
            metrics = self.rl_algo.train_step(batch_data)
            logger.info(f"Training metrics: {metrics}")
        else:
            logger.warning("No trajectories left after filtering, skipping update.")

    def run(self):
        """主执行循环"""
        import random
        num_iterations = self.config.total_training_steps // self.config.eval_interval
        logger.info(f"Starting RAGEN training for {num_iterations} iterations.")
        
        for i in range(num_iterations):
            # 从某处获取一批初始状态，使用 seed 保证每道题可以通过 reset 复现
            dummy_states = []
            for _ in range(self.config.rl_algo_config.batch_size):
                seed = random.randint(0, 2**31 - 1)
                dummy_states.append({"seed": seed})
                
            self.train_iteration(dummy_states)
            
            # TODO: Eval stage
