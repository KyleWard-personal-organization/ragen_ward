from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseRLAlgo(ABC):
    """
    强化学习算法基类 / Base RL Algorithm Class
    提供统一的训练、动作选择和保存加载接口。
    """
    
    def __init__(self, config: Any, agent: Any):
        """
        初始化算法
        
        Args:
            config (Any): 算法配置 (对应 RLAlgoConfig)
            agent (Any): 交互的Agent (如 HFAgent)
        """
        self.config = config
        self.agent = agent

    @abstractmethod
    def get_action(self, state: Any, evaluate: bool = False) -> Any:
        """
        根据当前状态选择动作。
        
        Args:
            state (Any): 当前状态
            evaluate (bool): 是否处于评估模式(评估模式下通常不加入探索噪声)
            
        Returns:
            Any: 选择的动作
        """
        pass

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        执行一次模型更新（反向传播）。
        
        Args:
            batch (Any): 从回放池或rollout buffer中采样的数据批次
            
        Returns:
            Dict[str, float]: 训练指标字典，例如 {'loss': 0.1, 'actor_loss': 0.05}
        """
        pass
        
    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """加载模型"""
        pass
