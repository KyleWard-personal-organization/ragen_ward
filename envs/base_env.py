from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional

class BaseEnv(ABC):
    """
    环境基类 / Base Environment Class
    所有自定义环境或gym包装环境都必须继承该类并实现对应接口。
    确保环境与LLM代理(基于文本)的交互解耦。
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.current_step = 0
        # 必填字段；缺失直接 AttributeError，不再静默兜底。
        self.max_steps = config.max_steps

    @abstractmethod
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict]:
        """
        重置环境状态
        
        Args:
            seed (int, optional): 随机种子
            
        Returns:
            Tuple[str, dict]: 返回(文本化的初始观察状态, 额外信息字典)
        """
        self.current_step = 0
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        """
        执行一步动作
        
        Args:
            action (str): 文本形式的动作
            
        Returns:
            Tuple[str, float, bool, bool, dict]: 
            返回 (文本化的新观察状态, 奖励值, 是否终止(terminated), 是否截断(truncated), 额外信息字典)
        """
        self.current_step += 1
        pass

    @abstractmethod
    def render(self) -> Any:
        """
        渲染环境（可选）
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> str:
        """
        获取当前有效的动作列表或动作空间描述，以文本形式返回给LLM参考。
        """
        pass
