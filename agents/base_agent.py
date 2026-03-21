from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseAgent(ABC):
    """
    智能体基类 / Base Agent Class
    定义了智能体与外界（环境或框架）交互的核心接口。
    所有模型（无论是本地加载还是API调用）都必须实现该接口。
    """
    
    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def chat_request(self, messages: List[Dict[str, str]]) -> str:
        """
        核心的文本交互接口。
        
        Args:
            messages (List[Dict[str, str]]): OpenAI API 格式的消息列表
                例如：[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": "What is 1 + 1?"}]
                      
        Returns:
            str: 智能体生成的回复文本。
        """
        pass
    
    @abstractmethod
    def get_log_probs(self, messages: List[Dict[str, str]], response: str) -> Any:
        """
        (可选) 获取生成回复的对数概率，用于强化学习(如PPO, GRPO)更新。
        仅对本地白盒模型有效。对于仅调用的API，可能无法返回。
        """
        pass
