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

    def batched_chat_request(self, messages_list: List[List[Dict[str, str]]]) -> List[str]:
        """
        Batched 版本的 ``chat_request``：对 N 条独立 messages 同时生成 N 条 response。

        语义保证（**与等价的 N 次串行 chat_request 在统计意义上同分布**）：
            - 每条 sequence 在自己的 logits 分布上独立采样（不共享 token）
            - 每条 sequence 在自己的 EOS / max_new_tokens 处独立截断
            - 不同 sequence 间不会通过 attention 互相干扰

        默认实现退化为 N 次串行调用 ``chat_request``，保证任何 BaseAgent 子类
        都可用。**性能敏感的子类（如 HFAgent）会重写为真正的 batch generate**，
        利用 GPU 并行获得 3~10x throughput。

        Args:
            messages_list: 长度 N 的列表，每个元素与 ``chat_request`` 入参同结构。

        Returns:
            List[str]: 长度 N 的 response，顺序与 ``messages_list`` 一一对应。
        """
        return [self.chat_request(msgs) for msgs in messages_list]
    
    @abstractmethod
    def get_log_probs(self, messages: List[Dict[str, str]], response: str) -> Any:
        """
        (可选) 获取生成回复的对数概率，用于强化学习(如PPO, GRPO)更新。
        仅对本地白盒模型有效。对于仅调用的API，可能无法返回。
        """
        pass
