from typing import List, Dict, Any
from .base_agent import BaseAgent
from utils.logger import logger

try:
    import openai
except ImportError:
    openai = None

class OpenAIAgent(BaseAgent):
    """
    基于 OpenAI API 接口的远程大语言模型代理。
    仅用于纯测试、评估或不需要梯度的强化学习(如部分零阶优化方法)，不适用于依赖梯度的PPO/GRPO。
    """
    def __init__(self, config: Any):
        super().__init__(config)
        if openai is None:
            logger.error("OpenAI library is not installed. Please install it via `pip install openai`.")
            raise ImportError("OpenAI library is not installed.")
            
        self.api_key = getattr(config, 'api_key', None)
        self.base_url = getattr(config, 'base_url', None)
        self.model_name = getattr(config, 'model_name_or_path', 'gpt-3.5-turbo')
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        logger.info(f"Initialized OpenAI Agent with model {self.model_name}")

    def chat_request(self, messages: List[Dict[str, str]]) -> str:
        """
        调用远程API生成文本回复
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=getattr(self.config, 'temperature', 0.7),
                max_tokens=getattr(self.config, 'max_new_tokens', 512),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
            return ""
            
    def get_log_probs(self, messages: List[Dict[str, str]], response: str) -> Any:
        """
        远程API通常不提供完整的梯度信息，对于基于策略梯度的RL算法无效。
        """
        raise NotImplementedError("OpenAI API agent does not support retrieving gradients/log_probs for RL.")
