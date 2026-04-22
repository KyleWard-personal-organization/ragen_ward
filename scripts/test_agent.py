import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.hf_agent import HFAgent
from configs.config import AgentConfig

def main():
    print("=== Testing HF Agent (with a small model for speed) ===")
    # 这里用一个极小的模型跑测试，防止下载 Qwen 很久
    # AgentConfig 所有必填字段都要显式传入（默认值不再从 dataclass 取）
    config = AgentConfig(
        agent_type="hf",
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_new_tokens=64,
    )
    
    try:
        agent = HFAgent(config)
        
        messages = [
            {"role": "user", "content": "Say hello world!"}
        ]
        
        print("\nSending prompt to agent...")
        response = agent.chat_request(messages)
        
        print("\nAgent Response:")
        print(response)
        print("\nAgent test passed!")
    except Exception as e:
        print(f"Agent test failed: {e}")

if __name__ == "__main__":
    main()
