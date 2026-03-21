import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.hf_agent import HFAgent
from configs.config import AgentConfig

def main():
    print("=== Testing HF Agent (with a small model for speed) ===")
    # 这里用一个极小的模型跑测试，防止下载 Qwen 很久
    config = AgentConfig(
        agent_type="hf",
        model_name_or_path="gpt2", # 仅用于快速连通性测试
        max_new_tokens=20
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
