import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import make_env
from configs.config import EnvConfig

def test_math_env():
    print("=== Testing MathEnv (Countdown Game) ===")
    config = EnvConfig(env_name="math", max_steps=5)
    env = make_env(config)
    
    # 第一次测试：测试非法输入
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation: {obs}")
    print(f"Valid actions: {env.get_valid_actions()}")
    
    print("\n--- Test 1: Invalid Format ---")
    obs, reward, terminated, truncated, info = env.step("The answer is unknown.")
    print(f"obs: {obs}")
    print(f"reward: {reward}, terminated: {terminated}")
    
    # 第二次测试：算对的情况
    obs, info = env.reset(seed=44)
    print(f"\n--- Test 2: Correct Expression ---")
    
    env.numbers = [1, 2, 3, 4]
    env.target = 24
    print("[Mocking problem for Test 2: target=24, nums=[1,2,3,4]]")
    
    obs, reward, terminated, truncated, info = env.step("<think>Let's do 1+2+3 then *4</think><answer>(1+2+3)*4</answer>")
    print(f"obs: {obs}")
    print(f"reward: {reward}, terminated: {terminated}")

def test_frozenlake_env():
    print("=== Testing FrozenLakeEnv ===")
    config = EnvConfig(env_name="frozenlake", max_steps=20)
    env = make_env(config)
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial Grid:\n{obs}")
    print(f"Valid actions: {env.get_valid_actions()}")
    
    print("\n--- Test: Move Down ---")
    obs, reward, terminated, truncated, info = env.step("<think>Need to go down</think><answer>Down</answer>")
    print(f"obs:\n{obs}")
    print(f"reward: {reward}, terminated: {terminated}, info: {info}")

def test_bandit_env():
    print("=== Testing BanditEnv ===")
    config = EnvConfig(env_name="bandit", max_steps=1)
    env = make_env(config)
    
    obs, info = env.reset(seed=123)
    print(f"\nInitial observation:\n{obs}")
    print(f"Valid actions: {env.get_valid_actions()}")
    
    print("\n--- Test: Pulling an arm ---")
    # 尝试拉一个存在于提示中的名字
    obs, reward, terminated, truncated, info = env.step(f"<answer> {env.name_a} </answer>")
    print(f"obs: {obs}")
    print(f"reward: {reward}, terminated: {terminated}, info: {info}")

def test_sokoban_env():
    print("=== Testing SokobanEnv ===")
    try:
        import sys
        import os
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            import gym_sokoban
        finally:
            sys.stderr.close()
            sys.stderr = original_stderr
    except ImportError:
        print("gym-sokoban is not installed. Skipping. Run `pip install gym-sokoban`")
        return
        
    config = EnvConfig(env_name="sokoban", max_steps=100)
    env = make_env(config)
    
    obs, info = env.reset(seed=10)
    print(f"\nInitial Grid:\n{obs}")
    print(f"Valid actions: {env.get_valid_actions()}")
    
    print("\n--- Test: Push Left ---")
    obs, reward, terminated, truncated, info = env.step("<answer> Left </answer>")
    print(f"obs:\n{obs}")
    print(f"reward: {reward}, terminated: {terminated}, info: {info}")

def main():
    parser = argparse.ArgumentParser(description="Test environment wrappers.")
    parser.add_argument("--env", type=str, choices=["math", "frozenlake", "bandit", "sokoban", "all"], 
                        default="all", help="Which environment to test")
    args = parser.parse_args()
    
    if args.env == "math" or args.env == "all":
        test_math_env()
        print("\n" + "="*40 + "\n")
    if args.env == "frozenlake" or args.env == "all":
        test_frozenlake_env()
        print("\n" + "="*40 + "\n")
    if args.env == "bandit" or args.env == "all":
        test_bandit_env()
        print("\n" + "="*40 + "\n")
    if args.env == "sokoban" or args.env == "all":
        test_sokoban_env()
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
