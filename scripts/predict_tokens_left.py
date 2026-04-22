from transformers import AutoTokenizer
from envs import make_env
from configs.config import EnvConfig

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

MAX_SEQ = 2048
MAX_NEW = 256
MAX_TURN = 5

for env_name in ["frozenlake", "sokoban", "cartpole", "bandit", "math"]:
    env = make_env(EnvConfig(env_name=env_name, max_steps=10))
    obs, _ = env.reset(seed=42)

    sys_msg = {"role": "system", "content": env.agent_system_prompt}
    first_user = (
        env.get_env_instruction() + "\n\n"
        + f"{obs}\n{env.get_valid_actions()}\nPlease reason step by step."
    )
    user_msg = {"role": "user", "content": first_user}
    prompt_ids = tok.apply_chat_template([sys_msg, user_msg], tokenize=True)

    mid_obs = f"Observation: {obs}\nReward for last step: 0.0\nNext action?"
    mid_len = len(tok(mid_obs).input_ids)

    worst = len(prompt_ids) + MAX_TURN * MAX_NEW + (MAX_TURN - 1) * mid_len
    margin = MAX_SEQ - worst
    flag = "OK" if margin > 0 else "OVERFLOW"
    print(f"[{env_name:10s}] prompt={len(prompt_ids):>4}  mid_obs={mid_len:>3}  "
          f"worst={worst:>4}  margin={margin:>5}  {flag}")