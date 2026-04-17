import gymnasium as gym
from typing import Optional, Tuple, Any
from .base_env import BaseEnv


class GymEnvWrapper(BaseEnv):
    """
    Gymnasium 环境的通用包装类
    将离散/连续空间的Gym环境转化为基于文本的LLM交互接口。
    """
    def __init__(self, config: Any):
        super().__init__(config)
        # env_name 为 EnvConfig 必填字段；kwargs 是预留扩展字段（dataclass 中默认为空 dict）。
        self.env_name = config.env_name
        env_kwargs = config.kwargs
        self.env = gym.make(self.env_name, **env_kwargs)
        
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict]:
        super().reset(seed=seed, **kwargs)
        obs, info = self.env.reset(seed=seed, **kwargs)
        return self._format_obs(obs), info

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        super().step(action)
        
        # 将文本动作转化为Gym所需动作
        gym_action = self._parse_action(action)
        
        # 如果解析失败，可以直接返回负奖励或当前状态
        if gym_action is None:
            return "Invalid action. Please output the action inside an <answer> tag.", -0.1, False, False, {"error": "Invalid action format.", "action_is_effective": False}
            
        obs, reward, terminated, truncated, info = self.env.step(gym_action)
        
        if self.current_step >= self.max_steps:
            truncated = True
            
        # 增加一些 RAGEN 兼容的 info 字段
        info["action_is_valid"] = True
        if "action_is_effective" not in info:
            info["action_is_effective"] = True
            
        return self._format_obs(obs), float(reward), terminated, truncated, info

    def render(self) -> Any:
        return self.env.render()
        
    def close(self):
        self.env.close()

    def _format_obs(self, obs: Any) -> str:
        """将数值型Observation转化为描述性文本"""
        return f"Observation: {obs}"
        
    def _parse_action(self, action: str) -> Any:
        """
        将文本动作转化为数值动作（子类可复写）
        默认提供提取 <answer>...</answer> 的能力
        """
        import re
        match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
        content = match.group(1).strip() if match else action.strip()
        try:
            return int(content)
        except ValueError:
            return None

    def get_valid_actions(self) -> str:
        """
        获取当前有效的动作列表或动作空间描述，以文本形式返回给LLM参考。
        """
        pass


class CartPoleEnv(GymEnvWrapper):
    """CartPole特定环境封装"""
    def __init__(self, config: Any):
        config.env_name = 'CartPole-v1'
        super().__init__(config)

    def _format_obs(self, obs: Any) -> str:
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        return (f"Cart Position: {cart_pos:.4f}, Cart Velocity: {cart_vel:.4f}, "
                f"Pole Angle: {pole_angle:.4f}, Pole Velocity at Tip: {pole_vel:.4f}")

    def get_valid_actions(self) -> str:
        return "Valid actions are: 0 (Push cart to the left), 1 (Push cart to the right)."


class SokobanEnv(GymEnvWrapper):
    """
    Sokoban (推箱子) 环境封装
    作为 RAGEN-main 中最核心的多步空间推理与规划测试床
    """
    def __init__(self, config: Any):
        # 如果尚未安装 gym-sokoban，需要提醒用户安装
        try:
            import sys
            import os
            # 屏蔽 gym 导入时硬编码打印到 stderr 的烦人版本警告
            original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            try:
                import gym_sokoban
            finally:
                sys.stderr.close()
                sys.stderr = original_stderr
        except ImportError:
            import logging
            logging.warning("gym_sokoban is not installed. Please run `pip install gym-sokoban`")
            
        from .base_env import BaseEnv
        BaseEnv.__init__(self, config)
        
        # ----------------------------------------------------
        # 直接配置推箱子的参数 (对齐 RAGEN 论文实验配置)
        # 默认使用 SimpleSokoban (6x6 网格, 1 个箱子)
        # 如果要测 LargerSokoban, 可以改为: dim_room=(8, 8), num_boxes=2
        # ----------------------------------------------------
        self.dim_room = (6, 6)     # (dim_x, dim_y)
        self.num_boxes = 1         # 箱子数量
        # 尊重 EnvConfig.max_steps；BaseEnv.__init__ 已经设好，这里保持不覆盖
        # （如果用户传入 max_steps，就使用用户值；否则沿用 BaseEnv 默认 100）
        
        from gym_sokoban.envs.sokoban_env import SokobanEnv as CoreSokobanEnv
        
        # 初始化核心环境
        self.env = CoreSokobanEnv(
            dim_room=self.dim_room,
            max_steps=self.max_steps,
            num_boxes=self.num_boxes
        )
        
        # 对应 RAGEN-main 中的网格字符映射
        self.GRID_LOOKUP = {
            0: "#", # 墙壁 (wall)
            1: "_", # 空地 (empty)
            2: "O", # 目标点 (target)
            3: "√", # 箱子在目标点上 (box on target)
            4: "X", # 箱子 (box)
            5: "P", # 玩家 (player)
            6: "S"  # 玩家在目标点上 (player on target)
        }

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict]:
        from .base_env import BaseEnv
        BaseEnv.reset(self, seed=seed, **kwargs)
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        return self._format_obs(obs), {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        from .base_env import BaseEnv
        BaseEnv.step(self, action)
        
        gym_action = self._parse_action(action)
        if gym_action is None:
            return "Invalid action. Please output the action inside an <answer> tag.", -0.1, False, False, {"error": "Invalid action format.", "action_is_effective": False}
            
        obs, reward, done, info = self.env.step(gym_action)
        
        terminated = done
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            
        info["action_is_valid"] = True
        if "action_is_effective" not in info:
            info["action_is_effective"] = True
            
        return self._format_obs(obs), float(reward), terminated, truncated, info

    def _format_obs(self, obs: Any) -> str:
        import numpy as np
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed
        
        # 如果玩家(5)站在目标点(2)上，标记为6
        room = np.where((room_state == 5) & (room_fixed == 2), 6, room_state)
        
        grid_str = ""
        for row in room.tolist():
            grid_str += "".join([self.GRID_LOOKUP.get(cell, "?") for cell in row]) + "\n"
            
        return grid_str.strip()

    def get_valid_actions(self) -> str:
        return (
            "Valid actions are: 1 (Push Up), 2 (Push Down), 3 (Push Left), 4 (Push Right). "
            "Please output the action number inside an <answer> tag."
        )

    def _parse_action(self, action: str) -> Optional[int]:
        """
        解析 Sokoban 动作 (在 gym_sokoban 中 1: Push Up, 2: Push Down, 3: Push Left, 4: Push Right)。
        同样先按数字解析，再按方向词 + word boundary 匹配。
        """
        # 调用父类提取 <answer> 内容
        content = super()._parse_action(action)
        if isinstance(content, int) and 1 <= content <= 4:
            return content

        import re
        match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
        act_str = match.group(1).strip().lower() if match else action.strip().lower()

        if act_str.isdigit():
            num = int(act_str)
            if 1 <= num <= 4:
                return num
            return None

        word_map = {"up": 1, "down": 2, "left": 3, "right": 4}
        for k, v in word_map.items():
            if re.search(rf'\b{k}\b', act_str):
                return v
        return None


class FrozenLakeEnv(GymEnvWrapper):
    """FrozenLake特定环境封装，对齐 RAGEN 网格化文本渲染"""
    def __init__(self, config: Any):
        # is_slippery / map_name 属于 FrozenLake **环境内部超参**，不是 EnvConfig 的必填字段。
        # 在 EnvConfig 未显式挂载时使用 RAGEN 的默认取值（is_slippery=True, map_name="4x4"）。
        is_slippery = getattr(config, 'is_slippery', True)
        config.env_name = 'FrozenLake-v1'
        config.kwargs = {"is_slippery": is_slippery, "map_name": "4x4"}
        super().__init__(config)
        
        self.MAP_LOOKUP = {b"S": 1, b"F": 1, b"H": 2, b"G": 3}
        self.GRID_LOOKUP = {0:"P", 1:"_", 2:"O", 3:"G", 4:"X", 5:"√"}
        
    def _format_obs(self, obs: Any) -> str:
        # 获取底层环境的网格信息
        import numpy as np
        desc = self.env.unwrapped.desc.copy()
        
        # 0. 提取当前玩家坐标 (row, col)
        ncol = self.env.unwrapped.ncol
        player_row, player_col = obs // ncol, obs % ncol
        
        # 1. 数字化网格
        room = np.vectorize(lambda x: self.MAP_LOOKUP.get(x, 1))(desc)
        
        # 2. 标记玩家位置
        if desc[player_row, player_col] == b'H':
            room[player_row, player_col] = 4 # X 掉进洞
        elif desc[player_row, player_col] == b'G':
            room[player_row, player_col] = 5 # √ 到达目标
        else:
            room[player_row, player_col] = 0 # P 玩家
            
        # 3. 转化为字符画
        grid_str = '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room)
        return grid_str
        
    def get_valid_actions(self) -> str:
        return "Valid actions are: 1 (Left), 2 (Down), 3 (Right), 4 (Up). Output action in <answer> tag."
        
    def _parse_action(self, action: str) -> Optional[int]:
        """
        解析 FrozenLake 动作：
        1. 先尝试提取 <answer> 标签中的内容
        2. 如果内容能转为合法数字 (1-4)，按数字分支返回
        3. 否则把内容转小写，严格匹配 left/down/right/up（使用 word boundary 避免 "step 12" 误匹配 "1" 的歧义）
        """
        # 调用父类先抽离出 <answer> 中的内容
        content = super()._parse_action(action)
        if isinstance(content, int) and 1 <= content <= 4:
            return content - 1  # Gym 内部是 0,1,2,3

        import re
        match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
        act_str = match.group(1).strip().lower() if match else action.strip().lower()

        # 优先按纯数字匹配：若 act_str 本身就是 "1"/"2"/"3"/"4"，直接返回
        if act_str.isdigit():
            num = int(act_str)
            if 1 <= num <= 4:
                return num - 1
            return None

        # 再按方向词匹配，使用 \b 防止 "left" 误匹配 "leftover"、或 "2" 误匹配 "12"
        word_map = {"left": 0, "down": 1, "right": 2, "up": 3}
        for k, v in word_map.items():
            if re.search(rf'\b{k}\b', act_str):
                return v
        return None
