"""
Gym 包装环境 / Gymnasium-based Environments
-----------------------------------------------
封装 FrozenLake / Sokoban / CartPole 等经典离散控制环境到"文本 in / 文本 out"的接口。

## RAGEN 对齐（路径 A）

为对齐 RAGEN 论文对多步环境的设计，FrozenLakeEnv / SokobanEnv 支持 **一次回复多个动作**：
    <answer>Left || Up || Up</answer>
-> env 在本 turn 内依次执行 3 个原子 step（上层 BaseEnv.step 负责循环与聚合）。

- 解析：子类覆盖 `_parse_action_sequence`，按 `||` 切分 `<answer>` 内容，对每个 token 跑
  `_parse_action`（单动作解析）。
- 执行：单个原子动作由 `_step_atomic(gym_action)` 负责，不再需要自己推进 `current_step`。

CartPole 保持"一次回复 = 一个动作"的旧行为（它是连续平衡控制，批量动作没意义）。
"""

import re
from typing import Any, List, Optional, Tuple

import gymnasium as gym

from .base_env import BaseEnv


# ---------------------------------------------------------------------------
# 通用基类：gymnasium.make 出来的离散空间环境
# ---------------------------------------------------------------------------

class GymEnvWrapper(BaseEnv):
    """
    Gymnasium 环境的通用包装类。

    子类需要提供（至少）：
        - `_parse_action(text) -> Optional[int]`：把单 token 解析成 gym 整数动作；
          如果无法解析返回 None（会被 `_step_atomic` 当成 invalid 处理）。
        - `_parse_action_sequence(text) -> List[Optional[int]]`：如需支持 `||`，覆盖它。
        - `get_valid_actions()` / `get_env_instruction()`。
    """

    def __init__(self, config: Any):
        super().__init__(config)
        # env_name 为 EnvConfig 必填字段；kwargs 是预留扩展字段（dataclass 中默认为空 dict）。
        self.env_name = config.env_name
        env_kwargs = config.kwargs
        # 显式把 BaseEnv.max_steps 传给 gymnasium 的 TimeLimit wrapper，消除"我们 BaseEnv
        # 的 max_steps 计数" 与 "gym spec 默认 max_episode_steps（FrozenLake=100, CartPole=500）"
        # 两套计数器共存的隐患 —— 严格遵循"单一真相"原则。
        self.env = gym.make(
            self.env_name,
            max_episode_steps=self.max_steps,
            **env_kwargs,
        )

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict]:
        super().reset(seed=seed, **kwargs)
        obs, info = self.env.reset(seed=seed, **kwargs)
        return self._format_obs(obs), info

    def _step_atomic(self, gym_action: Optional[int]) -> Tuple[str, float, bool, bool, dict]:
        if gym_action is None:
            return (
                "Invalid action. Please output the action inside an <answer> tag.",
                -0.1,
                False,
                False,
                {
                    "error": "Invalid action format.",
                    "action_is_valid": False,
                    "action_is_effective": False,
                },
            )

        obs, reward, terminated, truncated, info = self.env.step(gym_action)
        # current_step 检查由 BaseEnv.step 统一处理，这里不再提前 truncated=True
        info = dict(info) if isinstance(info, dict) else {}
        info["action_is_valid"] = True
        if "action_is_effective" not in info:
            info["action_is_effective"] = True
        return self._format_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        return self.env.render()

    def close(self):
        self.env.close()

    def _format_obs(self, obs: Any) -> str:
        """将数值型 observation 转化为描述性文本。"""
        return f"Observation: {obs}"

    def _parse_action(self, action_text: str) -> Any:
        """
        默认：提取 `<answer>...</answer>` 里的内容，尝试转 int，失败返回 None。
        子类通常会覆盖它实现方向词匹配等。
        """
        match = re.search(r'<answer>(.*?)</answer>', action_text, re.DOTALL)
        content = match.group(1).strip() if match else action_text.strip()
        try:
            return int(content)
        except (ValueError, TypeError):
            return None

    def get_valid_actions(self) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 共享工具：把 `<answer>A || B || C</answer>` 切成 ["A", "B", "C"]
# ---------------------------------------------------------------------------

def _split_action_tokens(action_text: str) -> List[str]:
    """
    从完整 LLM 回复中抽取 `<answer>` 内容（没有则退化成全文），按 `||` 切分。
    保证返回至少 1 个元素：
        - 若 `<answer>` 内容为空或无 `<answer>` 且全文为空字符串，则返回 `[""]`
          （后续 `_parse_action` 会把空字符串解析成 None，触发 invalid 分支）。
    """
    match = re.search(r'<answer>(.*?)</answer>', action_text, re.DOTALL)
    content = match.group(1).strip() if match else action_text.strip()

    if "||" in content:
        tokens = [t.strip() for t in content.split("||")]
        tokens = [t for t in tokens if t]  # 去空白 token
        if not tokens:
            return [""]
        return tokens
    return [content]


# ---------------------------------------------------------------------------
# CartPole：连续平衡控制，保持单动作语义
# ---------------------------------------------------------------------------

class CartPoleEnv(GymEnvWrapper):
    """CartPole 封装。连续平衡任务 → 仅单动作语义，不支持 `||` 序列。"""

    agent_system_prompt = (
        "You are a control agent playing CartPole. Your only goal is to keep the pole "
        "balanced upright by pushing the cart left or right. At each turn you observe "
        "four numbers: cart position, cart velocity, pole angle, pole angular velocity. "
        "The episode ends (failure) if the pole falls past ~±0.21 rad or the cart leaves "
        "the ~±2.4 track. A common heuristic: push in the direction the pole is leaning "
        "toward, weighted by the pole's angular velocity.\n"
        "Output format is strict and non-negotiable: first reason inside "
        "<think>...</think>, then output exactly ONE action (0 = left, 1 = right) inside "
        "<answer>...</answer>. Any deviation from this format is treated as an invalid turn.\n"
        "Example: <think>Pole angle is +0.05 and still increasing, push right to counter."
        "</think><answer>1</answer>"
    )

    def __init__(self, config: Any):
        config.env_name = 'CartPole-v1'
        super().__init__(config)

    def _step_atomic(self, gym_action: Optional[int]) -> Tuple[str, float, bool, bool, dict]:
        # CartPole 的"成功"语义：活到 max_steps 被 truncated（杆一直没倒）。
        # terminated=True 表示杆倒或出界 → 失败。
        # 不能沿用基类默认的 "terminated and not truncated = 成功"，那套逻辑对离散目标
        # 环境（FrozenLake/Sokoban）是对的，但对 CartPole 语义刚好相反。
        obs, reward, terminated, truncated, info = super()._step_atomic(gym_action)
        if gym_action is None:
            return obs, reward, terminated, truncated, info

        if bool(terminated):
            info["is_success"] = False
        elif bool(truncated):
            info["is_success"] = True
        else:
            info["is_success"] = False
        return obs, reward, terminated, truncated, info

    def _format_obs(self, obs: Any) -> str:
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        return (
            f"Cart Position: {cart_pos:.4f}, Cart Velocity: {cart_vel:.4f}, "
            f"Pole Angle: {pole_angle:.4f}, Pole Velocity at Tip: {pole_vel:.4f}"
        )

    def get_valid_actions(self) -> str:
        return "Valid actions are: 0 (Push cart to the left), 1 (Push cart to the right)."

    def get_env_instruction(self) -> str:
        return (
            "You are controlling a CartPole to keep the pole balanced. "
            "Each turn you output **one** action. "
            "Wrap the action number inside <answer>...</answer>. "
            "Example: <think>The pole is tilting right, push right.</think><answer>1</answer>"
        )


# ---------------------------------------------------------------------------
# Sokoban：RAGEN 主力多步规划环境
# ---------------------------------------------------------------------------

class SokobanEnv(GymEnvWrapper):
    """
    Sokoban (推箱子) 环境封装。RAGEN 中的多步空间推理测试床。

    对齐 RAGEN 论文的 `||` 动作序列：一次回复可以一次性规划多步推箱子。
    """

    agent_system_prompt = (
        "You are a spatial planning agent solving Sokoban. The grid uses: "
        "`#`=wall, `_`=empty floor, `O`=target, `X`=box, `√`=box already on a target, "
        "`P`=player, `S`=player standing on a target. Your goal is to push every X onto "
        "an O so it becomes √. You can only **push** boxes (by walking into them with "
        "free space on the other side); you cannot pull, and a box shoved into a wall "
        "corner is often unrecoverable, so plan several moves ahead.\n"
        "Output format is strict and non-negotiable: first reason inside "
        "<think>...</think>, then output the action sequence inside <answer>...</answer>. "
        "Use Up/Down/Left/Right (or 1/2/3/4) separated by `||` to execute multiple moves "
        "in a single turn — this is strongly encouraged for efficiency.\n"
        "Example: <think>Box at (2,3), target at (2,5). I need to approach from the left "
        "and push right twice.</think><answer>Right || Right || Right</answer>"
    )

    def __init__(self, config: Any):
        try:
            import sys
            import os
            # 屏蔽 gym 导入时硬编码打印到 stderr 的烦人版本警告
            original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            try:
                import gym_sokoban  # noqa: F401
            finally:
                sys.stderr.close()
                sys.stderr = original_stderr
        except ImportError:
            import logging
            logging.warning("gym_sokoban is not installed. Please run `pip install gym-sokoban`")

        # 跳过 GymEnvWrapper.__init__ 的 gym.make（Sokoban 要手动构造）
        BaseEnv.__init__(self, config)

        # ----------------------------------------------------
        # 对齐 RAGEN 默认 SimpleSokoban (6x6, 1 个箱子)
        # 如需 LargerSokoban 改成 dim_room=(8,8), num_boxes=2
        # ----------------------------------------------------
        self.dim_room = (6, 6)
        self.num_boxes = 1

        from gym_sokoban.envs.sokoban_env import SokobanEnv as CoreSokobanEnv
        self.env = CoreSokobanEnv(
            dim_room=self.dim_room,
            max_steps=self.max_steps,
            num_boxes=self.num_boxes,
        )

        self.GRID_LOOKUP = {
            0: "#",  # wall
            1: "_",  # empty
            2: "O",  # target
            3: "√",  # box on target
            4: "X",  # box
            5: "P",  # player
            6: "S",  # player on target
        }

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict]:
        BaseEnv.reset(self, seed=seed, **kwargs)
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        return self._format_obs(obs), {}

    def _step_atomic(self, gym_action: Optional[int]) -> Tuple[str, float, bool, bool, dict]:
        if gym_action is None:
            return (
                "Invalid action. Please output the action inside an <answer> tag.",
                -0.1,
                False,
                False,
                {
                    "error": "Invalid action format.",
                    "action_is_valid": False,
                    "action_is_effective": False,
                },
            )

        # gym_sokoban 返回 4 元组 (obs, reward, done, info)
        obs, reward, done, info = self.env.step(gym_action)
        info = dict(info) if isinstance(info, dict) else {}
        info["action_is_valid"] = True
        if "action_is_effective" not in info:
            info["action_is_effective"] = True

        # gym_sokoban 把"推完所有箱子"和"步数用完"都写成 done=True，
        # 但 RL 语义下这是两件事（terminated vs truncated）。
        # 从 info 里两个字段拆开（仅在 done=True 时 sokoban 才写这两个键）：
        #   info["all_boxes_on_target"] → True 表示任务成功完成
        #   info["maxsteps_used"]       → True 表示因 max_steps 超时
        terminated = False
        truncated = False
        if done:
            all_boxes = bool(info.get("all_boxes_on_target", False))
            if all_boxes:
                terminated = True
                info["is_success"] = True
            else:
                # done=True 但没全部在目标上 → 只能是超时
                truncated = True
                info["is_success"] = False
        return self._format_obs(obs), float(reward), terminated, truncated, info

    def _format_obs(self, obs: Any) -> str:
        import numpy as np
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed

        # 玩家(5) 站在目标(2) 上时标记为 6
        room = np.where((room_state == 5) & (room_fixed == 2), 6, room_state)
        grid_str = ""
        for row in room.tolist():
            grid_str += "".join([self.GRID_LOOKUP.get(cell, "?") for cell in row]) + "\n"
        return grid_str.strip()

    def get_valid_actions(self) -> str:
        return (
            "Valid actions: 1=Up, 2=Down, 3=Left, 4=Right. "
            "You may output a sequence separated by '||'. "
            "Example: <answer>Up || Right || Right</answer>"
        )

    def get_env_instruction(self) -> str:
        return (
            "You are solving Sokoban. Push every box (X) onto a target (O). "
            "You can push a box only by walking towards it; you cannot pull or walk "
            "through walls. When a box sits on a target it shows as √. The player is P "
            "(S when standing on a target).\n"
            "**Answer format**: put a sequence of moves inside <answer>...</answer>, "
            "separated by `||`. Words (Up/Down/Left/Right) or numbers (1-4) both work.\n"
            "Example: <think>I need to push the box right then go up.</think>"
            "<answer>Right || Right || Up</answer>"
        )

    def _parse_action_sequence(self, action_text: str) -> List[Optional[int]]:
        tokens = _split_action_tokens(action_text)
        return [self._parse_action(tok) for tok in tokens]

    def _parse_action(self, action_text: str) -> Optional[int]:
        """
        解析 Sokoban 单动作（1=Up, 2=Down, 3=Left, 4=Right）。
        既接受 "<answer>2</answer>" / "<answer>Up</answer>" 的完整 text，也接受 "Up" / "2" 的裸 token。
        """
        match = re.search(r'<answer>(.*?)</answer>', action_text, re.DOTALL)
        act_str = match.group(1).strip().lower() if match else action_text.strip().lower()

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


# ---------------------------------------------------------------------------
# FrozenLake：RAGEN 另一个核心多步规划环境
# ---------------------------------------------------------------------------

class FrozenLakeEnv(GymEnvWrapper):
    """
    FrozenLake 封装，对齐 RAGEN 网格化文本渲染 + `||` 多动作序列。

    默认使用 4x4 地图的确定性版本（降低小模型训练难度）。

    ## Reward shaping（本地开关，不上报到 EnvConfig / CLI）

    原生 FrozenLake 是"到 Goal = +1，其他一律 0"的极度稀疏奖励，对小模型 + 有限 rollout
    量的 cold-start 极不友好（4x4 地图下 32 rollout/step 里成功样本期望 < 0.01）。把类属性
    ``use_shaped_reward`` 切成 True 后，会在每个**非终止**原子 step 上按"曼哈顿距离变化 +
    是否撞墙"叠一个小幅 shaping reward（量级 ±0.01 ~ ±0.05），**绝不**会盖过终点 +1 的主
    信号；终止步（到 Goal / 掉洞）严格保持原生 reward 语义：

    - 到 Goal：reward=+1.0，不叠 shaping，保留完整主信号
    - 掉洞：  reward=0.0，不叠 shaping（不惩罚，避免模型学成"害怕探索"）
    - 其他：  reward=原生 0.0 + shaping（见下方常量）

    切换模式只需改 ``use_shaped_reward`` 一行，不用动 CLI / EnvConfig / 训练脚本。想跑 RAGEN
    论文对齐的 sparse baseline，把它改成 False 重训即可。
    """

    agent_system_prompt = (
        "You are a spatial reasoning agent playing FrozenLake on a 4x4 grid. The grid uses: "
        "`P`=you (the player), `_`=safe ice, `O`=hole (stepping in = episode ends with "
        "reward 0), `G`=goal (stepping in = episode ends with reward 1), `X`=you fell into "
        "a hole, `√`=you reached the goal. Coordinates grow **down** (row) and **right** "
        "(column); the goal is at the bottom-right. Each move shifts you one tile; walking "
        "into a wall leaves you in place (wasted move).\n"
        "Output format is strict and non-negotiable: first reason inside "
        "<think>...</think>, then output the action sequence inside <answer>...</answer>. "
        "Use Left/Down/Right/Up (or 1/2/3/4) separated by `||` to plan multiple moves per "
        "turn — strongly encouraged since each turn costs an LLM call.\n"
        "Example: <think>I'm at row 0 col 0. The goal is at row 3 col 3. Row 1 col 1 is a "
        "hole, so I'll go right twice first then down.</think>"
        "<answer>Right || Right || Down || Down || Down</answer>"
    )

    # gymnasium FrozenLake 的离散动作：0=Left, 1=Down, 2=Right, 3=Up
    _WORD_TO_GYM = {"left": 0, "down": 1, "right": 2, "up": 3}

    # ------------------------------------------------------------------
    # Reward shaping 本地开关 + 系数（见类 docstring）
    # ------------------------------------------------------------------
    # True  = 对非终止原子 step 叠加"距离 + 撞墙"的 shaping reward
    # False = 完全透传 gymnasium 原生 reward（对齐 RAGEN 论文原始 FrozenLake 语义）
    use_shaped_reward: bool = True

    # shaping 系数。刻意设计成 |shaping| <<|goal reward| = 1.0，避免盖过终点主信号。
    _SHAPING_CLOSER: float = 0.02    # 曼哈顿距离到 goal 减小（朝对方向走）
    _SHAPING_FARTHER: float = -0.01  # 走完一步距离没减（反方向 / 平行）
    _SHAPING_WASTE: float = -0.05    # 撞墙（curr_s == prev_s，动作未生效）

    def __init__(self, config: Any):
        config.env_name = 'FrozenLake-v1'
        config.kwargs = {"is_slippery": False, "map_name": "4x4"}
        super().__init__(config)

        self.MAP_LOOKUP = {b"S": 1, b"F": 1, b"H": 2, b"G": 3}
        self.GRID_LOOKUP = {0: "P", 1: "_", 2: "O", 3: "G", 4: "X", 5: "√"}

        # 从 desc 里查 Goal 位置（默认 4x4 地图 = (3,3)，不硬编码以兼容自定义地图 / 8x8）。
        # 仅在 `__init__` 查一次并缓存，避免 _step_atomic 里每步都扫描 desc。
        import numpy as np
        goal_positions = np.argwhere(self.env.unwrapped.desc == b'G')
        assert len(goal_positions) > 0, "FrozenLake map has no Goal cell"
        self._goal_row: int = int(goal_positions[0][0])
        self._goal_col: int = int(goal_positions[0][1])

    def _step_atomic(self, gym_action: Optional[int]) -> Tuple[str, float, bool, bool, dict]:
        # FrozenLake 的"成功"语义：踩到 Goal（唯一能给出 reward=1.0 的终止情形）。
        # 掉洞同样是 terminated=True 但 reward=0.0，绝不能算作成功。
        # 同时在这里顺便判 action_is_effective：gymnasium 的 FrozenLake obs 就是玩家
        # 的格子索引（self.env.unwrapped.s）——step 前后位置不变即"撞墙/出界"，即
        # 动作未生效。
        prev_s = int(self.env.unwrapped.s) if gym_action is not None else None
        obs, reward, terminated, truncated, info = super()._step_atomic(gym_action)
        if gym_action is None:
            return obs, reward, terminated, truncated, info

        if bool(terminated):
            info["is_success"] = bool(float(reward) >= 1.0 - 1e-6)
        else:
            info["is_success"] = False

        curr_s = int(self.env.unwrapped.s)
        action_effective = bool(curr_s != prev_s)
        info["action_is_effective"] = action_effective

        # Reward shaping（仅在非终止步上叠加；终止步保持原生 reward 语义，详见类 docstring）。
        if self.use_shaped_reward and not terminated:
            shaping = self._compute_shaping_reward(prev_s, curr_s, action_effective)
            reward = float(reward) + shaping
            info["reward_shaping"] = shaping

        return obs, float(reward), terminated, truncated, info

    def _compute_shaping_reward(self, prev_s: int, curr_s: int, action_effective: bool) -> float:
        """
        基于位置变化计算一次 atomic step 的 shaping reward。

        规则（按优先级）：
        1. 撞墙（action_is_effective=False）→ ``_SHAPING_WASTE``
        2. 曼哈顿距离到 goal 减小 → ``_SHAPING_CLOSER``
        3. 其他（距离不变 / 变远）→ ``_SHAPING_FARTHER``
        """
        if not action_effective:
            return self._SHAPING_WASTE

        ncol = int(self.env.unwrapped.ncol)
        prev_row, prev_col = prev_s // ncol, prev_s % ncol
        curr_row, curr_col = curr_s // ncol, curr_s % ncol

        prev_dist = abs(prev_row - self._goal_row) + abs(prev_col - self._goal_col)
        curr_dist = abs(curr_row - self._goal_row) + abs(curr_col - self._goal_col)

        if curr_dist < prev_dist:
            return self._SHAPING_CLOSER
        return self._SHAPING_FARTHER

    def _format_obs(self, obs: Any) -> str:
        import numpy as np
        desc = self.env.unwrapped.desc.copy()

        ncol = self.env.unwrapped.ncol
        player_row, player_col = obs // ncol, obs % ncol

        room = np.vectorize(lambda x: self.MAP_LOOKUP.get(x, 1))(desc)

        if desc[player_row, player_col] == b'H':
            room[player_row, player_col] = 4   # X 掉进洞
        elif desc[player_row, player_col] == b'G':
            room[player_row, player_col] = 5   # √ 到达目标
        else:
            room[player_row, player_col] = 0   # P 玩家

        grid_str = '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room)
        return grid_str

    def get_valid_actions(self) -> str:
        return (
            "Valid actions: 1=Left, 2=Down, 3=Right, 4=Up. "
            "You may output a sequence separated by '||'. "
            "Example: <answer>Right || Right || Down</answer>"
        )

    def get_env_instruction(self) -> str:
        return (
            "You are solving FrozenLake. Navigate from the player (P) to the goal (G) "
            "while avoiding holes (O). Each move shifts the player by exactly one tile "
            "in the chosen direction; moving into a wall leaves the player in place.\n"
            "**Answer format**: put a sequence of moves inside <answer>...</answer>, "
            "separated by `||`. Words (Left/Down/Right/Up) or numbers (1-4) both work.\n"
            "Example: <think>The goal is to the lower right. I'll head right then down.</think>"
            "<answer>Right || Right || Down || Down</answer>"
        )

    def _parse_action_sequence(self, action_text: str) -> List[Optional[int]]:
        tokens = _split_action_tokens(action_text)
        return [self._parse_action(tok) for tok in tokens]

    def _parse_action(self, action_text: str) -> Optional[int]:
        """
        解析 FrozenLake 单动作，返回 gym 动作索引 0-3。
        既接受 "<answer>Right</answer>" 的完整 text，也接受 "Right" / "3" 的裸 token。
        """
        match = re.search(r'<answer>(.*?)</answer>', action_text, re.DOTALL)
        act_str = match.group(1).strip().lower() if match else action_text.strip().lower()

        if act_str.isdigit():
            num = int(act_str)
            if 1 <= num <= 4:
                return num - 1  # 1-4 (用户可读) → 0-3 (gym 内部)
            return None

        for k, v in self._WORD_TO_GYM.items():
            if re.search(rf'\b{k}\b', act_str):
                return v
        return None
