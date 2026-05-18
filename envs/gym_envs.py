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
    """CartPole 封装。连续平衡任务 → 仅单动作语义，不支持 `||` 序列。

    路径 A 论文对齐：无 grid_vocab（连续 obs）；action_lookup 用论文风格短表；
    env_instruction 一句话，不泄露平衡策略。
    """

    # ---- 路径 A：论文风格 prompt 五件套 ----
    env_instruction = (
        "You are controlling a CartPole. Push the cart left or right to keep the pole "
        "balanced upright as long as possible. Each turn you output exactly one action."
    )
    grid_vocab = None  # 连续 obs，无网格符号
    action_lookup = {0: "Push cart left (0)", 1: "Push cart right (1)"}
    max_actions_per_traj = 1  # CartPole 不支持 ||
    max_response_tokens = 100

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


# ---------------------------------------------------------------------------
# Sokoban：RAGEN 主力多步规划环境
# ---------------------------------------------------------------------------

class SokobanEnv(GymEnvWrapper):
    """
    Sokoban (推箱子) 环境封装。RAGEN 中的多步空间推理测试床。

    对齐 RAGEN 论文的 `||` 动作序列：一次回复可以一次性规划多步推箱子。

    路径 A 论文对齐：env_instruction 直接复用论文 ``config/envs.yaml::SimpleSokoban``
    的字面文本（一句话级别，不含 heuristic、不泄露 reward）；grid_vocab 和
    action_lookup 与论文 SokobanEnvConfig 字段一致，自动拼到 system 里。
    """

    # ---- 路径 A：论文风格 prompt 五件套（与 RAGEN config/envs.yaml::SimpleSokoban 对齐）----
    env_instruction = (
        "You are solving the Sokoban puzzle. You are the player and you need to push "
        "all boxes to targets. When you are right next to a box, you can push it by "
        "moving in the same direction. You cannot push a box through a wall, and you "
        "cannot pull a box. The answer should be a sequence of actions, like "
        "<answer>Right || Right || Up</answer>"
    )
    grid_vocab = {
        "#": "wall", "_": "empty", "O": "target", "√": "box on target",
        "X": "box", "P": "player", "S": "player on target",
    }
    action_lookup = {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
    max_actions_per_traj = 10  # 论文 SimpleSokoban 默认值
    max_response_tokens = 100  # 论文 SimpleSokoban 默认值

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

    ## 三个相互独立的开关变量

    | 类属性                 | 含义                                                     | 论文对齐值 |
    |------------------------|----------------------------------------------------------|-----------|
    | ``is_slippery``        | 冰面是否打滑（True → 0.8 沿指示方向，0.1/0.1 给两侧）    | True      |
    | ``use_shaped_reward``  | 是否叠加距离/撞墙/通关 shaping reward                    | False     |
    | ``randomize_map``      | 每次 reset 是否重新随机生成地图（基于 seed）             | True      |

    三个开关彼此完全正交，2³=8 种组合都合法：
    - **(True, False, True)**  → **论文 100% baseline**：滑动 + 稀疏 reward + 每次 reset 一张随机图
    - (False, False, False)    → 老的"小模型友好"baseline：确定性 + 稀疏 reward + 固定 4x4 默认图
    - (False, True, False)     → 老的 reward shaping 实验
    - (True, True, True)       → 论文式滑动随机图 + dense reward（合法但 shaping 信号会被稀释）
    - 其他 4 组合同理。

    ## randomize_map 实现细节

    `randomize_map=False`（默认）→ 始终用 gymnasium 自带的固定 4x4 地图（``MAPS["4x4"]``
    = SFFF/FHFH/FFFH/HFFG），跟之前一样。

    `randomize_map=True` → 每次 ``reset(seed=X)`` 时：
    1. 调 gymnasium 自带的 ``generate_random_map(size=4, p=0.9, seed=X)`` 生成新地图
       （p=0.9 = 论文值：每格 90% 概率 frozen，10% 概率 hole；保证有一条 S→G 通路）
    2. 重建底层 ``self.env``（gymnasium 0.28.x 的 P 表是 ``__init__`` 时算的，desc 改了
       必须重建才能让 ``categorical_sample`` 用上新地图）
    3. 重新缓存 Goal 位置（用于 shaping）

    这跟 RAGEN ``ragen/env/frozen_lake/env.py::reset`` 的 ``self.__init__(self.config)``
    重建路径数学等价。

    ## is_slippery=True 的实现细节

    gymnasium 0.28.x 的 FrozenLake-v1 不支持 ``success_rate`` 参数（gymnasium 1.x
    才加进去），所以我们**始终把底层 gym env 跑成 deterministic** (``is_slippery=False``
    传给 gymnasium)，然后在 ``_step_atomic`` 里**自己根据 ``slippery_success_rate``
    重采样真实执行的 action**：

        若 self.is_slippery == True:
            真实 action = a            with prob = 0.8   ← slippery_success_rate
                        = (a - 1) % 4  with prob = 0.1   ← 沿轴左转 90°
                        = (a + 1) % 4  with prob = 0.1   ← 沿轴右转 90°

    这跟论文 ``GymFrozenLakeEnv(success_rate=0.8)`` 数学完全等价（gymnasium 1.x
    源码里写的就是 fail_rate = (1 - success_rate) / 2 给 perpendicular 两侧均分）。
    随机源直接复用 gymnasium env 自己的 ``np_random``，所以 ``reset(seed=...)``
    依然能完整复现整条轨迹。

    ## Reward shaping（``use_shaped_reward=True`` 时）

    原生 FrozenLake 是"到 Goal = +1，其他一律 0"的极度稀疏奖励，对小模型 +
    有限 rollout 量的 cold-start 不友好。开 shaping 后，每个**非终止**原子 step
    按"曼哈顿距离变化 + 是否撞墙"叠一个小幅 shaping reward；真正**到达目标**
    的终止步追加一个较大的 ``_SUCCESS_BONUS`` 以主导总 return，防止 policy 通过
    "多撞墙/多来回走"这类 proxy-hack 把 shaping 刷成比通关还高的 return。

    - 到 Goal：reward = +1.0（gym 原生） + ``_SUCCESS_BONUS``
    - 掉洞：  reward = 0.0，不叠 shaping（不惩罚，避免模型学成"害怕探索"）
    - 其他：  reward = 原生 0.0 + shaping（见下方常量）

    注意：shaping 用的是 ``prev_s`` vs ``curr_s`` 的真实位置变化，跟"被滑动改变
    了真实执行 action"无关 —— 即使 ``is_slippery=True`` 也能正常工作，只是
    shaping 信号会被打滑稀释（同样的 action 时而正向 closer 时而 waste）。
    """

    # ---- 路径 A：论文风格 prompt 五件套（与 RAGEN config/envs.yaml::FrozenLake 对齐）----
    # env_instruction 直接复用论文字面文本：一句话任务描述 + 一个 example，
    # 不含任何 heuristic（不告诉模型"goal 在右下角"、"撞墙是浪费"），
    # 不含任何 reward 规则泄露（不告诉模型"到 G 给 reward 1"），
    # 不告诉模型坐标系（让 RL 自己学）。
    env_instruction = (
        "You are solving the FrozenLake puzzle. Forbid the hole and go to the target. "
        "You may move to the unintended direction due to the slippery ice. "
        "Example answer format: To forbid the hole and go to the target, I should go "
        "left then go up. <answer>Left || Up</answer>"
    )
    grid_vocab = {
        "P": "player", "_": "empty", "O": "hole", "G": "goal",
        "X": "player in hole", "√": "player on goal",
    }
    action_lookup = {1: "Left", 2: "Down", 3: "Right", 4: "Up"}
    max_actions_per_traj = 10  # 论文 FrozenLake 默认值
    max_response_tokens = 100  # 论文 FrozenLake 默认值

    # gymnasium FrozenLake 的离散动作：0=Left, 1=Down, 2=Right, 3=Up
    # 注意 (a-1)%4 / (a+1)%4 在这个编码下刚好是该动作的两个 perpendicular 方向：
    #   Left=0  → Up=3 / Down=1
    #   Down=1  → Left=0 / Right=2
    #   Right=2 → Down=1 / Up=3
    #   Up=3    → Right=2 / Left=0
    # 这跟 gymnasium 1.x FrozenLakeEnv 源码里写的滑动逻辑完全一致。
    _WORD_TO_GYM = {"left": 0, "down": 1, "right": 2, "up": 3}

    # ------------------------------------------------------------------
    # 开关 1：is_slippery —— 论文对齐 baseline 应该置 True
    # ------------------------------------------------------------------
    is_slippery: bool = True
    slippery_success_rate: float = 0.8  # 论文 RAGEN config: success_rate=0.8

    # ------------------------------------------------------------------
    # 开关 2：use_shaped_reward —— 论文对齐 baseline 应该置 False
    # ------------------------------------------------------------------
    use_shaped_reward: bool = False

    # ------------------------------------------------------------------
    # 开关 3：randomize_map —— 论文对齐 baseline 应该置 True
    # ------------------------------------------------------------------
    # True  = 每次 reset(seed=X) 用 generate_random_map(size, p, seed=X) 重生成地图
    # False = 始终用 gymnasium MAPS["4x4"] 固定地图（SFFF/FHFH/FFFH/HFFG）
    randomize_map: bool = True
    random_map_size: int = 4    # 论文 RAGEN config: size=4
    random_map_frozen_p: float = 0.9  # 论文 RAGEN config: p=0.9 (frozen 比例)

    # shaping 系数（仅在 use_shaped_reward=True 时生效）
    _SHAPING_CLOSER: float = 0.1     # 曼哈顿距离到 goal 减小（朝对方向走）
    _SHAPING_FARTHER: float = -0.00  # 走完一步距离没减（反方向 / 平行）
    _SHAPING_WASTE: float = -0.30    # 撞墙（curr_s == prev_s，动作未生效）

    # 真正到达 Goal 时，在 gym 原生 +1 之外再追加一次性 bonus。设计意图是压制 shaping
    # proxy-hack：单条 rollout 即使把 _SHAPING_CLOSER 占满也只是线性累积几步 × 0.1，而
    # 到达 goal 会一次性多拿 +2.0，拉开 return 数量级差距。
    _SUCCESS_BONUS: float = 2.0

    def __init__(self, config: Any):
        # 缓存 config，randomize_map=True 时 reset 要用它重建 self.env（拷贝一份，
        # 避免 reset 时改 config.kwargs 影响外面持有的引用）。
        import copy
        self._base_config = copy.copy(config)

        config.env_name = 'FrozenLake-v1'
        # 关键：始终让底层 gymnasium 跑 deterministic（is_slippery=False）。
        # 我们的 self.is_slippery=True 时由 _step_atomic 自己重采样 action 实现滑动，
        # 完全绕开 gymnasium 0.28.x 不支持 success_rate 参数的限制。
        config.kwargs = {"is_slippery": False, "map_name": "4x4"}
        super().__init__(config)

        self.MAP_LOOKUP = {b"S": 1, b"F": 1, b"H": 2, b"G": 3}
        self.GRID_LOOKUP = {0: "P", 1: "_", 2: "O", 3: "G", 4: "X", 5: "√"}

        # 从 desc 里查 Goal 位置（默认 4x4 地图 = (3,3)，不硬编码以兼容自定义地图 / 8x8）。
        # 仅在 `__init__` 查一次并缓存，避免 _step_atomic 里每步都扫描 desc。
        # randomize_map=True 时每次 reset 还会重新缓存（地图变了 Goal 位置可能也变）。
        self._refresh_goal_cache()

    def _refresh_goal_cache(self) -> None:
        """从当前 self.env.unwrapped.desc 提取 Goal 位置并缓存到 self._goal_row/col。
        randomize_map=True 时每次 reset 调一次（地图换了 Goal 位置可能换了）。"""
        import numpy as np
        goal_positions = np.argwhere(self.env.unwrapped.desc == b'G')
        assert len(goal_positions) > 0, "FrozenLake map has no Goal cell"
        self._goal_row: int = int(goal_positions[0][0])
        self._goal_col: int = int(goal_positions[0][1])

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict]:
        """
        randomize_map=False（默认）→ 走 GymEnvWrapper.reset 原路径（同一张固定 4x4 图）。

        randomize_map=True → 用 ``generate_random_map(size, p, seed=seed)`` 生成新地图，
        重建 self.env，再走标准 reset 流程。这跟 RAGEN 论文 ``self.__init__(self.config)``
        重建路径数学等价（但只在地图层面重建，class attribute 不会被重置）。
        """
        if self.randomize_map:
            from gymnasium.envs.toy_text.frozen_lake import generate_random_map

            # generate_random_map 内部用 numpy seeded RNG，确保 (size, p, seed) 完全
            # 决定地图布局 → 同 seed reset 产出同一张图，复现性 OK。
            new_desc = generate_random_map(
                size=int(self.random_map_size),
                p=float(self.random_map_frozen_p),
                seed=seed,
            )

            # 旧 env 显式关掉避免 Windows 上的 pygame 资源泄漏，再用 desc=... 重建。
            try:
                self.env.close()
            except Exception:
                pass
            self.env = gym.make(
                'FrozenLake-v1',
                max_episode_steps=self.max_steps,
                is_slippery=False,   # 我们自己处理滑动（见 _maybe_apply_slippery）
                desc=new_desc,
            )
            self._refresh_goal_cache()

        return super().reset(seed=seed, **kwargs)

    def _maybe_apply_slippery(self, gym_action: int) -> Tuple[int, bool]:
        """
        若 ``self.is_slippery=True``，按 0.8 / 0.1 / 0.1 重采样真实执行的 gym action。
        否则原样返回。返回 (executed_action, was_slipped)。was_slipped 仅供 info 调试。

        随机源使用 ``self.env.unwrapped.np_random``（gymnasium 内置），所以 ``reset(seed)``
        会完整复现整条 trajectory（包括滑动结果）。
        """
        if not self.is_slippery:
            return gym_action, False

        np_random = self.env.unwrapped.np_random
        # 0.8 沿原方向 / 0.1 沿 (a-1)%4 / 0.1 沿 (a+1)%4
        p_main = float(self.slippery_success_rate)
        p_side = (1.0 - p_main) / 2.0
        candidates = [(gym_action - 1) % 4, gym_action, (gym_action + 1) % 4]
        probs = [p_side, p_main, p_side]
        idx = int(np_random.choice(3, p=probs))
        executed = candidates[idx]
        return executed, bool(executed != gym_action)

    def _step_atomic(self, gym_action: Optional[int]) -> Tuple[str, float, bool, bool, dict]:
        # FrozenLake 的"成功"语义：踩到 Goal（唯一能给出 reward=1.0 的终止情形）。
        # 掉洞同样是 terminated=True 但 reward=0.0，绝不能算作成功。
        # 同时在这里顺便判 action_is_effective：gymnasium 的 FrozenLake obs 就是玩家
        # 的格子索引（self.env.unwrapped.s）——step 前后位置不变即"撞墙/出界"，即
        # 动作未生效。
        prev_s = int(self.env.unwrapped.s) if gym_action is not None else None

        # is_slippery=True 时，截获 agent 给的 action，按论文 0.8/0.1/0.1 重采样真实执行的
        # action；is_slippery=False 时原样透传。底层 gymnasium 永远是 deterministic 的，
        # 滑动语义完全由我们这一层实现 —— 跟论文 success_rate=0.8 数学等价。
        executed_action = gym_action
        was_slipped = False
        if gym_action is not None:
            executed_action, was_slipped = self._maybe_apply_slippery(gym_action)

        obs, reward, terminated, truncated, info = super()._step_atomic(executed_action)
        if gym_action is None:
            return obs, reward, terminated, truncated, info

        # 把"agent 想做什么 / 实际执行了什么"都记到 info 里供事后分析（不影响 reward）。
        info["agent_action"] = int(gym_action)
        info["executed_action"] = int(executed_action)
        info["was_slipped"] = was_slipped

        if bool(terminated):
            info["is_success"] = bool(float(reward) >= 1.0 - 1e-6)
        else:
            info["is_success"] = False

        # action_is_effective 看真实位置变化（是否离开 prev_s），跟"是 agent 自己输出
        # 的方向" 还是 "被滑出来的方向" 无关 —— 这正是论文 RAGEN 评估时的语义：只要
        # 动作让 agent 真的动了，就算 effective。
        curr_s = int(self.env.unwrapped.s)
        action_effective = bool(curr_s != prev_s)
        info["action_is_effective"] = action_effective

        # Reward shaping：非终止步叠加距离/撞墙 shaping；终止且成功时追加 _SUCCESS_BONUS；
        # 终止但失败（掉洞）保持原生 0.0（不惩罚，避免"害怕探索"）。详见类 docstring。
        if self.use_shaped_reward:
            if not terminated:
                shaping = self._compute_shaping_reward(prev_s, curr_s, action_effective)
                reward = float(reward) + shaping
                info["reward_shaping"] = shaping
            elif info.get("is_success", False):
                reward = float(reward) + self._SUCCESS_BONUS
                info["reward_shaping"] = self._SUCCESS_BONUS

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
