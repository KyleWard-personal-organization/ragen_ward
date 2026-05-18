"""
Batched rollout / Batched chat 单元测试
-----------------------------------------
这一组测试守住 ``ragen_core.rollout_utils.batched_rollout_for_prompt`` +
``HFAgent.batched_chat_request`` 跟原串行 ``rollout_one_trajectory`` /
``HFAgent.chat_request`` 的等价性，确保改成 batched rollout 之后：

1. **env 实例隔离**：N 个独立 env 的状态互不污染（``test_independent_env_instances_no_state_sharing``）。
2. **算法/数据流等价**：deterministic mock agent + 同 seed 下，batched 路径产出的
   trajectory 字段与串行路径 **逐字 bit-equivalent**（``test_batched_matches_sequential_with_deterministic_agent``）。
3. **批量 API 真的被调用**：batched 路径每 turn 只调用一次 ``batched_chat_request``，
   不会偷偷退化到 N 次 ``chat_request``（``test_batched_path_calls_batched_api_per_turn``）。
4. **真模型 sanity（GPU 可选）**：用 0.5B Qwen 跑一次 batched generate，验证
   不同长度 prompt 凑 batch 后没有 padding 污染（``test_hf_batched_sanity_no_padding_pollution``）。

第 1-3 项纯 CPU + mock agent 即可跑，1 秒内出结果；第 4 项需要 GPU + 模型本地缓存，
否则会被 skip。

运行：
    python -m pytest tests/test_batched_rollout.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import EnvConfig  # noqa: E402
from envs import make_env  # noqa: E402
from ragen_core.rollout_utils import (  # noqa: E402
    batched_rollout_for_prompt,
    rollout_one_trajectory,
)


def _make_frozenlake():
    """所有测试都用 FrozenLake-4x4-deterministic（最稳定、reset 最快的环境）。"""
    return make_env(EnvConfig(env_name="frozenlake", max_steps=10))


def _strip_messages(turn: dict) -> dict:
    """trajectory 比对时去掉 messages 字段——它是 list of dict 的引用，
    内容由"系统拼出来的固定文案 + agent 输出的固定 response"组成，跟其他字段
    高度耦合。其余 11 个字段任意一个不一致都能立即定位 bug，无需再比 messages。"""
    return {k: v for k, v in turn.items() if k != "messages"}


# ============================================================================
# Test 1: env 实例隔离
# ============================================================================
def test_independent_env_instances_no_state_sharing():
    """构造 4 个独立 FrozenLake 实例：
    - 同 seed reset → 起点 state 必须完全一致（gymnasium 的可重现性保证）
    - 对其中 1 个 step → 其他 3 个内部 state 必须不变
    """
    envs = [_make_frozenlake() for _ in range(4)]
    try:
        states_after_reset = []
        for env in envs:
            env.reset(seed=12345)
            states_after_reset.append(int(env.env.unwrapped.s))
        assert len(set(states_after_reset)) == 1, (
            f"same-seed reset yielded different start states: {states_after_reset}"
        )

        snapshots_other = {i: int(envs[i].env.unwrapped.s) for i in [0, 2, 3]}
        envs[1].step("<answer>Right</answer>")
        for i, snap in snapshots_other.items():
            assert int(envs[i].env.unwrapped.s) == snap, (
                f"env[{i}] state mutated after stepping env[1] "
                f"(was {snap}, now {int(envs[i].env.unwrapped.s)})"
            )
    finally:
        for env in envs:
            env.close()


# ============================================================================
# Test 2: deterministic mock agent → 串行 vs batched trajectory 字段 bit-equivalent
# ============================================================================
class _DeterministicAgent:
    """每次 chat_request / batched_chat_request 都返回同一个固定 response。
    配合"同 env seed reset → 同 init state"，整条 traj 完全 deterministic：
    串行 N 次和 batched N 条 应得到 **完全一致** 的 trajectory 字段。
    """

    def __init__(self, response: str):
        self._response = response

    def chat_request(self, messages):
        return self._response

    def batched_chat_request(self, messages_list):
        return [self._response for _ in messages_list]


def test_batched_matches_sequential_with_deterministic_agent():
    """同 seed + deterministic agent → 两条路径产出的 trajectory 必须逐字段相同。
    
    这条测试一旦失败，说明 batched 路径在以下任意一处偏离了串行语义：
    - 第一轮 user message 拼接（env_instruction / valid_actions / obs）
    - format_ok / format_penalty 计算
    - env.step 的调度顺序（reward / terminated / truncated / info）
    - turn_idx 计数 / max_turn 截断时机
    """
    n_rollouts = 4
    seed = 7777
    response = "<think>plan</think><answer>Right || Down</answer>"
    agent = _DeterministicAgent(response=response)

    sequential_trajs = []
    for _ in range(n_rollouts):
        env = _make_frozenlake()
        try:
            traj = rollout_one_trajectory(
                env=env, agent=agent, seed=seed, max_turn=5,
                use_format_reward=True, format_penalty=-0.1,
            )
            sequential_trajs.append(traj)
        finally:
            env.close()

    envs = [_make_frozenlake() for _ in range(n_rollouts)]
    try:
        batched_trajs = batched_rollout_for_prompt(
            envs=envs, agent=agent, seed=seed, max_turn=5,
            use_format_reward=True, format_penalty=-0.1,
        )
    finally:
        for env in envs:
            env.close()

    assert len(sequential_trajs) == len(batched_trajs) == n_rollouts
    for k in range(n_rollouts):
        seq_t = sequential_trajs[k]
        bat_t = batched_trajs[k]
        assert len(seq_t) == len(bat_t), (
            f"traj #{k} length mismatch: sequential={len(seq_t)} batched={len(bat_t)}"
        )
        for turn_idx_, (s, b) in enumerate(zip(seq_t, bat_t)):
            ss = _strip_messages(s)
            bb = _strip_messages(b)
            assert ss == bb, (
                f"traj #{k} turn #{turn_idx_} field mismatch:\n"
                f"  sequential={ss}\n"
                f"  batched   ={bb}"
            )


# ============================================================================
# Test 3: batched 路径每 turn 只调用一次 batched_chat_request（不偷偷退化到串行）
# ============================================================================
class _CallTrackingAgent:
    """记录所有 API 调用类型 + batch size，用来确认 batched 路径走的是 batched API。"""

    def __init__(self):
        self.calls: list = []
        self._fixed_response = "<think>go</think><answer>Down</answer>"

    def chat_request(self, messages):
        self.calls.append(("chat_request", 1))
        return self._fixed_response

    def batched_chat_request(self, messages_list):
        self.calls.append(("batched_chat_request", len(messages_list)))
        return [self._fixed_response for _ in messages_list]


def test_batched_path_calls_batched_api_per_turn():
    """每 turn 都应是一次 batched_chat_request 调用，且 batch size = 当前 alive env 数。"""
    n_rollouts = 3
    max_turn = 5
    agent = _CallTrackingAgent()
    envs = [_make_frozenlake() for _ in range(n_rollouts)]
    try:
        trajs = batched_rollout_for_prompt(
            envs=envs, agent=agent, seed=1, max_turn=max_turn,
            use_format_reward=False, format_penalty=0.0,
        )
    finally:
        for env in envs:
            env.close()

    # 必须只走 batched 路径，不能退化
    assert all(c[0] == "batched_chat_request" for c in agent.calls), (
        f"expected only batched_chat_request calls, got mixed: {agent.calls}"
    )
    assert len(agent.calls) >= 1, "at least one batched call expected"
    # 每个 batch size 必须 <= n_rollouts（活着的 env 数随时间减少不增加）
    sizes = [c[1] for c in agent.calls]
    assert all(1 <= s <= n_rollouts for s in sizes), (
        f"batch sizes out of range [1, {n_rollouts}]: {sizes}"
    )
    # alive set 单调不增（一旦某条 traj terminated/truncated，后续 batch 不会再包含它）
    for prev, curr in zip(sizes, sizes[1:]):
        assert curr <= prev, (
            f"batch size grew from {prev} to {curr}; alive env count must be monotonic-non-increasing"
        )

    for traj in trajs:
        assert len(traj) >= 1, "every traj should produce at least one turn"


# ============================================================================
# Test 4: seed 以 list 形式传入时，每个 env 用对应位置的独立 seed
# ----------------------------------------------------------------------------
# 这是 evaluate batch 化的核心契约：
# - 训练侧 collect_rollouts: seed=int（同 prompt 内 R 条 traj 共用同一 reset seed）
# - 评估侧 evaluate:        seed=List[int]（N 个 episode 各自独立 seed，每条互不相关）
# 必须保证 batched 路径在 seed=list 时，每条 traj 跟串行 N 次 rollout_one_trajectory(seed=对应的)
# 严格等价（deterministic agent + 不同 seed → 不同 env 起点 → 不同 trajectory，但每条
# 跟串行版本完全一致）。
# ============================================================================
def test_batched_with_independent_seeds_matches_sequential():
    """N 个独立 seed 时，batched 路径每条 traj 跟串行版本逐字段一致。"""
    seeds = [100, 200, 300, 400]
    response = "<think>x</think><answer>Right || Down</answer>"
    agent = _DeterministicAgent(response=response)

    # 串行：N 次独立 rollout，每次用 seeds[k]
    sequential_trajs = []
    for s in seeds:
        env = _make_frozenlake()
        try:
            traj = rollout_one_trajectory(
                env=env, agent=agent, seed=s, max_turn=5,
                use_format_reward=False, format_penalty=0.0,
            )
            sequential_trajs.append(traj)
        finally:
            env.close()

    # batched：seeds 以 list 形式一次传入
    envs = [_make_frozenlake() for _ in seeds]
    try:
        batched_trajs = batched_rollout_for_prompt(
            envs=envs, agent=agent, seed=seeds, max_turn=5,
            use_format_reward=False, format_penalty=0.0,
        )
    finally:
        for env in envs:
            env.close()

    assert len(sequential_trajs) == len(batched_trajs) == len(seeds)
    for k, s in enumerate(seeds):
        seq_t = sequential_trajs[k]
        bat_t = batched_trajs[k]
        assert len(seq_t) == len(bat_t), (
            f"traj #{k} (seed={s}) length mismatch: sequential={len(seq_t)} batched={len(bat_t)}"
        )
        for turn_idx_, (sa, ba) in enumerate(zip(seq_t, bat_t)):
            ss = _strip_messages(sa)
            bb = _strip_messages(ba)
            assert ss == bb, (
                f"traj #{k} (seed={s}) turn #{turn_idx_} field mismatch:\n"
                f"  sequential={ss}\n"
                f"  batched   ={bb}"
            )


def test_batched_seed_list_length_mismatch_raises():
    """seed list 长度跟 envs 数量不匹配必须立即 ValueError——否则会偷偷错位 reset。"""
    envs = [_make_frozenlake() for _ in range(3)]
    try:
        with pytest.raises(ValueError, match="seed list length"):
            batched_rollout_for_prompt(
                envs=envs, agent=_DeterministicAgent("<think>a</think><answer>Up</answer>"),
                seed=[1, 2],  # 长度 2，envs 长度 3，必须 raise
                max_turn=3, use_format_reward=False, format_penalty=0.0,
            )
    finally:
        for env in envs:
            env.close()


# ============================================================================
# Test 5: 真模型 sanity（GPU + 已缓存模型时才跑）
# ============================================================================
def _hf_local_model_available(repo_id: str) -> bool:
    from configs.constants import MODELS_DIR
    safe = repo_id.replace("/", "_")
    return os.path.exists(os.path.join(MODELS_DIR, safe))


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available()
    or not _hf_local_model_available("Qwen/Qwen2.5-0.5B-Instruct"),
    reason="real-model batched generate needs GPU + locally cached Qwen2.5-0.5B-Instruct",
)
def test_hf_batched_sanity_no_padding_pollution():
    """用 0.5B Qwen 跑一次混合长度 prompt 的 batched generate，验证：
    - 返回数量 == 输入数量
    - 每条 response 都不是空串、不是纯 padding 解码出的 garbage
    - 长 prompt 和短 prompt 的 response 互不影响（短的不会因为长的还在生成而被多 pad 一堆 token）
    """
    from configs.config import AgentConfig
    from agents.hf_agent import HFAgent

    agent_cfg = AgentConfig(
        agent_type="hf",
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=1.0,
        max_new_tokens=48,
    )
    agent = HFAgent(agent_cfg)

    short_msg = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Reply with 'OK'."},
    ]
    long_msg = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 17 plus 25? " * 15},
    ]
    messages_list = [short_msg, long_msg, short_msg, long_msg]

    responses = agent.batched_chat_request(messages_list)
    assert len(responses) == len(messages_list)
    for i, r in enumerate(responses):
        assert isinstance(r, str), f"response[{i}] is not str: {type(r)}"
        assert len(r) > 0, f"response[{i}] is empty (likely padding-only generation bug)"
        # 必须包含可打印字母/数字字符，否则就是 padding token 解码出的乱码
        assert any(ch.isalnum() for ch in r), (
            f"response[{i}] has no alnum char (padding garbage?): {r!r}"
        )


# ============================================================================
# Test 6: 路径 A 论文风格 prompt 形状
# ----------------------------------------------------------------------------
# 检查 rollout 实际下发给模型的 messages 结构是否对齐 RAGEN ctx_manager.py：
#   - system content：以 "You're a helpful assistant." 开头 +
#     env_instruction + 自动拼装的 grid_vocab + action_lookup
#   - 每个 turn 的 user content：含 "State:" + 末尾的 FORMAT_PROMPT (含
#     "Always output: <think>...<answer>...</answer>")
#   - 第二轮起的 user content：额外含 "Reward:"
# 这条测试一旦失败，说明 rollout_utils 的 prompt 拼装偏离了路径 A 的契约。
# ============================================================================
def test_paper_aligned_prompt_shape():
    """FrozenLake 上跑 1 条 trajectory，逐条检查 messages 字段的论文风格契约。"""
    response = "<think>plan</think><answer>Right || Down</answer>"
    agent = _DeterministicAgent(response=response)
    env = _make_frozenlake()
    try:
        traj = rollout_one_trajectory(
            env=env, agent=agent, seed=0, max_turn=3,
            use_format_reward=False, format_penalty=0.0,
        )
    finally:
        env.close()

    assert len(traj) >= 1, "expected at least 1 turn"

    # ---- system content 论文风格契约 ----
    msgs0 = traj[0]["messages"]
    assert msgs0[0]["role"] == "system"
    sys_content = msgs0[0]["content"]
    # 路径 A 固定前缀
    assert sys_content.startswith("You're a helpful assistant."), (
        f"system content must start with paper prefix; got: {sys_content[:100]!r}"
    )
    # env_instruction 必须出现在 system 里
    assert "FrozenLake" in sys_content, (
        "FrozenLake env_instruction missing from system content"
    )
    # grid_vocab 自动拼装产物
    assert "The meaning of each symbol in the state is:" in sys_content, (
        "grid_vocab block missing"
    )
    # action_lookup 自动拼装产物
    assert "Your available actions are:" in sys_content, (
        "action_lookup block missing"
    )
    # max_actions_per_traj > 1 → 必须告诉模型可以串多个动作
    assert "You can make up to" in sys_content, (
        "max_actions_per_traj hint missing for FrozenLake (>1 actions allowed)"
    )
    assert "||" in sys_content, "action_separator hint missing"

    # 路径 A 严禁的 heuristic / reward 泄露
    forbidden = [
        "plan ahead",
        "lean",
        "wasted",
        "bottom-right",
        "reward 0",
        "reward 1",
        "reward = 1",
    ]
    for kw in forbidden:
        assert kw.lower() not in sys_content.lower(), (
            f"forbidden heuristic / reward leak '{kw}' must NOT appear in system content"
        )

    # ---- 第一轮 user content 论文风格契约 ----
    user0 = msgs0[1]
    assert user0["role"] == "user"
    user0_content = user0["content"]
    assert user0_content.startswith("State:"), (
        f"first user message must start with 'State:'; got: {user0_content[:50]!r}"
    )
    # 第一轮**不该**带 Reward（还没产生过 reward）
    assert "Reward:" not in user0_content, (
        "first user message must not contain Reward: (no prior step yet)"
    )
    # FORMAT_PROMPT + LENGTH_PROMPT 必须每 turn 都注入
    assert "Always output:" in user0_content, "FORMAT_PROMPT missing in first turn"
    assert "<think>" in user0_content and "<answer>" in user0_content
    assert "actions left" in user0_content, "actions_left hint missing"
    assert "Max response length:" in user0_content, "LENGTH_PROMPT missing"

    # ---- 如果有第二轮，user content 应当带 Reward + State + FORMAT_PROMPT ----
    if len(traj) >= 2:
        msgs1 = traj[1]["messages"]
        # 拿到第二轮发给模型时的最后一条 user message
        last_user_in_turn1 = next(
            m for m in reversed(msgs1) if m["role"] == "user"
        )
        c1 = last_user_in_turn1["content"]
        assert "Reward:" in c1, "second turn user message must contain Reward:"
        assert "State:" in c1, "second turn user message must contain State:"
        assert "Always output:" in c1, "FORMAT_PROMPT must repeat in every turn"


# ============================================================================
# Test 7: FrozenLake is_slippery=True 概率分布（论文 success_rate=0.8 对齐）
# ----------------------------------------------------------------------------
# 我们的 slippery 实现是绕开 gymnasium 0.28.x 不支持 success_rate 参数的限制，
# 在 _step_atomic 里自己按 0.8 / 0.1 / 0.1 重采样真实执行的 action。这条测试
# 守住：
#   1. is_slippery=False（默认）：agent action == executed action（永远不滑）
#   2. is_slippery=True：N 次 step 后 executed action 的分布近似 0.8 / 0.1 / 0.1
#   3. is_slippery=True 但 reset(seed) 后整条 trajectory 完全可复现（np_random 复用）
# ============================================================================
def _make_frozenlake_for_slippery_test(slippery: bool):
    """临时把类属性 is_slippery 设成 True，构造一个 FrozenLake 实例。
    类属性级别的设置 → 实例创建后会读到这个值。测试结束后必须复位。"""
    from envs.gym_envs import FrozenLakeEnv
    FrozenLakeEnv.is_slippery = slippery
    return _make_frozenlake()


def test_frozenlake_no_slippery_by_default():
    """is_slippery=False（默认）时 _maybe_apply_slippery 必须永远返回原 action。"""
    from envs.gym_envs import FrozenLakeEnv
    original = FrozenLakeEnv.is_slippery
    try:
        env = _make_frozenlake_for_slippery_test(slippery=False)
        try:
            env.reset(seed=42)
            for a in range(4):
                for _ in range(50):
                    executed, was_slipped = env._maybe_apply_slippery(a)
                    assert executed == a, (
                        f"is_slippery=False but action {a} was changed to {executed}"
                    )
                    assert was_slipped is False
        finally:
            env.close()
    finally:
        FrozenLakeEnv.is_slippery = original


def test_frozenlake_slippery_distribution_matches_paper_080():
    """is_slippery=True 时，N 次 step 后 executed_action 的分布应近似论文 success_rate=0.8。

    用 ``env.env.unwrapped.np_random`` 作为随机源所以是 deterministic 的，
    不会偶然 flake。这里用一个固定的 large N + 宽容的容差。
    """
    from envs.gym_envs import FrozenLakeEnv
    original = FrozenLakeEnv.is_slippery
    try:
        env = _make_frozenlake_for_slippery_test(slippery=True)
        try:
            env.reset(seed=2024)
            agent_action = 2  # gym Right
            n_total = 2000
            counts = {0: 0, 1: 0, 2: 0, 3: 0}
            n_slipped = 0
            for _ in range(n_total):
                executed, was_slipped = env._maybe_apply_slippery(agent_action)
                counts[executed] += 1
                if was_slipped:
                    n_slipped += 1

            # 论文 success_rate=0.8 → 沿 agent_action 比例 ≈ 0.8
            ratio_main = counts[agent_action] / n_total
            assert 0.75 < ratio_main < 0.85, (
                f"slippery_success_rate=0.8 expected ~0.8, got {ratio_main:.3f} "
                f"(counts={counts})"
            )
            # 两侧 perpendicular 各占 0.1
            ratio_left = counts[(agent_action - 1) % 4] / n_total
            ratio_right = counts[(agent_action + 1) % 4] / n_total
            assert 0.06 < ratio_left < 0.14, f"left side prob ≈ 0.10 expected, got {ratio_left:.3f}"
            assert 0.06 < ratio_right < 0.14, f"right side prob ≈ 0.10 expected, got {ratio_right:.3f}"
            # 完全不应滑到 (a+2)%4（反方向）
            assert counts[(agent_action + 2) % 4] == 0, (
                f"slippery should NEVER produce opposite direction; got "
                f"{counts[(agent_action + 2) % 4]} hits"
            )
            # was_slipped 计数与"非主方向"次数一致
            assert n_slipped == (n_total - counts[agent_action])
        finally:
            env.close()
    finally:
        FrozenLakeEnv.is_slippery = original


def test_frozenlake_randomize_map_false_keeps_fixed_4x4():
    """randomize_map=False（默认）时，5 个不同 seed reset 必须出同一张地图。
    这条测试守住"非论文 baseline 路径不被无意中破坏"的契约。"""
    from envs.gym_envs import FrozenLakeEnv
    original = FrozenLakeEnv.randomize_map
    try:
        FrozenLakeEnv.randomize_map = False
        descs = set()
        for s in [0, 42, 999, 12345, 67890]:
            env = _make_frozenlake()
            env.reset(seed=s)
            desc_str = b'|'.join(b''.join(row) for row in env.env.unwrapped.desc).decode()
            descs.add(desc_str)
            env.close()
        assert len(descs) == 1, (
            f"randomize_map=False expected fixed map across seeds, got {len(descs)} unique descs"
        )
        # 必须就是 gymnasium MAPS["4x4"] 那张
        assert "SFFF|FHFH|FFFH|HFFG" in descs
    finally:
        FrozenLakeEnv.randomize_map = original


def test_frozenlake_randomize_map_true_yields_distinct_maps():
    """randomize_map=True 时，5 个不同 seed 应该至少出 ≥3 张不同地图。
    不要求 5 张全不同（generate_random_map 内部有 valid-path 检验，偶尔重复属正常）。"""
    from envs.gym_envs import FrozenLakeEnv
    original = FrozenLakeEnv.randomize_map
    try:
        FrozenLakeEnv.randomize_map = True
        descs = []
        for s in [0, 42, 999, 12345, 67890]:
            env = _make_frozenlake()
            env.reset(seed=s)
            desc_str = b'|'.join(b''.join(row) for row in env.env.unwrapped.desc).decode()
            descs.append(desc_str)
            env.close()
        unique = set(descs)
        assert len(unique) >= 3, (
            f"randomize_map=True expected ≥3 distinct maps across 5 seeds; "
            f"got {len(unique)}: {descs}"
        )
        # 每张图都必须有 'S' 在 (0,0) 和 'G' 在 (size-1, size-1)（论文 generate_random_map 契约）
        for d in descs:
            rows = d.split('|')
            assert rows[0][0] == 'S', f"map {d!r} missing S at top-left"
            assert rows[-1][-1] == 'G', f"map {d!r} missing G at bottom-right"
    finally:
        FrozenLakeEnv.randomize_map = original


def test_frozenlake_randomize_map_reproducible_with_seed():
    """randomize_map=True 时，同 seed reset 必须出同一张图（论文 baseline 复现性）。"""
    from envs.gym_envs import FrozenLakeEnv
    original = FrozenLakeEnv.randomize_map
    try:
        FrozenLakeEnv.randomize_map = True

        def desc_for_seed(seed: int):
            env = _make_frozenlake()
            env.reset(seed=seed)
            d = b'|'.join(b''.join(row) for row in env.env.unwrapped.desc).decode()
            env.close()
            return d

        d1 = desc_for_seed(2024)
        d2 = desc_for_seed(2024)
        d3 = desc_for_seed(7777)
        assert d1 == d2, f"same seed produced different maps: {d1} vs {d2}"
        assert d1 != d3, f"different seeds produced same map: {d1} == {d3} (extremely unlikely)"
    finally:
        FrozenLakeEnv.randomize_map = original


def test_frozenlake_slippery_reproducible_with_seed():
    """is_slippery=True 时，同 seed reset 两次跑相同动作序列，executed_action 必须完全一致。

    这条测试守住"slippery 复用 env 的 np_random，所以 reset(seed) 能完整复现"
    这个核心契约 —— 训练里我们就是靠这个来 deterministic 复现整条 trajectory 的。
    """
    from envs.gym_envs import FrozenLakeEnv
    original = FrozenLakeEnv.is_slippery
    try:
        actions = [2, 1, 2, 3, 0, 1, 2, 1]  # gym 索引（任意序列）

        def collect_executed(seed: int):
            env = _make_frozenlake_for_slippery_test(slippery=True)
            executed_seq = []
            try:
                env.reset(seed=seed)
                for a in actions:
                    executed, _ = env._maybe_apply_slippery(a)
                    executed_seq.append(executed)
            finally:
                env.close()
            return executed_seq

        seq1 = collect_executed(seed=999)
        seq2 = collect_executed(seed=999)
        assert seq1 == seq2, (
            f"same seed must produce identical executed_action sequence; "
            f"got {seq1} vs {seq2}"
        )

        # 不同 seed 应该产出不同 sequence（弱 sanity，避免 _maybe_apply_slippery
        # 退化成永远走主方向）
        seq3 = collect_executed(seed=42)
        assert seq1 != seq3, "different seeds should yield different slippery sequences"
    finally:
        FrozenLakeEnv.is_slippery = original


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
