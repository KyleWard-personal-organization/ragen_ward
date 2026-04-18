"""
评估指标实现 / Evaluation Metrics Implementation
-----------------------------------
对齐 RAGEN 原论文 `es_manager.get_rollout_states` 给每个 episode 记录的一组指标：

| 字段                     | 含义                                                              |
|--------------------------|-------------------------------------------------------------------|
| success                  | `terminated and not truncated`（RAGEN 严格口径，见 judge_success） |
| episodic_return / reward | `sum(env_reward)` 本 episode 原生环境累计奖励（不含 format penalty）|
| num_actions              | `sum(executed_action_count)` 本 episode 执行的原子 env step 数     |
| trajectory_length        | 本 episode 的 LLM turn 数（= len(trajectory)）                    |
| action_valid_rate        | `1 - mean(any_invalid_in_sequence)` turn 级合规率                  |
| action_effective_rate    | `mean(all_effective_in_sequence)` turn 级有效率                    |
| format_compliance        | `mean(format_ok)` 本 episode 回复满足 `<think>/<answer>` 格式的比例  |

这些 episode 级指标再由 `EvaluatorMetrics.summary()` 做跨 episode 的 mean 聚合，
得到最终的 `eval/*` 标量，和 `StarPOTrainer.evaluate()` 的字段名直接对齐。
"""

from typing import Any, Dict, List
import numpy as np


def extract_episode_metrics(trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    从一条完整 trajectory 抽取 episode 级指标（RAGEN 口径）。

    调用方：`scripts/evaluate.py` 和 `StarPOTrainer.evaluate`（如果想扩展）都用它。

    Returns:
        dict: 键固定为 `episodic_return / num_actions / trajectory_length /
              action_valid_rate / action_effective_rate / format_compliance`。
              trajectory 为空时返回全 0，方便 caller 直接累加。
    """
    if not trajectory:
        return {
            "episodic_return": 0.0,
            "num_actions": 0,
            "trajectory_length": 0,
            "action_valid_rate": 0.0,
            "action_effective_rate": 0.0,
            "format_compliance": 0.0,
        }

    # episode 级奖励：只累加环境原生 reward，和训练时记在 trajectory[*]["env_reward"] 对齐。
    # 对没有记 env_reward 的老 trajectory 兜底用 "reward"（= env_reward + format_penalty），
    # 这时候 penalty 一般也是 0（当前 evaluate 不扣 penalty）。
    episodic_return = float(sum(step.get("env_reward", step.get("reward", 0.0)) for step in trajectory))

    # BaseEnv.step 在每 turn 的 info 里写了 executed_action_count（原子 env step 数）。
    # 部分老代码可能不写，缺省认为 1（一次 chat_request → 1 个原子动作）。
    num_actions = int(sum(
        (step.get("info") or {}).get("executed_action_count", 1)
        for step in trajectory
    ))

    trajectory_length = len(trajectory)

    # turn 级 invalid / effective 标志：都是 BaseEnv.step 在 last_info 里合并出来的。
    # 缺省认为 valid=True / effective=True，与 BaseEnv 里的默认一致。
    any_invalid_list = [
        bool((step.get("info") or {}).get("any_invalid_in_sequence", False))
        for step in trajectory
    ]
    all_effective_list = [
        bool((step.get("info") or {}).get("all_effective_in_sequence", True))
        for step in trajectory
    ]
    action_valid_rate = 1.0 - float(np.mean(any_invalid_list))
    action_effective_rate = float(np.mean(all_effective_list))

    # format_ok 由 rollout_one_trajectory 打在每 turn 上；若 caller 自己的 rollout
    # 没有写（兼容老路径），就用 False 做最保守兜底。
    format_compliance = float(np.mean([bool(step.get("format_ok", False)) for step in trajectory]))

    return {
        "episodic_return": episodic_return,
        "num_actions": num_actions,
        "trajectory_length": trajectory_length,
        "action_valid_rate": action_valid_rate,
        "action_effective_rate": action_effective_rate,
        "format_compliance": format_compliance,
    }


class EvaluatorMetrics:
    """
    跨 episode 的指标累加器。
    - 旧接口 `add_episode(reward, success, length)` 仍然可用（PureRLTrainer / StarPOTrainer 现仍在调用）。
    - 新接口 `add_episode_from_trajectory(trajectory, success)` 直接喂一整条 trajectory，
      自动抽取 RAGEN 对齐的 6 个字段。

    `summary()` 返回的 key 都以 `eval/` 作为前缀，可以直接丢给 TrainingTracker。
    """

    def __init__(self):
        # 兼容旧接口
        self.success_rates: List[float] = []
        self.rewards: List[float] = []
        self.trajectory_lengths: List[int] = []
        # 新接口
        self.num_actions_list: List[int] = []
        self.action_valid_rates: List[float] = []
        self.action_effective_rates: List[float] = []
        self.format_compliances: List[float] = []

    def add_episode(
        self,
        reward: float,
        success: bool,
        length: int,
        *,
        num_actions: int | None = None,
        action_valid_rate: float | None = None,
        action_effective_rate: float | None = None,
        format_compliance: float | None = None,
    ):
        """向下兼容的记录接口。前 3 个位置参数保持不变；新增 4 个 keyword-only 字段可选。"""
        self.rewards.append(reward)
        self.success_rates.append(1.0 if success else 0.0)
        self.trajectory_lengths.append(length)
        if num_actions is not None:
            self.num_actions_list.append(int(num_actions))
        if action_valid_rate is not None:
            self.action_valid_rates.append(float(action_valid_rate))
        if action_effective_rate is not None:
            self.action_effective_rates.append(float(action_effective_rate))
        if format_compliance is not None:
            self.format_compliances.append(float(format_compliance))

    def add_episode_from_trajectory(self, trajectory: List[Dict[str, Any]], success: bool):
        """直接喂一条 trajectory，自动抽所有字段（RAGEN 对齐）。"""
        ep = extract_episode_metrics(trajectory)
        self.add_episode(
            reward=ep["episodic_return"],
            success=success,
            length=ep["trajectory_length"],
            num_actions=ep["num_actions"],
            action_valid_rate=ep["action_valid_rate"],
            action_effective_rate=ep["action_effective_rate"],
            format_compliance=ep["format_compliance"],
        )

    def summary(self) -> Dict[str, float]:
        """跨 episode 聚合成 eval/* 标量字段。无数据则返回空 dict。"""
        if not self.rewards:
            return {}

        summary: Dict[str, float] = {
            "eval/success_rate": float(np.mean(self.success_rates)),
            "eval/avg_reward": float(np.mean(self.rewards)),
            "eval/avg_trajectory_length": float(np.mean(self.trajectory_lengths)),
        }
        if self.num_actions_list:
            summary["eval/avg_num_actions"] = float(np.mean(self.num_actions_list))
        if self.action_valid_rates:
            summary["eval/action_valid_rate"] = float(np.mean(self.action_valid_rates))
        if self.action_effective_rates:
            summary["eval/action_effective_rate"] = float(np.mean(self.action_effective_rates))
        if self.format_compliances:
            summary["eval/format_compliance"] = float(np.mean(self.format_compliances))
        return summary


def compute_reward_variance(rewards: List[float]) -> float:
    """辅助函数：计算奖励方差（StarPO-S 核心过滤指标 / Echo Trap 预警信号）。"""
    if not rewards:
        return 0.0
    return float(np.var(rewards))


def check_echo_trap_signs(reward_variances: List[float], entropies: List[float]) -> bool:
    """
    检测是否陷入"回声陷阱"(Echo Trap)：
    当奖励方差骤降且输出熵急剧减少时，往往预示着模型正在陷入局部捷径。
    """
    if len(reward_variances) < 5 or len(entropies) < 5:
        return False

    var_trend = np.mean(reward_variances[-3:]) < np.mean(reward_variances[:3]) * 0.1
    entropy_trend = np.mean(entropies[-3:]) < np.mean(entropies[:3]) * 0.5

    return bool(var_trend and entropy_trend)
