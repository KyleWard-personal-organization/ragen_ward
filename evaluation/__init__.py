"""
评估指标模块 / Evaluation Metrics Module
-----------------------------------
原目录名为 ``evaluate``，但这个名字会与 HuggingFace 的 ``evaluate`` 包冲突
（Python 解析器会优先识别已安装的同名第三方包），导致 ``ModuleNotFoundError:
'evaluate' is not a package``。改名为 ``evaluation`` 以彻底避免这类名字冲突。
"""

from .metrics import (
    EvaluatorMetrics,
    extract_episode_metrics,
    compute_reward_variance,
    compute_in_group_reward_std,
)

__all__ = [
    "EvaluatorMetrics",
    "extract_episode_metrics",
    "compute_reward_variance",
    "compute_in_group_reward_std",
]
