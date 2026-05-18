"""results/plot_styles.py — 统一的绘图样式配置（颜色、字体、rcParams）。

颜色编码（按 filter ratio）：
- filter=1.0  -> 红
- filter=0.5  -> 蓝
- filter=0.25 -> 绿

线型编码（按算法）：
- PPO  -> 实线
- GRPO -> 虚线
"""
from __future__ import annotations

from typing import Optional

import matplotlib
import matplotlib.font_manager as fm


# ============== 颜色 / 线型 ==============
FILTER_COLORS = {
    1.00: "#D62728",   # red
    0.50: "#1F77B4",   # blue
    0.25: "#2CA02C",   # green
}

ALGO_LINESTYLES = {
    "ppo":  "-",
    "grpo": "--",
}

ALGO_MARKERS = {
    "ppo":  "o",
    "grpo": "s",
}


# ============== 中文字体 fallback ==============
_CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Source Han Sans CN",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "PingFang SC",
    "Arial Unicode MS",
    "WenQuanYi Micro Hei",
]


def setup_fonts() -> Optional[str]:
    """选择第一个可用的中文字体；失败时返回 None 但不报错。"""
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in _CHINESE_FONT_CANDIDATES if c in available), None)
    if chosen is not None:
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


def setup_rcparams() -> None:
    matplotlib.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.7",
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def init_style() -> Optional[str]:
    chosen = setup_fonts()
    setup_rcparams()
    return chosen
