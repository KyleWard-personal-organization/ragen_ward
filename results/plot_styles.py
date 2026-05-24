"""Shared plotting style for the result figures.

Color encoding by filter ratio:
- filter=1.0  -> red
- filter=0.5  -> blue
- filter=0.25 -> green

Line encoding by algorithm:
- PPO  -> solid
- GRPO -> dashed
"""
from __future__ import annotations

from typing import Optional

import matplotlib


# ============== Color and line encodings ==============
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


def setup_fonts() -> Optional[str]:
    """Use a stable default font for English-only figures."""
    chosen = "DejaVu Sans"
    matplotlib.rcParams["font.sans-serif"] = [chosen]
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
