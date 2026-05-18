"""results/make_figures.py — 生成最终报告所需的全部图表。

用法：
    python results/make_figures.py             # 生成所有图
    python results/make_figures.py --list      # 列出所有图
    python results/make_figures.py --only fig01,fig05  # 只生成指定图

图表清单：
  fig01_eval_reward_6groups       —— 6 组 eval reward 曲线（核心主图）
  fig02_eval_success_6groups      —— 6 组 eval success_rate 曲线
  fig03_eval_format_6groups       —— 6 组 eval format_compliance 曲线
  fig04_eval_panel_5metrics       —— 6 组 eval 5 指标多面板（综合视图）
  fig05_ppo_vs_grpo_vanilla       —— PPO vanilla vs GRPO vanilla 双线叙事核心图
  fig06_filter_tradeoff_bar       —— filter trade-off 柱状图（PPO U-shape vs GRPO 单调）
  fig07_train_entropy_6groups     —— 6 组 train entropy 形态对比（mode convergence 证据）
  fig08_train_kl_grad_6groups     —— 6 组 train kl_penalty + grad_norm 训练动态
  fig09_n_grad_steps_6groups      —— 6 组 n_grad_steps（filter=0.25 算法退化可视化）
  fig10_echo_trap_proxy           —— echo trap proxy 形态对比（format collapse vs mode convergence）

输出：results/figures/*.png（300 DPI）

运行前提：
- 已激活 conda 环境（项目用 `conda activate CASSC`）
- 已安装 matplotlib + pandas + numpy
  pip install matplotlib pandas numpy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

# Windows cp936 终端的中文兼容
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd  # noqa: F401  -- 间接被 data_loader 使用
except ImportError as exc:
    print(f"[error] 缺依赖: {exc.name}")
    print("[hint] 在已激活的 conda 环境中执行: pip install matplotlib pandas numpy")
    sys.exit(1)

from data_loader import RUNS, Run, load_all, smooth, get_run
from plot_styles import FILTER_COLORS, ALGO_LINESTYLES, init_style


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ============== 通用 helpers ==============

def style_for(run: Run, lw: float = 2.0) -> dict:
    return dict(
        color=FILTER_COLORS[run.filter_ratio],
        linestyle=ALGO_LINESTYLES[run.algo],
        linewidth=lw,
    )


def save(
    fig: plt.Figure,
    name: str,
    use_tight: bool = True,
    suptitle_top: float | None = None,
) -> None:
    """保存 figure。
    - use_tight=True 时调用 fig.tight_layout()
    - 如有 suptitle，传入 suptitle_top（如 0.94）让 tight_layout 给主标题留空间
    """
    if use_tight:
        if suptitle_top is not None:
            fig.tight_layout(rect=(0, 0, 1, suptitle_top))
        else:
            fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: figures/{name}")


def plot_runs_metric(
    ax: plt.Axes,
    data: Dict,
    metric_key: str,
    source: str = "eval",
    smooth_span: int | None = None,
    lw: float = 2.0,
) -> None:
    """在 ax 上叠画所有 6 组的 metric 曲线。"""
    for run in RUNS:
        df = data[run.short][source]
        if metric_key not in df.columns:
            continue
        x = df["step"].to_numpy()
        y = df[metric_key]
        if smooth_span:
            y = smooth(y, span=smooth_span)
        ax.plot(x, y.to_numpy(), label=run.label, **style_for(run, lw=lw))


def annotate_point(
    ax: plt.Axes,
    x,
    y,
    text: str,
    xy_text_axes,
    fontsize: float = 9.5,
):
    """在 (x, y) 数据坐标处加箭头注释，注释框位置使用 axes-fraction 坐标
    (xy_text_axes 取值范围 [0, 1])，确保不会被 axis 边界裁掉。"""
    ax.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=xy_text_axes,
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0,
                        connectionstyle="arc3,rad=0.15"),
        fontsize=fontsize,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", alpha=0.92),
    )


# ==================================================
# fig01: 6 组 eval reward 曲线（核心主图）
# ==================================================

def fig01_eval_reward_6groups(data) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    plot_runs_metric(ax, data, "eval/avg_reward", source="eval")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.6, zorder=0)

    # GRPO vanilla 峰值标注（axes 右上区）
    g = data["grpo_f10"]["eval"]
    pk_idx = g["eval/avg_reward"].idxmax()
    pk = g.loc[pk_idx]
    annotate_point(
        ax, pk["step"], pk["eval/avg_reward"],
        f'GRPO vanilla 峰值\n+{pk["eval/avg_reward"]:.3f} @ step {int(pk["step"])}',
        xy_text_axes=(0.42, 0.92),
    )

    # PPO vanilla 终点崩溃标注（axes 右下区）
    p = data["ppo_f10"]["eval"]
    en = p.iloc[-1]
    annotate_point(
        ax, en["step"], en["eval/avg_reward"],
        f'PPO vanilla 崩溃\n{en["eval/avg_reward"]:.3f} @ step {int(en["step"])}',
        xy_text_axes=(0.70, 0.30),
    )

    ax.set_xlabel("训练 step")
    ax.set_ylabel("eval avg_reward")
    ax.set_title("六组实验 eval reward 演化对比 (PPO 实线 / GRPO 虚线)")
    ax.legend(loc="lower left", ncol=2, fontsize=9.5)
    ax.set_xlim(0, 210)
    ax.margins(y=0.10)

    save(fig, "fig01_eval_reward_6groups.png")


# ==================================================
# fig02: 6 组 eval success_rate 曲线
# ==================================================

def fig02_eval_success_6groups(data) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    plot_runs_metric(ax, data, "eval/success_rate", source="eval")

    g = data["grpo_f10"]["eval"]
    en = g.iloc[-1]
    annotate_point(
        ax, en["step"], en["eval/success_rate"],
        f'GRPO vanilla 终点\nsuccess_rate = {en["eval/success_rate"]:.3f}',
        xy_text_axes=(0.55, 0.55),
    )

    ax.set_xlabel("训练 step")
    ax.set_ylabel("eval success_rate")
    ax.set_title("六组实验 eval success_rate 演化对比 (PPO 实线 / GRPO 虚线)")
    ax.legend(loc="upper left", ncol=2, fontsize=9.5)
    ax.set_xlim(0, 210)
    ax.margins(y=0.12)
    ax.set_ylim(bottom=-0.01)

    save(fig, "fig02_eval_success_6groups.png")


# ==================================================
# fig03: 6 组 eval format_compliance 曲线
# ==================================================

def fig03_eval_format_6groups(data) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    plot_runs_metric(ax, data, "eval/format_compliance", source="eval")

    g = data["grpo_f10"]["eval"]
    en = g.iloc[-1]
    annotate_point(
        ax, en["step"], en["eval/format_compliance"],
        f'GRPO vanilla 终点\nformat = {en["eval/format_compliance"]:.3f}',
        xy_text_axes=(0.40, 0.92),
    )

    p = data["ppo_f10"]["eval"]
    pe = p.iloc[-1]
    annotate_point(
        ax, pe["step"], pe["eval/format_compliance"],
        f'PPO vanilla 终点\nformat = {pe["eval/format_compliance"]:.3f}',
        xy_text_axes=(0.72, 0.20),
    )

    ax.set_xlabel("训练 step")
    ax.set_ylabel("eval format_compliance")
    ax.set_title("六组实验 eval format_compliance 演化对比 (PPO 实线 / GRPO 虚线)")
    ax.legend(loc="center left", ncol=2, fontsize=9.5)
    ax.set_xlim(0, 210)
    ax.set_ylim(-0.02, 1.02)

    save(fig, "fig03_eval_format_6groups.png")


# ==================================================
# fig04: 6 组 eval 5 指标多面板
# ==================================================

def fig04_eval_panel_5metrics(data) -> None:
    panels = [
        ("eval/avg_reward",            "eval avg_reward",          "(a) eval reward"),
        ("eval/success_rate",          "eval success_rate",        "(b) eval success_rate"),
        ("eval/format_compliance",     "eval format_compliance",   "(c) eval format_compliance"),
        ("eval/action_valid_rate",     "eval action_valid_rate",   "(d) eval action_valid_rate"),
        ("eval/action_effective_rate", "eval action_effective_rate","(e) eval action_effective_rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, (key, ylabel, title) in enumerate(panels):
        ax = axes[i]
        plot_runs_metric(ax, data, key, source="eval", lw=1.8)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 210)

    # 最后一个 panel 用作 legend
    legend_ax = axes[-1]
    legend_ax.axis("off")
    handles = []
    for run in RUNS:
        line, = legend_ax.plot([], [], label=run.label, **style_for(run, lw=2.5))
        handles.append(line)
    legend_ax.legend(handles=handles, loc="center", fontsize=12, frameon=True,
                     title="实验组（6 组）", title_fontsize=12)

    fig.suptitle("六组实验 eval 五指标多面板对比", fontsize=14, fontweight="bold", y=0.99)
    save(fig, "fig04_eval_panel_5metrics.png", suptitle_top=0.94)


# ==================================================
# fig05: PPO vs GRPO vanilla 双线叙事核心图
# ==================================================

def fig05_ppo_vs_grpo_vanilla(data) -> None:
    panels = [
        ("eval/avg_reward",        "eval avg_reward",        "(a) eval reward (主性能信号)"),
        ("eval/format_compliance", "eval format_compliance", "(b) eval format_compliance"),
        ("eval/success_rate",      "eval success_rate",      "(c) eval success_rate"),
    ]

    train_panels = [
        ("train/entropy", "train entropy", "(d) train entropy (锐化方向相反)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    axes = axes.flatten()

    ppo_run = get_run("ppo_f10")
    grpo_run = get_run("grpo_f10")

    for i, (key, ylabel, title) in enumerate(panels):
        ax = axes[i]
        for run in (ppo_run, grpo_run):
            df = data[run.short]["eval"]
            ax.plot(df["step"], df[key], label=run.label, **style_for(run, lw=2.4))
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 210)
        ax.legend(loc="best", fontsize=10)

    # train entropy panel
    ax = axes[3]
    for run in (ppo_run, grpo_run):
        df = data[run.short]["train"]
        ax.plot(df["step"], smooth(df["train/entropy"], span=8),
                label=run.label, **style_for(run, lw=2.4))
    ax.set_xlabel("step")
    ax.set_ylabel("train entropy (EWM-8 smoothed)")
    ax.set_title("(d) train entropy (锐化方向相反)")
    ax.set_xlim(0, 210)
    ax.legend(loc="best", fontsize=10)

    # 在 (d) 中部右侧添加 mode convergence vs format collapse 注释
    ax.text(
        0.98, 0.50,
        "PPO: entropy ↑ + format ↓\n→ format collapse\n\n"
        "GRPO: entropy ↓ + format ↑\n→ mode convergence",
        transform=ax.transAxes, fontsize=9, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8d8", ec="0.6", alpha=0.92),
    )

    fig.suptitle("PPO vanilla vs GRPO vanilla：双线叙事核心对比 (filter=1.0)",
                 fontsize=14, fontweight="bold", y=0.99)
    save(fig, "fig05_ppo_vs_grpo_vanilla.png", suptitle_top=0.94)


# ==================================================
# fig06: filter trade-off 柱状图
# ==================================================

def _final_eval(data, short: str, key: str, last_n: int = 3) -> float:
    """取最后 N 个 eval 点的均值（终点 ± 噪声平滑）。"""
    df = data[short]["eval"]
    return float(df[key].tail(last_n).mean())


def fig06_filter_tradeoff_bar(data) -> None:
    filters = [1.0, 0.5, 0.25]
    x_labels = ["filter=1.0", "filter=0.5", "filter=0.25"]

    metrics_to_plot = [
        ("eval/avg_reward", "eval avg_reward", "(a) eval reward (终点)"),
        ("eval/success_rate", "eval success_rate", "(b) eval success_rate (终点)"),
        ("eval/format_compliance", "eval format_compliance", "(c) eval format_compliance (终点)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))

    bar_w = 0.36
    x = np.arange(len(filters))

    ppo_short = ["ppo_f10", "ppo_f05", "ppo_f025"]
    grpo_short = ["grpo_f10", "grpo_f05", "grpo_f025"]

    for i, (key, ylabel, title) in enumerate(metrics_to_plot):
        ax = axes[i]
        ppo_vals = [_final_eval(data, s, key) for s in ppo_short]
        grpo_vals = [_final_eval(data, s, key) for s in grpo_short]

        b1 = ax.bar(x - bar_w / 2, ppo_vals, bar_w, label="PPO",
                    color=[FILTER_COLORS[f] for f in filters], edgecolor="black", linewidth=1.0)
        b2 = ax.bar(x + bar_w / 2, grpo_vals, bar_w, label="GRPO",
                    color=[FILTER_COLORS[f] for f in filters], edgecolor="black", linewidth=1.0,
                    hatch="///")

        # 数值标签
        for bars, vals in ((b1, ppo_vals), (b2, grpo_vals)):
            for bar, v in zip(bars, vals):
                offset = 0.01 * (max(abs(np.array(ppo_vals + grpo_vals))) or 1.0)
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (offset if v >= 0 else -offset * 3),
                        f"{v:+.3f}" if key == "eval/avg_reward" else f"{v:.3f}",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if key == "eval/avg_reward":
            ax.axhline(0, color="black", linewidth=0.7, alpha=0.7)

    # 共享 legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="0.5", edgecolor="black", label="PPO (实色)"),
        plt.Rectangle((0, 0), 1, 1, color="0.5", edgecolor="black", hatch="///", label="GRPO (斜线填充)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Variance filter trade-off：PPO U-shape vs GRPO 单调下降",
                 fontsize=14, fontweight="bold", y=0.99)
    save(fig, "fig06_filter_tradeoff_bar.png", suptitle_top=0.92)


# ==================================================
# fig07: 6 组 train entropy 对比
# ==================================================

def fig07_train_entropy_6groups(data) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for run in RUNS:
        df = data[run.short]["train"]
        ax.plot(df["step"], smooth(df["train/entropy"], span=10),
                label=run.label, **style_for(run, lw=2.0))

    ax.set_xlabel("训练 step")
    ax.set_ylabel("train entropy (EWM-10 smoothed)")
    ax.set_title(
        "六组实验 train entropy 形态对比 (PPO 实线 / GRPO 虚线)\n"
        "GRPO vanilla 单调下降 → mode convergence；PPO vanilla 升高 → format collapse",
        fontsize=12, pad=12,
    )
    ax.legend(loc="center right", ncol=2, fontsize=9.5)
    ax.set_xlim(0, 210)

    save(fig, "fig07_train_entropy_6groups.png")


# ==================================================
# fig08: 6 组 train kl_penalty + grad_norm
# ==================================================

def fig08_train_kl_grad_6groups(data) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # KL penalty
    ax = axes[0]
    for run in RUNS:
        df = data[run.short]["train"]
        ax.plot(df["step"], smooth(df["train/kl_penalty"], span=8),
                label=run.label, **style_for(run, lw=2.0))
    ax.set_xlabel("训练 step")
    ax.set_ylabel("train kl_penalty (EWM-8 smoothed)")
    ax.set_title("(a) train kl_penalty 演化")
    ax.set_xlim(0, 210)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    # Grad norm (log scale 因为量级跨度大)
    ax = axes[1]
    for run in RUNS:
        df = data[run.short]["train"]
        ax.plot(df["step"], smooth(df["train/grad_norm"], span=8),
                label=run.label, **style_for(run, lw=2.0))
    ax.set_xlabel("训练 step")
    ax.set_ylabel("train grad_norm (EWM-8, log scale)")
    ax.set_yscale("log")
    ax.set_title("(b) train grad_norm 演化（log scale）")
    ax.set_xlim(0, 210)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    fig.suptitle("六组实验训练动态：KL penalty + grad_norm",
                 fontsize=14, fontweight="bold", y=0.99)
    save(fig, "fig08_train_kl_grad_6groups.png", suptitle_top=0.93)


# ==================================================
# fig09: 6 组 n_grad_steps（filter=0.25 算法退化可视化）
# ==================================================

def fig09_n_grad_steps_6groups(data) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    for run in RUNS:
        df = data[run.short]["train"]
        if "train/n_grad_steps" not in df.columns:
            continue
        ax.plot(df["step"], df["train/n_grad_steps"],
                label=run.label, **style_for(run, lw=2.0))

    ax.set_xlabel("训练 step")
    ax.set_ylabel("train n_grad_steps")
    ax.set_title(
        "六组实验 n_grad_steps 对比 (filter=0.25 全程退化为 1)\n"
        "filter=0.25 → 64 traj ≤ mini_batch=64 → ratio=1 → clip 不触发 → 退化为 single-step REINFORCE",
        fontsize=11.5, pad=12,
    )
    ax.legend(loc="center right", fontsize=9.5, ncol=2)
    ax.set_xlim(0, 210)

    # 在 y=1 处加水平参考线
    ax.axhline(1, color="black", linewidth=0.8, alpha=0.5, linestyle=":")

    save(fig, "fig09_n_grad_steps_6groups.png")


# ==================================================
# fig10: echo trap proxy 对比图
# ==================================================

def fig10_echo_trap_proxy(data) -> None:
    """左 = PPO vanilla（format collapse），右 = GRPO vanilla（mode convergence），
    每个子图同时显示 entropy / format / reward 三轴信号。
    三轴外置布局 → 用手动 subplots_adjust + 公共 figure 底部 legend，
    避免 axes 内 legend 与多行 axes title 视觉挤压。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.8))
    fig.subplots_adjust(
        left=0.06, right=0.91, top=0.83, bottom=0.18, wspace=0.65,
    )

    cases = [
        ("ppo_f10",
         "(a) PPO vanilla\nentropy ↑ + format ↓ + reward ↓ → format collapse"),
        ("grpo_f10",
         "(b) GRPO vanilla\nentropy ↓ + format ↑ + reward ↑ → mode convergence"),
    ]

    saved_handles = None  # 第一次 panel 生成的三条线作为 figure 底部 legend handles
    for ax, (short, title) in zip(axes, cases):
        train_df = data[short]["train"]
        eval_df = data[short]["eval"]

        # 主轴 = reward (红)
        l1, = ax.plot(eval_df["step"], eval_df["eval/avg_reward"],
                      color="#D62728", linewidth=2.4, label="eval avg_reward")
        ax.set_xlabel("训练 step")
        ax.set_ylabel("eval avg_reward", color="#D62728")
        ax.tick_params(axis="y", labelcolor="#D62728")
        ax.set_xlim(0, 210)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.6, zorder=0)

        # 副轴 1：format (绿)
        ax2 = ax.twinx()
        l2, = ax2.plot(eval_df["step"], eval_df["eval/format_compliance"],
                       color="#2CA02C", linewidth=2.0, linestyle="--",
                       label="eval format_compliance")
        ax2.set_ylabel("eval format_compliance", color="#2CA02C")
        ax2.tick_params(axis="y", labelcolor="#2CA02C")
        ax2.set_ylim(-0.02, 1.02)

        # 副轴 2：entropy (蓝，外置 axes 1.18 处)
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.18))
        l3, = ax3.plot(train_df["step"], smooth(train_df["train/entropy"], span=8),
                       color="#1F77B4", linewidth=1.8, linestyle=":",
                       label="train entropy (smoothed)")
        ax3.set_ylabel("train entropy", color="#1F77B4")
        ax3.tick_params(axis="y", labelcolor="#1F77B4")

        ax.set_title(title, pad=10)
        if saved_handles is None:
            saved_handles = [l1, l2, l3]

    # 公共 figure 底部 legend，避免 axes 内 legend 挤压
    fig.legend(
        handles=saved_handles,
        labels=["eval avg_reward (红)",
                "eval format_compliance (绿)",
                "train entropy, EWM-8 smoothed (蓝)"],
        loc="lower center", ncol=3, fontsize=11,
        bbox_to_anchor=(0.5, 0.04),
        framealpha=0.95,
    )

    fig.suptitle("Echo trap proxy 形态对比：format collapse vs mode convergence",
                 fontsize=14, fontweight="bold", y=0.96)
    save(fig, "fig10_echo_trap_proxy.png", use_tight=False)


# ==================================================
# 入口
# ==================================================

ALL_FIGURES: Dict[str, Callable] = {
    "fig01": fig01_eval_reward_6groups,
    "fig02": fig02_eval_success_6groups,
    "fig03": fig03_eval_format_6groups,
    "fig04": fig04_eval_panel_5metrics,
    "fig05": fig05_ppo_vs_grpo_vanilla,
    "fig06": fig06_filter_tradeoff_bar,
    "fig07": fig07_train_entropy_6groups,
    "fig08": fig08_train_kl_grad_6groups,
    "fig09": fig09_n_grad_steps_6groups,
    "fig10": fig10_echo_trap_proxy,
}


def main():
    parser = argparse.ArgumentParser(description="生成最终报告所需的全部图表")
    parser.add_argument("--only", default=None,
                        help="仅生成指定的 figure（逗号分隔），例如 fig01,fig05")
    parser.add_argument("--list", action="store_true", help="列出所有可用图")
    args = parser.parse_args()

    if args.list:
        print("可用图表：")
        for name, fn in ALL_FIGURES.items():
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {name}  {doc}")
        return

    chosen_font = init_style()
    if chosen_font is None:
        print("[warn] 未找到中文字体；标签可能显示为方框。建议安装 'Microsoft YaHei' 或 'SimHei'。")
    else:
        print(f"[info] 使用字体: {chosen_font}")

    targets = (
        list(ALL_FIGURES.keys())
        if args.only is None
        else [s.strip() for s in args.only.split(",") if s.strip()]
    )

    print(f"[info] 加载 6 组实验数据...")
    data = load_all()
    for short, dfs in data.items():
        n_train = len(dfs["train"])
        n_eval = len(dfs["eval"])
        print(f"  {short}: train={n_train} 行, eval={n_eval} 行")

    print(f"[info] 开始生成图表，输出目录: results/figures/")
    for tgt in targets:
        if tgt not in ALL_FIGURES:
            print(f"[warn] 未知图表: {tgt}（可用: {list(ALL_FIGURES.keys())}）")
            continue
        ALL_FIGURES[tgt](data)

    print(f"\n[done] 共生成 {len(targets)} 张图。位于 results/figures/")


if __name__ == "__main__":
    main()
