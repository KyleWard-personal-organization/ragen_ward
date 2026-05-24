"""Generate the final result figures.

Usage:
    python results/make_figures.py
    python results/make_figures.py --list
    python results/make_figures.py --only fig01,fig05

Figure list:
  fig01_eval_reward_6groups       - eval reward curves for six runs
  fig02_eval_success_6groups      - eval success_rate curves for six runs
  fig03_eval_format_6groups       - eval format_compliance curves for six runs
  fig04_eval_panel_5metrics       - five eval metrics across six runs
  fig05_ppo_vs_grpo_vanilla       - PPO vanilla vs GRPO vanilla summary
  fig06_filter_tradeoff_bar       - variance filter trade-off bars
  fig07_train_entropy_6groups     - train entropy across six runs
  fig08_train_kl_grad_6groups     - train kl_penalty and grad_norm dynamics
  fig09_n_grad_steps_6groups      - n_grad_steps under filter settings
  fig10_echo_trap_proxy           - format collapse vs mode convergence proxy

Output: results/figures/*.png at 300 DPI

Requirements:
- Activate the project environment, for example `conda activate CASSC`
- Install matplotlib, pandas, and numpy if needed:
  pip install matplotlib pandas numpy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

# Keep console output UTF-8 on platforms that support reconfiguration.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd  # noqa: F401  -- Imported for data_loader dependency checks
except ImportError as exc:
    print(f"[error] Missing dependency: {exc.name}")
    print("[hint] In the active conda environment, run: pip install matplotlib pandas numpy")
    sys.exit(1)

from data_loader import RUNS, Run, load_all, smooth, get_run
from plot_styles import FILTER_COLORS, ALGO_LINESTYLES, init_style


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ============== Shared helpers ==============

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
    crop: bool = True,
) -> None:
    """Save a figure.

    When suptitle_top is set, tight_layout leaves room for the figure title.
    """
    if use_tight:
        if suptitle_top is not None:
            fig.tight_layout(rect=(0, 0, 1, suptitle_top))
        else:
            fig.tight_layout()
    out = FIG_DIR / name
    if crop:
        fig.savefig(out)
    else:
        with plt.rc_context({"savefig.bbox": None}):
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
    """Plot one metric for all six runs on an axis."""
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
    """Annotate a data point with text positioned in axes-fraction space."""
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
# fig01: Eval reward curves for six runs
# ==================================================

def fig01_eval_reward_6groups(data) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    plot_runs_metric(ax, data, "eval/avg_reward", source="eval")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.6, zorder=0)

    # GRPO vanilla peak annotation.
    g = data["grpo_f10"]["eval"]
    pk_idx = g["eval/avg_reward"].idxmax()
    pk = g.loc[pk_idx]
    annotate_point(
        ax, pk["step"], pk["eval/avg_reward"],
        f'GRPO vanilla peak\n+{pk["eval/avg_reward"]:.3f} @ step {int(pk["step"])}',
        xy_text_axes=(0.42, 0.92),
    )

    # PPO vanilla final collapse annotation.
    p = data["ppo_f10"]["eval"]
    en = p.iloc[-1]
    annotate_point(
        ax, en["step"], en["eval/avg_reward"],
        f'PPO vanilla collapse\n{en["eval/avg_reward"]:.3f} @ step {int(en["step"])}',
        xy_text_axes=(0.70, 0.30),
    )

    ax.set_xlabel("training step")
    ax.set_ylabel("eval avg_reward")
    ax.set_title("Eval reward across six runs (PPO solid / GRPO dashed)")
    ax.legend(loc="lower left", ncol=2, fontsize=9.5)
    ax.set_xlim(0, 210)
    ax.margins(y=0.10)

    save(fig, "fig01_eval_reward_6groups.png")


# ==================================================
# fig02: Eval success_rate curves for six runs
# ==================================================

def fig02_eval_success_6groups(data) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    plot_runs_metric(ax, data, "eval/success_rate", source="eval")

    g = data["grpo_f10"]["eval"]
    en = g.iloc[-1]
    annotate_point(
        ax, en["step"], en["eval/success_rate"],
        f'GRPO vanilla final\nsuccess_rate = {en["eval/success_rate"]:.3f}',
        xy_text_axes=(0.55, 0.55),
    )

    ax.set_xlabel("training step")
    ax.set_ylabel("eval success_rate")
    ax.set_title("Eval success_rate across six runs (PPO solid / GRPO dashed)")
    ax.legend(loc="upper left", ncol=2, fontsize=9.5)
    ax.set_xlim(0, 210)
    ax.margins(y=0.12)
    ax.set_ylim(bottom=-0.01)

    save(fig, "fig02_eval_success_6groups.png")


# ==================================================
# fig03: Eval format_compliance curves for six runs
# ==================================================

def fig03_eval_format_6groups(data) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    plot_runs_metric(ax, data, "eval/format_compliance", source="eval")

    g = data["grpo_f10"]["eval"]
    en = g.iloc[-1]
    annotate_point(
        ax, en["step"], en["eval/format_compliance"],
        f'GRPO vanilla final\nformat = {en["eval/format_compliance"]:.3f}',
        xy_text_axes=(0.40, 0.92),
    )

    p = data["ppo_f10"]["eval"]
    pe = p.iloc[-1]
    annotate_point(
        ax, pe["step"], pe["eval/format_compliance"],
        f'PPO vanilla final\nformat = {pe["eval/format_compliance"]:.3f}',
        xy_text_axes=(0.72, 0.20),
    )

    ax.set_xlabel("training step")
    ax.set_ylabel("eval format_compliance")
    ax.set_title("Eval format_compliance across six runs (PPO solid / GRPO dashed)")
    ax.legend(loc="center left", ncol=2, fontsize=9.5)
    ax.set_xlim(0, 210)
    ax.set_ylim(-0.02, 1.02)

    save(fig, "fig03_eval_format_6groups.png")


# ==================================================
# fig04: Five eval metrics across six runs
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

    # Use the last panel for the legend.
    legend_ax = axes[-1]
    legend_ax.axis("off")
    handles = []
    for run in RUNS:
        line, = legend_ax.plot([], [], label=run.label, **style_for(run, lw=2.5))
        handles.append(line)
    legend_ax.legend(handles=handles, loc="center", fontsize=12, frameon=True,
                     title="Runs", title_fontsize=12)

    fig.suptitle("Five eval metrics across six runs", fontsize=14, fontweight="bold", y=0.99)
    save(fig, "fig04_eval_panel_5metrics.png", suptitle_top=0.94)


# ==================================================
# fig05: PPO vs GRPO vanilla summary
# ==================================================

def fig05_ppo_vs_grpo_vanilla(data) -> None:
    panels = [
        ("eval/avg_reward",        "eval avg_reward",        "(a) eval reward"),
        ("eval/format_compliance", "eval format_compliance", "(b) eval format_compliance"),
        ("eval/success_rate",      "eval success_rate",      "(c) eval success_rate"),
    ]

    train_panels = [
        ("train/entropy", "train entropy", "(d) train entropy"),
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
    ax.set_title("(d) train entropy")
    ax.set_xlim(0, 210)
    ax.legend(loc="best", fontsize=10)

    # Add a compact note contrasting the two failure modes.
    ax.text(
        0.98, 0.50,
        "PPO: entropy up + format down\n-> format collapse\n\n"
        "GRPO: entropy down + format up\n-> mode convergence",
        transform=ax.transAxes, fontsize=9, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8d8", ec="0.6", alpha=0.92),
    )

    fig.suptitle("PPO vanilla vs GRPO vanilla (filter=1.0)",
                 fontsize=14, fontweight="bold", y=0.99)
    save(fig, "fig05_ppo_vs_grpo_vanilla.png", suptitle_top=0.94)


# ==================================================
# fig06: Filter trade-off bars
# ==================================================

def _final_eval(data, short: str, key: str, last_n: int = 3) -> float:
    """Average the last N eval points to smooth endpoint noise."""
    df = data[short]["eval"]
    return float(df[key].tail(last_n).mean())


def fig06_filter_tradeoff_bar(data) -> None:
    filters = [1.0, 0.5, 0.25]
    x_labels = ["filter=1.0", "filter=0.5", "filter=0.25"]

    metrics_to_plot = [
        ("eval/avg_reward", "eval avg_reward", "(a) final eval reward"),
        ("eval/success_rate", "eval success_rate", "(b) final eval success_rate"),
        ("eval/format_compliance", "eval format_compliance", "(c) final eval format_compliance"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.24, wspace=0.28)

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

        # Value labels.
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
        ax.margins(y=0.18)
        if key == "eval/avg_reward":
            ax.axhline(0, color="black", linewidth=0.7, alpha=0.7)

    # Shared legend.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor="black", label="PPO (solid fill)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor="black", hatch="///", label="GRPO (hatched fill)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 0.035))

    fig.suptitle("Variance filter trade-off: PPO U-shape vs GRPO monotonic decline",
                 fontsize=14, fontweight="bold", y=0.99)
    save(fig, "fig06_filter_tradeoff_bar.png", use_tight=False)


# ==================================================
# fig07: Train entropy across six runs
# ==================================================

def fig07_train_entropy_6groups(data) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.24)
    handles = []
    for run in RUNS:
        df = data[run.short]["train"]
        line, = ax.plot(df["step"], smooth(df["train/entropy"], span=10),
                        label=run.label, **style_for(run, lw=2.0))
        handles.append(line)

    ax.set_xlabel("training step")
    ax.set_ylabel("train entropy (EWM-10 smoothed)")
    ax.set_title(
        "Train entropy across six runs (PPO solid / GRPO dashed)\n"
        "GRPO vanilla declines toward mode convergence; PPO vanilla rises toward format collapse",
        fontsize=12, pad=12,
    )
    ax.set_xlim(0, 210)
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.045), framealpha=0.95)

    save(fig, "fig07_train_entropy_6groups.png", use_tight=False, crop=False)


# ==================================================
# fig08: Train kl_penalty and grad_norm
# ==================================================

def fig08_train_kl_grad_6groups(data) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.23, wspace=0.20)

    # KL penalty
    ax = axes[0]
    handles = []
    for run in RUNS:
        df = data[run.short]["train"]
        line, = ax.plot(df["step"], smooth(df["train/kl_penalty"], span=8),
                        label=run.label, **style_for(run, lw=2.0))
        handles.append(line)
    ax.set_xlabel("training step")
    ax.set_ylabel("train kl_penalty (EWM-8 smoothed)")
    ax.set_title("(a) train kl_penalty")
    ax.set_xlim(0, 210)

    # Grad norm uses log scale because the values span orders of magnitude.
    ax = axes[1]
    for run in RUNS:
        df = data[run.short]["train"]
        ax.plot(df["step"], smooth(df["train/grad_norm"], span=8),
                label=run.label, **style_for(run, lw=2.0))
    ax.set_xlabel("training step")
    ax.set_ylabel("train grad_norm (EWM-8, log scale)")
    ax.set_yscale("log")
    ax.set_title("(b) train grad_norm (log scale)")
    ax.set_xlim(0, 210)

    fig.suptitle("Training dynamics across six runs: KL penalty and grad_norm",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.045), framealpha=0.95)
    save(fig, "fig08_train_kl_grad_6groups.png", use_tight=False, crop=False)


# ==================================================
# fig09: n_grad_steps across six runs
# ==================================================

def fig09_n_grad_steps_6groups(data) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    for run in RUNS:
        df = data[run.short]["train"]
        if "train/n_grad_steps" not in df.columns:
            continue
        ax.plot(df["step"], df["train/n_grad_steps"],
                label=run.label, **style_for(run, lw=2.0))

    ax.set_xlabel("training step")
    ax.set_ylabel("train n_grad_steps")
    ax.set_title(
        "n_grad_steps across six runs (filter=0.25 stays at 1)\n"
        "filter=0.25: 64 trajectories <= mini_batch=64, ratio=1, clip inactive -> single-step REINFORCE",
        fontsize=11.5, pad=12,
    )
    ax.legend(loc="center right", fontsize=9.5, ncol=2)
    ax.set_xlim(0, 210)

    # Reference line at y=1.
    ax.axhline(1, color="black", linewidth=0.8, alpha=0.5, linestyle=":")

    save(fig, "fig09_n_grad_steps_6groups.png")


# ==================================================
# fig10: Echo trap proxy comparison
# ==================================================

def fig10_echo_trap_proxy(data) -> None:
    """Compare PPO vanilla and GRPO vanilla with reward, format, and entropy.

    The third y-axis is placed outside the panel, and a shared figure legend is
    placed at the bottom to keep labels from crowding the plot area.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.8))
    fig.subplots_adjust(
        left=0.06, right=0.91, top=0.83, bottom=0.18, wspace=0.65,
    )

    cases = [
        ("ppo_f10",
         "(a) PPO vanilla\nentropy up + format down + reward down -> format collapse"),
        ("grpo_f10",
         "(b) GRPO vanilla\nentropy down + format up + reward up -> mode convergence"),
    ]

    saved_handles = None  # First-panel lines become the shared legend handles.
    for ax, (short, title) in zip(axes, cases):
        train_df = data[short]["train"]
        eval_df = data[short]["eval"]

        # Primary axis: reward.
        l1, = ax.plot(eval_df["step"], eval_df["eval/avg_reward"],
                      color="#D62728", linewidth=2.4, label="eval avg_reward")
        ax.set_xlabel("training step")
        ax.set_ylabel("eval avg_reward", color="#D62728")
        ax.tick_params(axis="y", labelcolor="#D62728")
        ax.set_xlim(0, 210)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.6, zorder=0)

        # Secondary axis 1: format.
        ax2 = ax.twinx()
        l2, = ax2.plot(eval_df["step"], eval_df["eval/format_compliance"],
                       color="#2CA02C", linewidth=2.0, linestyle="--",
                       label="eval format_compliance")
        ax2.set_ylabel("eval format_compliance", color="#2CA02C")
        ax2.tick_params(axis="y", labelcolor="#2CA02C")
        ax2.set_ylim(-0.02, 1.02)

        # Secondary axis 2: entropy, offset to the right.
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

    # Shared bottom legend.
    fig.legend(
        handles=saved_handles,
        labels=["eval avg_reward (red)",
                "eval format_compliance (green)",
                "train entropy, EWM-8 smoothed (blue)"],
        loc="lower center", ncol=3, fontsize=11,
        bbox_to_anchor=(0.5, 0.04),
        framealpha=0.95,
    )

    fig.suptitle("Echo trap proxy: format collapse vs mode convergence",
                 fontsize=14, fontweight="bold", y=0.96)
    save(fig, "fig10_echo_trap_proxy.png", use_tight=False)


# ==================================================
# Entry point
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
    parser = argparse.ArgumentParser(description="Generate the final result figures")
    parser.add_argument("--only", default=None,
                        help="Generate only selected figures, comma-separated, for example fig01,fig05")
    parser.add_argument("--list", action="store_true", help="List available figures")
    args = parser.parse_args()

    if args.list:
        print("Available figures:")
        for name, fn in ALL_FIGURES.items():
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {name}  {doc}")
        return

    chosen_font = init_style()
    if chosen_font is None:
        print("[warn] No plotting font was configured.")
    else:
        print(f"[info] Using font: {chosen_font}")

    targets = (
        list(ALL_FIGURES.keys())
        if args.only is None
        else [s.strip() for s in args.only.split(",") if s.strip()]
    )

    print(f"[info] Loading six experiment runs...")
    data = load_all()
    for short, dfs in data.items():
        n_train = len(dfs["train"])
        n_eval = len(dfs["eval"])
        print(f"  {short}: train={n_train} rows, eval={n_eval} rows")

    print(f"[info] Generating figures into results/figures/")
    for tgt in targets:
        if tgt not in ALL_FIGURES:
            print(f"[warn] Unknown figure: {tgt} (available: {list(ALL_FIGURES.keys())})")
            continue
        ALL_FIGURES[tgt](data)

    print(f"\n[done] Generated {len(targets)} figures in results/figures/")


if __name__ == "__main__":
    main()
