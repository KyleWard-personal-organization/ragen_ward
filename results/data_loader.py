"""Load metrics.jsonl files for the six result runs.

Each run corresponds to one `<exp_name>_metrics.jsonl` file, with one JSON
object per line. Rows are split into train and eval DataFrames by metric prefix.

PPO and GRPO expose slightly different schemas:
- PPO has `train/critic_loss` and `train/in_group_reward_std`
- GRPO has `train/group_adv_mean` and `train/group_adv_std`, but no critic
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
LOG_DIR = REPO_ROOT / "logs"


@dataclass(frozen=True)
class Run:
    """Metadata for one experiment run."""
    algo: str               # 'ppo' or 'grpo'
    filter_ratio: float     # 1.0, 0.5, or 0.25
    file_stem: str          # logs/<file_stem>_metrics.jsonl
    label: str              # Full legend label
    short_label: str        # Compact panel label
    short: str              # Dict key and filename id

    @property
    def linestyle(self) -> str:
        return "-" if self.algo == "ppo" else "--"


# Six runs, each mapped to one metrics file under logs/.
RUNS = [
    Run("ppo",  1.00, "ragen_baseline_0.5B_ppo_nofilter",     "PPO + filter=1.0",  "PPO 1.0",  "ppo_f10"),
    Run("ppo",  0.50, "ragen_baseline_0.5B_ppo_filter05_v2",  "PPO + filter=0.5",  "PPO 0.5",  "ppo_f05"),
    Run("ppo",  0.25, "ragen_baseline_0.5B_ppo_filter025",    "PPO + filter=0.25", "PPO 0.25", "ppo_f025"),
    Run("grpo", 1.00, "ragen_baseline_0.5B_grpo_nofilter",    "GRPO + filter=1.0", "GRPO 1.0", "grpo_f10"),
    Run("grpo", 0.50, "ragen_baseline_0.5B_grpo_filter05",    "GRPO + filter=0.5", "GRPO 0.5", "grpo_f05"),
    Run("grpo", 0.25, "ragen_baseline_0.5B_grpo_filter025",   "GRPO + filter=0.25","GRPO 0.25","grpo_f025"),
]


def load_run(run: Run) -> Dict[str, pd.DataFrame]:
    """Load one run and return {'train': df, 'eval': df}."""
    path = LOG_DIR / f"{run.file_stem}_metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    train_rows, eval_rows = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("_event") == "run_start":
                continue
            keys = data.keys()
            has_train = any(k.startswith("train/") for k in keys)
            has_eval = any(k.startswith("eval/") for k in keys)
            if has_eval:
                eval_rows.append(data)
            elif has_train:
                train_rows.append(data)

    train_df = (
        pd.DataFrame(train_rows).sort_values("step").reset_index(drop=True)
        if train_rows else pd.DataFrame()
    )
    eval_df = (
        pd.DataFrame(eval_rows).sort_values("step").reset_index(drop=True)
        if eval_rows else pd.DataFrame()
    )

    # Average duplicate numeric rows at the same step. This handles cases where
    # periodic eval and final eval rows are both present at the last step.
    if not eval_df.empty:
        num_cols = eval_df.select_dtypes(include="number").columns
        eval_df = (
            eval_df.groupby("step", as_index=False)[num_cols].mean()
            .sort_values("step").reset_index(drop=True)
        )
    if not train_df.empty:
        num_cols = train_df.select_dtypes(include="number").columns
        train_df = (
            train_df.groupby("step", as_index=False)[num_cols].mean()
            .sort_values("step").reset_index(drop=True)
        )
    return {"train": train_df, "eval": eval_df}


def load_all() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all six runs as {short: {'train': df, 'eval': df}}."""
    return {run.short: load_run(run) for run in RUNS}


def smooth(s: pd.Series, span: int = 10) -> pd.Series:
    """Exponentially weighted smoothing for step-level noise."""
    return s.ewm(span=span, adjust=False).mean()


def get_run(short: str) -> Run:
    """Find a run by short id."""
    for r in RUNS:
        if r.short == short:
            return r
    raise KeyError(short)
