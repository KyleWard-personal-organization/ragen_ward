"""
训练指标追踪模块 / Training Metric Tracker
-----------------------------------
一个轻量级的指标追踪器，专为单机本地训练场景设计：
- 默认只打 loguru 日志 + 写 JSON Lines（JSONL）文件
- 如果用户安装了 wandb 且显式启用，会同时把指标推送到 wandb

使用方式（来自 ragen_core/starpo_trainer.py）：
    tracker = TrainingTracker(exp_name="exp1", log_dir="logs/", use_wandb=False)
    tracker.log({"train/actor_loss": 0.05, "train/reward_mean": 0.7}, step=3)
    tracker.close()
"""

import json
import os
import time
from typing import Any, Dict, Optional

from utils.logger import logger
from configs.constants import LOG_DIR


class TrainingTracker:
    """
    训练指标追踪器。
    - 所有 `log` 调用都会 1) 格式化输出到 loguru 日志；2) 追加一行 JSON 到 metrics.jsonl 文件；
      3) 如果 `use_wandb=True` 且 wandb 可用，则 `wandb.log(...)`。
    """

    def __init__(
        self,
        exp_name: str,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "ragen_ward",
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        self.exp_name = exp_name
        self.log_dir = log_dir if log_dir is not None else LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.log_dir, f"{exp_name}_metrics.jsonl")

        # 清空之前的 metrics 文件（避免不同 run 互相叠加），加时间戳分隔
        self._jsonl_fp = open(self.jsonl_path, "a", encoding="utf-8")
        self._jsonl_fp.write(json.dumps({"_event": "run_start", "time": time.time()}) + "\n")
        self._jsonl_fp.flush()

        # 可选 wandb
        self.use_wandb = False
        if use_wandb:
            try:
                import wandb  # type: ignore

                wandb.init(project=wandb_project, name=exp_name, config=wandb_config or {})
                self._wandb = wandb
                self.use_wandb = True
                logger.info(f"[Tracker] wandb enabled, project={wandb_project}, run={exp_name}")
            except Exception as e:
                logger.warning(f"[Tracker] wandb requested but init failed ({e}), falling back to loguru only.")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录一批指标。step 一般是训练循环中的 global_step。
        """
        # 清洗 None 值和不可序列化的张量（避免 JSON 报错）
        cleaned: Dict[str, Any] = {}
        for k, v in metrics.items():
            if v is None:
                continue
            if hasattr(v, "item") and callable(v.item):
                try:
                    v = v.item()
                except Exception:
                    pass
            cleaned[k] = v

        # loguru 可读格式
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in cleaned.items()]
        prefix = f"[step={step}] " if step is not None else ""
        logger.info(prefix + " ".join(parts))

        # JSONL 写盘，便于事后 pandas 读取分析
        record = dict(cleaned)
        if step is not None:
            record["step"] = step
        record["_time"] = time.time()
        try:
            self._jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_fp.flush()
        except TypeError:
            # 理论上 cleaned 已经去掉了张量，但保险处理一下
            safe_record = {k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v))
                           for k, v in record.items()}
            self._jsonl_fp.write(json.dumps(safe_record, ensure_ascii=False) + "\n")
            self._jsonl_fp.flush()

        if self.use_wandb:
            try:
                self._wandb.log(cleaned, step=step)
            except Exception as e:
                logger.warning(f"[Tracker] wandb.log failed: {e}")

    def close(self) -> None:
        """关闭文件句柄、收尾 wandb。"""
        try:
            self._jsonl_fp.write(json.dumps({"_event": "run_end", "time": time.time()}) + "\n")
            self._jsonl_fp.close()
        except Exception:
            pass
        if self.use_wandb:
            try:
                self._wandb.finish()
            except Exception:
                pass
