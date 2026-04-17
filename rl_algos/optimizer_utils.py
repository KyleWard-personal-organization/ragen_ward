"""
Optimizer 构造工具 / Optimizer Builder
-----------------------------------
PPO / GRPO 共享的 optimizer 创建入口，集中处理三种后端：

- ``adamw``       : torch.optim.AdamW，fp32 一阶/二阶矩 (约 8 B/param)。
                    最忠实于 RAGEN/verl 的默认实现；显存代价最高。
- ``adamw8bit``   : bitsandbytes.optim.AdamW8bit，block-wise 8-bit 量化的 state
                    (约 2 B/param)。语义上与 AdamW 几乎等价，显存占用大幅下降。
                    **对 embedding / lm_head 自动开启 32-bit override**（社区
                    成熟做法），规避大 vocab 稀疏梯度带来的偶发 NaN。
- ``adafactor``   : torch.optim.Adafactor，因子化二阶矩 (row+col 近似)，不存
                    一阶矩 m。省显存但更新规则和 AdamW 不等价，复现论文时需
                    重新调 LR；列在这里作为 adamw8bit 装不上时的 backup。

所有函数都只负责"构造 optimizer"这一件事，不修改 actor / critic 参数本身。
"""

from typing import Any, Iterable, List, Optional

import torch


def build_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    actor: Optional[torch.nn.Module] = None,
) -> torch.optim.Optimizer:
    """按 ``name`` 构造 optimizer。

    Parameters
    ----------
    name : str
        "adamw" / "adamw8bit" / "adafactor"。
    params : iterable of Parameter
        需要优化的参数；调用方负责把 actor / critic 合并好。
    lr : float
        学习率。
    actor : nn.Module, optional
        仅 ``name == "adamw8bit"`` 时使用——会在 actor 的
        input/output embedding 上注册 32-bit optimizer state override，
        避免大 vocab embedding 在 8-bit state 下偶发 NaN。
    """
    params = list(params)

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)

    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError(
                "Optimizer 'adamw8bit' requires bitsandbytes. "
                "Install via `pip install bitsandbytes` (Windows + CUDA 12 + "
                "Python>=3.10 + torch>=2.3 has a prebuilt wheel), "
                "or switch to `--optimizer adamw`."
            ) from e

        if actor is not None:
            _register_embedding_32bit_override(actor)

        return bnb.optim.AdamW8bit(params, lr=lr)

    if name == "adafactor":
        try:
            adafactor_cls = torch.optim.Adafactor
        except AttributeError as e:
            raise AttributeError(
                "torch.optim.Adafactor is only available in PyTorch >= 2.1. "
                "Upgrade torch, or use `--optimizer adamw` / `--optimizer adamw8bit`."
            ) from e
        return adafactor_cls(
            params,
            lr=lr,
            beta2_decay=-0.8,
            eps=(None, 1e-3),
            d=1.0,
            weight_decay=0.0,
            foreach=None,
            maximize=False,
        )

    raise ValueError(f"Unknown optimizer: {name!r}. Expected one of: adamw / adamw8bit / adafactor.")


def _register_embedding_32bit_override(actor: torch.nn.Module) -> None:
    """把 actor 的 input/output embedding 注册为 32-bit optimizer state。

    bitsandbytes 官方推荐：embedding 层（vocab × hidden）梯度稀疏、尺度大，
    用 8-bit state 偶发 NaN。推荐的规避方案就是对这两层强制 32-bit state，
    其余 transformer block 继续享受 8-bit 的显存收益。
    """
    try:
        from bitsandbytes.optim import GlobalOptimManager
    except ImportError:
        return

    manager = GlobalOptimManager.get_instance()

    modules_to_override: List[torch.nn.Module] = []

    if hasattr(actor, "get_input_embeddings"):
        in_emb = actor.get_input_embeddings()
        if in_emb is not None:
            modules_to_override.append(in_emb)

    if hasattr(actor, "get_output_embeddings"):
        out_emb = actor.get_output_embeddings()
        if out_emb is not None and out_emb not in modules_to_override:
            modules_to_override.append(out_emb)

    for mod in modules_to_override:
        try:
            manager.register_module_override(mod, "weight", {"optim_bits": 32})
        except Exception:
            # embedding tie / 参数共享时重复注册会抛错；忽略即可
            pass
