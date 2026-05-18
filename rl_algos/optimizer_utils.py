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

``params`` 参数同时支持两种形式（PyTorch optimizer 原生约定，bnb 也兼容）：
1. ``Iterable[Parameter]``  → 单一 lr 应用到所有参数。
2. ``List[Dict[str, Any]]`` → param_groups，每个 dict 内可写 ``"lr"`` 等
   超参 override。RAGEN 论文 actor lr=1e-6 / critic lr=1e-5 就是用这条路：
   ``[{"params": actor_p, "lr": 1e-6}, {"params": critic_p, "lr": 1e-5}]``
"""

from typing import Any, Iterable, List, Optional, Sequence, Union

import torch

# ParamSpec：要么是参数迭代器，要么是 param_groups（list of dict）
ParamSpec = Union[Iterable[torch.nn.Parameter], Sequence[dict]]


def build_optimizer(
    name: str,
    params: ParamSpec,
    lr: float,
    actor: Optional[torch.nn.Module] = None,
) -> torch.optim.Optimizer:
    """按 ``name`` 构造 optimizer。

    Parameters
    ----------
    name : str
        "adamw" / "adamw8bit" / "adafactor"。
    params : iterable of Parameter, OR list of dict (param_groups)
        需要优化的参数。两种形式（PyTorch optimizer 原生约定）：
        - ``Iterable[Parameter]``：单一 lr 应用到所有参数（旧调用方式）。
        - ``List[Dict]``：param_groups，每个 dict 里可单独写 ``"lr"`` 等超参，
          会覆盖下面的 ``lr`` 默认值。
    lr : float
        默认学习率。当 ``params`` 是 param_groups 且某 group 没写 ``"lr"`` 时
        会 fallback 到这个值（PyTorch 内部行为）。
    actor : nn.Module, optional
        仅 ``name == "adamw8bit"`` 时使用——会在 actor 的
        input/output embedding 上注册 32-bit optimizer state override，
        避免大 vocab embedding 在 8-bit state 下偶发 NaN。
    """
    # 区分两种 params 形式 → 都展开成 list（PyTorch optimizer 接受任一形式，
    # 这里 list-ify 是为了让 list of dict / list of Parameter 都能被下面 if 链复用）。
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
