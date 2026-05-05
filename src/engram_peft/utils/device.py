# device.py — Unified device backend abstraction for CUDA, NPU, and CPU
# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportAttributeAccessIssue=none
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NPU 惰性检测
# ---------------------------------------------------------------------------

_NPU_AVAILABLE: bool | None = None
_NPU_MODULE: Any = None


def _lazy_npu_check() -> tuple[bool, Any]:
    global _NPU_AVAILABLE, _NPU_MODULE
    if _NPU_AVAILABLE is not None:
        return _NPU_AVAILABLE, _NPU_MODULE

    # 1. If torch already has npu attribute (torch_npu might be pre-loaded)
    npu_attr = getattr(torch, "npu", None)
    if npu_attr is not None:
        try:
            available = bool(npu_attr.is_available())
        except Exception:
            available = False
        _NPU_AVAILABLE = available
        _NPU_MODULE = npu_attr if available else None
        return _NPU_AVAILABLE, _NPU_MODULE

    # 2. Try importing torch_npu
    try:
        import torch_npu  # noqa: F401, PLC0415  # pyright: ignore[reportMissingImports, reportUnusedImport]

        npu_module = torch.npu
        _NPU_AVAILABLE = bool(npu_module.is_available())  # pyright: ignore[reportUnknownArgumentType]
        _NPU_MODULE = npu_module if _NPU_AVAILABLE else None
    except (ImportError, AttributeError):
        _NPU_AVAILABLE = False
        _NPU_MODULE = None
        logger.debug("torch_npu not available, NPU support disabled.")
    return _NPU_AVAILABLE, _NPU_MODULE


# ---------------------------------------------------------------------------
# 公共检测 API
# ---------------------------------------------------------------------------


def is_cuda_available() -> bool:
    """CUDA 可用性检查（类型安全包装）。"""
    return bool(torch.cuda.is_available())


def is_npu_available() -> bool:
    """NPU 可用性检查（惰性导入 torch_npu）。"""
    available, _ = _lazy_npu_check()
    return available


def get_available_device() -> str:
    """返回当前最优设备类型字符串。

    Return:
        "npu" — NPU 可用
        "cuda" — CUDA 可用
        "cpu"  — 均不可用
    """
    if is_npu_available():
        return "npu"
    if is_cuda_available():
        return "cuda"
    return "cpu"


DeviceType = str  # "cuda" | "npu" | "cpu"


def is_bf16_supported(device_type: str | None = None) -> bool:
    """检测当前设备是否支持 bfloat16。

    Args:
        device_type: 设备类型。为 None 时自动检测。

    Returns:
        bool: BF16 支持状态。
    """
    if device_type is None:
        device_type = get_available_device()

    if device_type == "npu":
        _, npu_module = _lazy_npu_check()
        if npu_module is not None:
            try:
                return bool(npu_module.is_bf16_supported())
            except Exception:
                return False
        return False

    if device_type == "cuda":
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False

    return False


def get_amp_device_type() -> str:
    """返回 autocast 所需的 device_type 字符串。

    优先级：npu > cuda > cpu。
    """
    if is_npu_available():
        return "npu"
    if is_cuda_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# 分布式后端检测
# ---------------------------------------------------------------------------


def get_distributed_backend() -> str:
    """返回当前硬件对应的 ``torch.distributed`` 后端名称。

    Returns:
        ``"hccl"`` — NPU 可用，应使用华为集合通信库 (HCCL)，而非 NCCL。
        ``"nccl"`` — CUDA 可用（标准 NVIDIA NCCL）。
        ``""``     — 仅 CPU，无分布式后端。
    """
    if is_npu_available():
        return "hccl"
    if is_cuda_available():
        return "nccl"
    return ""


def is_hccl_available() -> bool:
    """HCCL（华为集合通信库）可用性检测。

    当 NPU 可用时，HCCL 通常随 ``torch_npu`` 一同可用。
    本函数简单代理给 ``is_npu_available()``。
    """
    return is_npu_available()


# ---------------------------------------------------------------------------
# AMP 梯度缩放器工厂
# ---------------------------------------------------------------------------


class _CudaGradScalerWrapper:
    """CUDA GradScaler 的轻量包装，统一接口。"""

    def __init__(self) -> None:
        self._scaler: Any = torch.amp.GradScaler("cuda")

    def unscale_(self, optimizer: Optimizer) -> None:
        self._scaler.unscale_(optimizer)

    def step(self, optimizer: Optimizer, **kwargs: Any) -> Any:
        return self._scaler.step(optimizer, **kwargs)

    def update(self) -> None:
        self._scaler.update()

    def get_scale(self) -> float:
        return float(self._scaler.get_scale())


class _NpuGradScalerWrapper:
    """NPU GradScaler 的轻量包装，统一接口。"""

    def __init__(self) -> None:
        _, npu_module = _lazy_npu_check()
        if npu_module is None:
            raise RuntimeError("torch_npu is not available.")
        self._scaler: Any = npu_module.amp.GradScaler()

    def unscale_(self, optimizer: Optimizer) -> None:
        self._scaler.unscale_(optimizer)

    def step(self, optimizer: Optimizer, **kwargs: Any) -> Any:
        return self._scaler.step(optimizer, **kwargs)

    def update(self) -> None:
        self._scaler.update()

    def get_scale(self) -> float:
        return float(self._scaler.get_scale())


def create_grad_scaler(
    device_type: str | None = None,
) -> Any:
    """根据设备类型创建对应的 GradScaler 实例。

    Args:
        device_type: 设备类型。为 None 时自动检测。

    Returns:
        满足 GradScalerProtocol 的包装实例，或 None（CPU 环境）。
    """
    if device_type is None:
        device_type = get_available_device()

    if device_type == "npu":
        return _NpuGradScalerWrapper()
    if device_type == "cuda":
        return _CudaGradScalerWrapper()
    return None


def get_optimal_precision_config() -> dict[str, bool]:
    """返回基于可用硬件的最佳精度配置。"""
    device_type = get_available_device()
    if device_type == "cpu":
        return {"bf16": False, "fp16": False}

    supports_bf16 = is_bf16_supported(device_type)
    return {
        "bf16": supports_bf16,
        "fp16": not supports_bf16,
    }


# ---------------------------------------------------------------------------
# Multi-GPU safety
# ---------------------------------------------------------------------------


def ensure_single_gpu() -> None:
    """Force single GPU when DDP/torchrun is not explicitly configured.

    Engram's nested ``nn.ModuleDict`` is incompatible with PyTorch's
    ``DataParallel`` (``nn.parallel.replicate`` does not properly copy
    deeply nested submodules). When multiple GPUs are detected but no DDP
    launcher is active, set ``CUDA_VISIBLE_DEVICES=0`` to prevent the
    Trainer from wrapping the model in DataParallel.

    Call this early in your entrypoint (before any model loading).
    Has no effect when ``WORLD_SIZE`` is set (DDP/torchrun mode).

    Example::

        from engram_peft.utils.device import ensure_single_gpu
        ensure_single_gpu()
        model = get_engram_model(...)
        trainer = EngramTrainer(...)
    """
    if "WORLD_SIZE" not in os.environ and torch.cuda.device_count() > 1:
        gpu_count = torch.cuda.device_count()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        logger.info(
            "Engram: detected %d GPUs but no DDP config. "
            + "Limiting to GPU 0 (DataParallel incompatible with "
            + "nested ModuleDict). Use torchrun for multi-GPU.",
            gpu_count,
        )
