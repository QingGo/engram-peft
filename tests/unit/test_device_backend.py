# pyright: reportUnknownMemberType=none, reportPrivateUsage=none
from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest
import torch

from engram_peft.utils.device import (
    _NPU_AVAILABLE,
    _NPU_MODULE,
    _lazy_npu_check,
    _NpuGradScalerWrapper,
    create_grad_scaler,
    get_amp_device_type,
    get_available_device,
    get_optimal_precision_config,
    is_bf16_supported,
    is_cuda_available,
    is_npu_available,
)

# ---------------------------------------------------------------------------
# Helpers: reset device module global state between tests
# ---------------------------------------------------------------------------


def _reset_npu_state() -> None:
    """Reset cached NPU state to force re-detection."""
    import engram_peft.utils.device as devmod

    devmod._NPU_AVAILABLE = None
    devmod._NPU_MODULE = None


# ---------------------------------------------------------------------------
# is_cuda_available
# ---------------------------------------------------------------------------


def test_cuda_available_true() -> None:
    """When torch.cuda.is_available returns True."""
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        assert is_cuda_available() is True


def test_cuda_available_false() -> None:
    """When torch.cuda.is_available returns False."""
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        assert is_cuda_available() is False


# ---------------------------------------------------------------------------
# is_npu_available
# ---------------------------------------------------------------------------


def test_npu_available_true() -> None:
    """When torch_npu import succeeds and torch.npu.is_available() is True."""
    _reset_npu_state()
    mock_npu_module = mock.MagicMock()
    mock_npu_module.is_available.return_value = True

    with mock.patch.object(torch, "npu", mock_npu_module, create=True):
        assert is_npu_available() is True


def test_npu_available_false_import_error() -> None:
    """When torch_npu is not installed (ImportError)."""
    _reset_npu_state()
    # Remove torch.npu attribute + block import
    with mock.patch.object(torch, "npu", None, create=True):
        with mock.patch.object(importlib, "import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named torch_npu")
            _reset_npu_state()
            assert is_npu_available() is False


# ---------------------------------------------------------------------------
# get_available_device
# ---------------------------------------------------------------------------


def test_get_device_npu_first() -> None:
    """NPU takes priority over CUDA."""
    _reset_npu_state()
    mock_npu_module = mock.MagicMock()
    mock_npu_module.is_available.return_value = True

    with mock.patch.object(torch, "npu", mock_npu_module, create=True):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            assert get_available_device() == "npu"


def test_get_device_cuda_fallback() -> None:
    """When NPU not available, CUDA is next priority."""
    _reset_npu_state()
    # No torch.npu attribute
    with mock.patch.object(torch, "npu", None, create=True):
        with mock.patch.object(importlib, "import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named torch_npu")
            _reset_npu_state()
            with mock.patch.object(torch.cuda, "is_available", return_value=True):
                assert get_available_device() == "cuda"


def test_get_device_cpu_fallback() -> None:
    """When neither NPU nor CUDA, falls back to cpu."""
    _reset_npu_state()
    with mock.patch.object(torch, "npu", None, create=True):
        with mock.patch.object(importlib, "import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named torch_npu")
            _reset_npu_state()
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                assert get_available_device() == "cpu"


# ---------------------------------------------------------------------------
# is_bf16_supported
# ---------------------------------------------------------------------------


def test_bf16_npu_supported() -> None:
    """NPU BF16 support check."""
    mock_npu_module = mock.MagicMock()
    mock_npu_module.is_bf16_supported.return_value = True

    with mock.patch(
        "engram_peft.utils.device._lazy_npu_check",
        return_value=(True, mock_npu_module),
    ):
        assert is_bf16_supported("npu") is True


def test_bf16_npu_unsupported() -> None:
    """NPU BF16 unsupported (exception or False)."""
    mock_npu_module = mock.MagicMock()
    mock_npu_module.is_bf16_supported.return_value = False

    with mock.patch(
        "engram_peft.utils.device._lazy_npu_check",
        return_value=(True, mock_npu_module),
    ):
        assert is_bf16_supported("npu") is False


def test_bf16_cuda_supported() -> None:
    """CUDA BF16 support check."""
    with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
        assert is_bf16_supported("cuda") is True


def test_bf16_cpu() -> None:
    """CPU never supports BF16 via this check."""
    assert is_bf16_supported("cpu") is False


def test_bf16_auto_cpu() -> None:
    """Auto-detect on a CPU-only machine."""
    _reset_npu_state()
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        with mock.patch(
            "engram_peft.utils.device._lazy_npu_check",
            return_value=(False, None),
        ):
            assert is_bf16_supported() is False


# ---------------------------------------------------------------------------
# get_amp_device_type
# ---------------------------------------------------------------------------


def test_amp_device_npu() -> None:
    """Autocast device type is 'npu' when NPU available."""
    with mock.patch(
        "engram_peft.utils.device._lazy_npu_check",
        return_value=(True, mock.MagicMock()),
    ):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            assert get_amp_device_type() == "npu"


def test_amp_device_cuda() -> None:
    """Autocast device type is 'cuda' when only CUDA available."""
    with mock.patch(
        "engram_peft.utils.device._lazy_npu_check",
        return_value=(False, None),
    ):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            assert get_amp_device_type() == "cuda"


def test_amp_device_cpu() -> None:
    """Autocast device type is 'cpu' when no accelerator."""
    with mock.patch(
        "engram_peft.utils.device._lazy_npu_check",
        return_value=(False, None),
    ):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            assert get_amp_device_type() == "cpu"


# ---------------------------------------------------------------------------
# create_grad_scaler
# ---------------------------------------------------------------------------


def test_grad_scaler_npu() -> None:
    """Creating NPU GradScaler returns _NpuGradScalerWrapper."""
    mock_npu_module = mock.MagicMock()
    mock_npu_module.amp.GradScaler.return_value = mock.MagicMock()

    with mock.patch(
        "engram_peft.utils.device._lazy_npu_check",
        return_value=(True, mock_npu_module),
    ):
        scaler = create_grad_scaler("npu")
        assert scaler is not None
        assert isinstance(scaler, _NpuGradScalerWrapper)
        # Verify the underlying scaler was created
        mock_npu_module.amp.GradScaler.assert_called_once()


def test_grad_scaler_cuda() -> None:
    """Creating CUDA GradScaler returns _CudaGradScalerWrapper."""
    scaler = create_grad_scaler("cuda")
    assert scaler is not None
    # Verify it has the expected interface
    assert hasattr(scaler, "unscale_")
    assert hasattr(scaler, "step")
    assert hasattr(scaler, "update")
    assert hasattr(scaler, "get_scale")


def test_grad_scaler_cpu() -> None:
    """Creating GradScaler on CPU returns None."""
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        scaler = create_grad_scaler("cpu")
        assert scaler is None


# ---------------------------------------------------------------------------
# get_optimal_precision_config
# ---------------------------------------------------------------------------


def test_precision_cpu() -> None:
    """On CPU, both bf16 and fp16 are False."""
    _reset_npu_state()
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        with mock.patch(
            "engram_peft.utils.device._lazy_npu_check",
            return_value=(False, None),
        ):
            cfg = get_optimal_precision_config()
            assert cfg == {"bf16": False, "fp16": False}


def test_precision_cuda_bf16() -> None:
    """On CUDA with BF16 support."""
    _reset_npu_state()
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            with mock.patch(
                "engram_peft.utils.device._lazy_npu_check",
                return_value=(False, None),
            ):
                cfg = get_optimal_precision_config()
                assert cfg == {"bf16": True, "fp16": False}


def test_precision_cuda_fp16() -> None:
    """On CUDA without BF16 support, fall back to fp16."""
    _reset_npu_state()
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            with mock.patch(
                "engram_peft.utils.device._lazy_npu_check",
                return_value=(False, None),
            ):
                cfg = get_optimal_precision_config()
                assert cfg == {"bf16": False, "fp16": True}


def test_precision_npu_bf16() -> None:
    """On NPU with BF16 support."""
    mock_npu_module = mock.MagicMock()
    mock_npu_module.is_bf16_supported.return_value = True

    with mock.patch(
        "engram_peft.utils.device._lazy_npu_check",
        return_value=(True, mock_npu_module),
    ):
        cfg = get_optimal_precision_config()
        assert cfg == {"bf16": True, "fp16": False}


def test_precision_npu_fp16() -> None:
    """On NPU without BF16 support, fall back to fp16."""
    mock_npu_module = mock.MagicMock()
    mock_npu_module.is_bf16_supported.return_value = False

    with mock.patch(
        "engram_peft.utils.device._lazy_npu_check",
        return_value=(True, mock_npu_module),
    ):
        cfg = get_optimal_precision_config()
        assert cfg == {"bf16": False, "fp16": True}


# ---------------------------------------------------------------------------
# GradScaler Protocol compliance
# ---------------------------------------------------------------------------


def test_cuda_scaler_protocol() -> None:
    """CUDA GradScaler wrapper satisfies GradScalerProtocol."""
    from engram_peft.types import GradScalerProtocol

    scaler = create_grad_scaler("cuda")
    assert isinstance(scaler, GradScalerProtocol)


# ---------------------------------------------------------------------------
# compat.py helpers
# ---------------------------------------------------------------------------


def test_safe_npu_is_available() -> None:
    """safe_npu_is_available delegates to device module."""
    from engram_peft.utils.compat import safe_npu_is_available

    _reset_npu_state()
    with mock.patch.dict("sys.modules", {"torch_npu": mock.MagicMock()}):
        result = safe_npu_is_available()
        assert isinstance(result, bool)


def test_safe_is_bf16_supported() -> None:
    """safe_is_bf16_supported delegates to device module."""
    from engram_peft.utils.compat import safe_is_bf16_supported

    result = safe_is_bf16_supported()
    assert isinstance(result, bool)


def test_safe_get_available_device() -> None:
    """safe_get_available_device returns a string in {'cuda', 'npu', 'cpu'}."""
    from engram_peft.utils.compat import safe_get_available_device

    dev = safe_get_available_device()
    assert dev in ("cuda", "npu", "cpu")
