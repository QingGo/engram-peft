import logging
import sys
from types import ModuleType

import torch.nn as nn

logger = logging.getLogger(__name__)


def patch_set_submodule() -> None:
    """Polyfill for set_submodule which is missing in some PyTorch versions."""
    if not hasattr(nn.Module, "set_submodule"):
        logger.info("Patching nn.Module.set_submodule")

        def set_submodule(self: nn.Module, target: str, module: nn.Module) -> None:
            if not target:
                raise ValueError("Cannot set empty submodule")
            atoms = target.split(".")
            name = atoms.pop(-1)
            parent = self.get_submodule(".".join(atoms))
            setattr(parent, name, module)

        nn.Module.set_submodule = set_submodule  # type: ignore


def patch_dtensor() -> None:
    """
    CRITICAL COMPATIBILITY HACK:
    Latest transformers (for Qwen3.5/Gemma-4) expects torch.distributed.tensor.DTensor.
    """

    class DummyDTensor:
        pass

    try:
        import torch.distributed.tensor as _dt  # noqa: PLC0415

        if not hasattr(_dt, "DTensor"):
            logger.info("Patching torch.distributed.tensor.DTensor")
            _dt.DTensor = DummyDTensor  # type: ignore
    except ImportError:
        logger.info("Mocking torch.distributed.tensor.DTensor")
        mock_dt = ModuleType("torch.distributed.tensor")
        mock_dt.DTensor = DummyDTensor  # type: ignore
        sys.modules["torch.distributed.tensor"] = mock_dt


def patch_all() -> None:
    """Apply all compatibility patches."""
    patch_set_submodule()
    patch_dtensor()
