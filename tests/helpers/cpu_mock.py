from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import patch

import torch


@contextmanager
def meta_device_context() -> Generator[None, None, None]:
    """
    Context manager to simulate a CPU/Meta-device environment for GPU-free verification.
    Mocks torch.cuda.is_available to False and provides helpers for meta-device loading.
    """
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.cuda.device_count", return_value=0):
            yield


def force_meta_device(model_config: torch.nn.Module) -> torch.nn.Module:
    """Helper to load a model structure onto the meta device."""
    # This is a conceptual helper - in practice, users wrap their model loading
    # in 'with torch.device("meta"):' context.
    return model_config
