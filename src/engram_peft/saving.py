# saving.py
import logging
import os
from typing import TYPE_CHECKING, Any, cast

import torch

from engram_peft.utils import (
    as_tensor_dict,
    safe_load,
    safe_load_file,
    safe_save,
    safe_save_file,
)

if TYPE_CHECKING:
    from engram_peft.model import EngramModel


logger = logging.getLogger(__name__)

# Constants for file naming
ADAPTER_SAFE_NAME = "engram_adapters.safetensors"
ADAPTER_LEGACY_NAME = "engram_weights.pt"


def save_pretrained_engram(
    model: "EngramModel",
    save_directory: str,
    safe_serialization: bool = True,
) -> None:
    """
    Saves only the Engram configurations and explicitly the Engram layers' weights.
    This provides a PEFT-like standalone checkpoint for Engram.
    """
    os.makedirs(save_directory, exist_ok=True)

    # 1. Save Engram Config
    model.config.save_pretrained(save_directory)

    # 2. Save Engram Weights
    state_dict = model.engram_layers.state_dict()

    if safe_serialization:
        weights_path = os.path.join(save_directory, ADAPTER_SAFE_NAME)
        # Ensure all tensors are on CPU and contiguous for safetensors
        cpu_state_dict: dict[str, torch.Tensor] = {
            k: v.cpu().contiguous() if torch.is_tensor(v) else v
            for k, v in state_dict.items()
        }
        safe_save_file(cpu_state_dict, weights_path, metadata={"format": "engram-peft"})
        logger.info(f"Saved Engram weights to {weights_path}")
    else:
        weights_path = os.path.join(save_directory, ADAPTER_LEGACY_NAME)
        safe_save(state_dict, weights_path)
        logger.info(f"Saved Engram weights (legacy format) to {weights_path}")


def load_engram_state_dict(
    path_or_dir: str,
) -> dict[str, torch.Tensor]:
    """
    Loads an Engram state dict from a file or directory.
    If a directory is provided, it looks for engram_adapters.safetensors first,
    then engram_weights.pt.
    """
    if os.path.isdir(path_or_dir):
        safe_path = os.path.join(path_or_dir, ADAPTER_SAFE_NAME)
        legacy_path = os.path.join(path_or_dir, ADAPTER_LEGACY_NAME)
        if os.path.exists(safe_path):
            logger.info(f"Loading Engram weights from {safe_path}")
            return safe_load_file(safe_path)
        elif os.path.exists(legacy_path):
            logger.info(f"Loading Engram weights from {legacy_path} (legacy format)")
            return as_tensor_dict(safe_load(legacy_path))
        else:
            raise FileNotFoundError(f"No Engram weight file found in {path_or_dir}")

    # It's a file path
    if path_or_dir.endswith(".safetensors"):
        return safe_load_file(path_or_dir)
    else:
        # Fallback to safe_load (likely .pt)
        return as_tensor_dict(safe_load(path_or_dir))


def load_engram_weights(
    model: "EngramModel",
    engram_path: str,
    strict: bool = False,
) -> None:
    """
    Loads Engram weights from a directory/file into the model.
    """
    try:
        state_dict = load_engram_state_dict(engram_path)
        res = model.engram_layers.load_state_dict(state_dict, strict=strict)
        missing: list[str] = list(cast("list[str]", res.missing_keys))
        unexpected: list[str] = list(cast("list[str]", res.unexpected_keys))
        if missing:
            logger.warning(f"Missing keys during loading: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys during loading: {unexpected}")
    except (FileNotFoundError, RuntimeError) as e:
        logger.warning(f"Failed to load Engram weights: {e}")


def save_pretrained_unified(
    model: "EngramModel",
    save_directory: str,
    safe_serialization: bool = True,
    **kwargs: Any,
) -> None:
    """
    Unified entry point for saving EngramModel.
    Automatically detects and saves base-model adapters (e.g., LoRA) if present.
    """
    os.makedirs(save_directory, exist_ok=True)

    # 1. Identify and Save Base Model Adapters (PEFT Integration)
    # Use feature-detection to avoid hard dependency or circular import issues
    base_model = model.base_model
    # Detect PeftModel by checking for common attributes
    is_peft = getattr(base_model, "_is_peft_model", False) or hasattr(
        base_model, "peft_config"
    )

    if is_peft:
        logger.info("Detecting PeftModel (LoRA). Saving base model adapters...")
        # Direct check for save_pretrained method to be more robust than Protocol check
        save_pretrained_func = getattr(base_model, "save_pretrained", None)
        if callable(save_pretrained_func):
            save_pretrained_func(
                save_directory, safe_serialization=safe_serialization, **kwargs
            )
        else:
            logger.warning(
                "Detected PEFT model but 'save_pretrained' method is not available or callable."
            )

    # 2. Save Engram Artifacts
    save_pretrained_engram(model, save_directory, safe_serialization=safe_serialization)
