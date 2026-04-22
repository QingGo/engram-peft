import logging
import os
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors.torch import load_file, save_file

if TYPE_CHECKING:
    from engram_peft.model import EngramModel

from engram_peft.utils.typing import HFModelProtocol

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
        cpu_state_dict = {
            k: v.cpu().contiguous() if torch.is_tensor(v) else v
            for k, v in state_dict.items()
        }
        save_file(cpu_state_dict, weights_path, metadata={"format": "engram-peft"})
        logger.info(f"Saved Engram weights to {weights_path}")
    else:
        weights_path = os.path.join(save_directory, ADAPTER_LEGACY_NAME)
        torch.save(state_dict, weights_path)
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
            return cast("dict[str, torch.Tensor]", load_file(safe_path, device="cpu"))
        elif os.path.exists(legacy_path):
            logger.info(f"Loading Engram weights from {legacy_path} (legacy format)")
            return cast(
                "dict[str, torch.Tensor]",
                torch.load(legacy_path, map_location="cpu", weights_only=True),
            )
        else:
            raise FileNotFoundError(f"No Engram weight file found in {path_or_dir}")

    # It's a file path
    if path_or_dir.endswith(".safetensors"):
        return cast("dict[str, torch.Tensor]", load_file(path_or_dir, device="cpu"))
    else:
        # Fallback to torch.load (likely .pt)
        return cast(
            "dict[str, torch.Tensor]",
            torch.load(path_or_dir, map_location="cpu", weights_only=True),
        )


def load_engram_weights(
    model: "EngramModel",
    engram_path: str,
) -> None:
    """
    Loads Engram weights from a directory/file into the model.
    """
    try:
        state_dict = load_engram_state_dict(engram_path)
        missing, unexpected = model.engram_layers.load_state_dict(
            state_dict, strict=False
        )
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
    if getattr(model.base_model, "_is_peft_model", False):
        logger.info("Detecting PeftModel (LoRA). Saving base model adapters...")
        if isinstance(model.base_model, HFModelProtocol):
            model.base_model.save_pretrained(
                save_directory, safe_serialization=safe_serialization, **kwargs
            )
        else:
            # Fallback for PeftModel that doesn't strictly match Protocol but has the method
            save_pretrained_func = getattr(model.base_model, "save_pretrained", None)
            if save_pretrained_func:
                save_pretrained_func(
                    save_directory, safe_serialization=safe_serialization, **kwargs
                )

    # 2. Save Engram Artifacts
    save_pretrained_engram(model, save_directory, safe_serialization=safe_serialization)
