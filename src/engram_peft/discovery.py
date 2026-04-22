import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from engram_peft.protocols import HFConfigProtocol, HFModelProtocol

logger = logging.getLogger(__name__)

# Map HF model_type to standard layer container paths
# Constant moved from model.py for centralization
ARCH_LAYER_MAPPING = {
    "llama": ["model.layers", "layers"],
    "qwen2": ["model.layers", "layers"],
    "mistral": ["model.layers", "layers"],
    "mixtral": ["model.layers", "layers"],
    "deepseek_v2": ["model.layers", "layers"],
    "deepseek_v3": ["model.layers", "layers"],
    "gemma": ["model.layers", "layers"],
    "gemma2": ["model.layers", "layers"],
    "chatglm": ["transformer.encoder.layers", "encoder.layers"],
    "glm": ["transformer.encoder.layers", "encoder.layers"],
    "bert": ["bert.encoder.layer", "encoder.layer"],
    "longformer": ["longformer.encoder.layer", "encoder.layer"],
    "roberta": ["roberta.encoder.layer", "encoder.layer"],
    "gpt2": ["transformer.h", "h"],
    "gpt_neox": ["gpt_neox.layers", "layers"],
    "phi": ["model.layers", "layers"],
    "phi3": ["model.layers", "layers"],
    # Multimodal variants (e.g. Pixtral, Mistral-VL, Qwen-VL)
    "mistral_vl": ["language_model.model.layers", "language_model.layers"],
    "qwen2_vl": ["visual.blocks", "model.layers"],
    "qwen3": ["model.layers", "layers"],
    "qwen3_vl": ["model.layers", "language_model.model.layers"],
    "llava": ["language_model.model.layers", "model.layers"],
}


@dataclass
class ResolvedMetadata:
    """Container for auto-detected model metadata."""

    hidden_size: int
    original_vocab_size: int
    pad_token_id: int
    layer_container_path: str
    tokenizer_name_or_path: str | None = None
    model_type: str | None = None


class ArchitectureResolver:
    """
    Centralized utility for discovering model architecture details and metadata.
    Provides best-effort detection with transparent logging.
    """

    @staticmethod
    def resolve(
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
        config: Any | None = None,
    ) -> ResolvedMetadata:
        """
        Performs a full discovery pass on the model and environment.

        Args:
            model: The base model to analyze.
            tokenizer: Optional tokenizer instance.
            config: Optional EngramConfig or model config.

        Returns:
            ResolvedMetadata: The detected parameters.
        """
        logger.info("[Engram-PEFT] Starting best-effort architecture discovery...")

        # 1. Resolve Hidden Size
        hidden_size, h_source = ArchitectureResolver._resolve_hidden_size(model, config)
        logger.info(
            f"[Engram-PEFT] Determined hidden_size={hidden_size} (source: {h_source})"
        )

        # 2. Resolve Original Vocabulary Size
        orig_vocab_size, v_source = ArchitectureResolver._resolve_original_vocab_size(
            model, tokenizer, config
        )
        logger.info(
            f"[Engram-PEFT] Determined original_vocab_size={orig_vocab_size} (source: {v_source})"
        )

        # 3. Resolve Pad Token ID
        pad_id, p_source = ArchitectureResolver._resolve_pad_id(
            model, tokenizer, config
        )
        logger.info(f"[Engram-PEFT] Determined pad_id={pad_id} (source: {p_source})")

        # 4. Resolve Layer Container Path
        layer_path, l_source = ArchitectureResolver._resolve_layer_container(
            model, config
        )
        logger.info(
            f"[Engram-PEFT] Determined layer_container_path='{layer_path}' (source: {l_source})"
        )

        model_type = ArchitectureResolver._get_model_type(model)

        return ResolvedMetadata(
            hidden_size=hidden_size,
            original_vocab_size=orig_vocab_size,
            pad_token_id=pad_id,
            layer_container_path=layer_path,
            tokenizer_name_or_path=getattr(config, "tokenizer_name_or_path", None),
            model_type=model_type,
        )

    @staticmethod
    def _resolve_hidden_size(model: nn.Module, config: Any | None) -> tuple[int, str]:
        # 1. explicit config
        if config is not None:
            val = getattr(config, "hidden_size", None)
            if val is not None:
                return int(val), "EngramConfig.hidden_size"

        # 2. model config
        base_config = None
        if isinstance(model, HFModelProtocol):
            base_config = model.config
        else:
            base_config = getattr(model, "config", None)

        if base_config is not None:
            if isinstance(base_config, HFConfigProtocol):
                return base_config.hidden_size, "model.config.hidden_size"

            # Check top level first for non-standard configs
            for attr in [
                "hidden_size",
                "d_model",
                "dim",
                "n_embd",
                "word_embed_proj_dim",
            ]:
                val = getattr(base_config, attr, None)
                if val is not None:
                    return int(val), f"model.config.{attr}"

            # Check nested text_config (common in multimodal models)
            text_config = getattr(base_config, "text_config", None)
            if text_config is not None:
                if isinstance(text_config, HFConfigProtocol):
                    return (
                        text_config.hidden_size,
                        "model.config.text_config.hidden_size",
                    )

                for attr in ["hidden_size", "d_model", "dim", "n_embd"]:
                    val = getattr(text_config, attr, None)
                    if val is not None:
                        return int(val), f"model.config.text_config.{attr}"

        # 3. Direct attribute
        for attr in ["hidden_size", "d_model"]:
            val = getattr(model, attr, None)
            if val is not None:
                return int(val), f"model.{attr}"

        # 4. Ultimate fallback (with warning)
        logger.warning(
            "[Engram-PEFT] Could not detect hidden_size. Using default 2048."
        )
        return 2048, "Default fallback (2048)"

    @staticmethod
    def _resolve_original_vocab_size(
        model: nn.Module, tokenizer: Any | None, config: Any | None
    ) -> tuple[int, str]:
        # 1. explicit config
        if config is not None:
            val = getattr(config, "original_vocab_size", None)
            if val is not None:
                return int(val), "EngramConfig.original_vocab_size"

        # 2. Tokenizer
        if tokenizer is not None:
            return len(tokenizer), "tokenizer.vocab_size"

        # 3. model config
        base_config = None
        if isinstance(model, HFModelProtocol):
            base_config = model.config
        else:
            base_config = getattr(model, "config", None)

        if base_config is not None:
            if isinstance(base_config, HFConfigProtocol):
                return base_config.vocab_size, "model.config.vocab_size"

            # Check top level
            val = getattr(base_config, "vocab_size", None)
            if val is not None:
                return int(val), "model.config.vocab_size"

            # Check nested text_config
            text_config = getattr(base_config, "text_config", None)
            if text_config is not None:
                if isinstance(text_config, HFConfigProtocol):
                    return text_config.vocab_size, "model.config.text_config.vocab_size"

                val = getattr(text_config, "vocab_size", None)
                if val is not None:
                    return int(val), "model.config.text_config.vocab_size"

        raise ValueError(
            "Could not detect original vocab_size. Please provide a tokenizer or set original_vocab_size in EngramConfig."
        )

    @staticmethod
    def _resolve_pad_id(
        model: nn.Module, tokenizer: Any | None, config: Any | None
    ) -> tuple[int, str]:
        # 1. explicit config
        if config is not None:
            val = getattr(config, "pad_id", None)
            if val is not None:
                return int(val), "EngramConfig.pad_id"

        # 2. Tokenizer
        if (
            tokenizer is not None
            and getattr(tokenizer, "pad_token_id", None) is not None
        ):
            return int(tokenizer.pad_token_id), "tokenizer.pad_token_id"

        # 3. model config
        base_config = None
        if isinstance(model, HFModelProtocol):
            base_config = model.config
        else:
            base_config = getattr(model, "config", None)

        if base_config is not None:
            if isinstance(base_config, HFConfigProtocol):
                if base_config.pad_token_id is not None:
                    return int(base_config.pad_token_id), "model.config.pad_token_id"

            # Check top level
            val = getattr(base_config, "pad_token_id", None)
            if val is not None:
                return int(val), "model.config.pad_token_id"

            # Check nested text_config
            text_config = getattr(base_config, "text_config", None)
            if text_config is not None:
                if isinstance(text_config, HFConfigProtocol):
                    if text_config.pad_token_id is not None:
                        return (
                            int(text_config.pad_token_id),
                            "model.config.text_config.pad_token_id",
                        )

                val = getattr(text_config, "pad_token_id", None)
                if val is not None:
                    return int(val), "model.config.text_config.pad_token_id"

        # 4. Absolute fallback for common models if everything fails
        logger.warning("[Engram-PEFT] Could not detect pad_id. Using default 2.")
        return 2, "Default fallback (2)"

    @staticmethod
    def _resolve_layer_container(
        model: nn.Module, config: Any | None
    ) -> tuple[str, str]:
        # 1. Explicit path
        if (
            config is not None
            and getattr(config, "layer_container_path", None) is not None
        ):
            path = config.layer_container_path
            try:
                container = ArchitectureResolver.get_submodule_by_path(model, path)
                if isinstance(container, nn.ModuleList):
                    return path, "EngramConfig.layer_container_path"

                raise ValueError(
                    f"Explicit layer_container_path '{path}' exists but is not a nn.ModuleList "
                    f"(found {type(container)})."
                ) from None
            except AttributeError:
                raise ValueError(
                    f"Explicit layer_container_path '{path}' not found in model."
                ) from None
            except Exception as e:
                # Re-raise if it's already a ValueError from above, otherwise wrap
                if isinstance(e, ValueError):
                    raise e
                raise ValueError(
                    f"Error resolving explicit layer_container_path '{path}': {e}"
                ) from e

        # 2. Registry
        model_type = ArchitectureResolver._get_model_type(model)
        if model_type in ARCH_LAYER_MAPPING:
            paths = ARCH_LAYER_MAPPING[model_type]
            for path in paths:
                try:
                    container = ArchitectureResolver.get_submodule_by_path(model, path)
                    if isinstance(container, nn.ModuleList):
                        return path, f"Architecture Registry ({model_type})"
                except (AttributeError, ValueError):
                    continue

        # 3. Heuristic
        path = ArchitectureResolver.find_largest_module_list(model)
        if path:
            return path, "Heuristic scanner"

        raise ValueError(
            "Could not find transformer layers container (nn.ModuleList). "
            "Please specify 'layer_container_path' in EngramConfig."
        )

    @staticmethod
    def _get_model_type(model: nn.Module) -> str | None:
        base_config = None
        if isinstance(model, HFModelProtocol):
            base_config = model.config
        else:
            base_config = getattr(model, "config", None)

        if base_config is not None:
            if isinstance(base_config, HFConfigProtocol):
                return base_config.model_type
            return getattr(base_config, "model_type", None)
        return None

    @staticmethod
    def get_submodule_by_path(model: nn.Module, path: str) -> nn.Module:
        """
        Returns the submodule at the given dot-separated path.
        Automatically traverses through common wrappers like PeftModel or EngramModel
        if an attribute is missing on the wrapper but present in the base_model.
        """
        if not path:
            return model
        segments = path.split(".")
        curr: Any = model
        for seg in segments:
            if hasattr(curr, seg):
                curr = getattr(curr, seg)
            else:
                # Check for PEFT/Engram wrappers (nominal and structural)
                base_model = getattr(curr, "base_model", None)
                if isinstance(base_model, nn.Module) and hasattr(base_model, seg):
                    curr = getattr(base_model, seg)
                    continue

                raise AttributeError(
                    f"Module {type(curr).__name__} has no attribute {seg}. "
                    "Traversal failed at this segment. If this is a wrapped model, "
                    "ensure the path reflects the wrapped structure or update ArchitectureResolver."
                )
        if not isinstance(curr, nn.Module):
            raise TypeError(
                f"Path '{path}' did not resolve to an nn.Module (found {type(curr)})."
            )
        return curr

    @staticmethod
    def resolve_layer_dtype(
        target_module: nn.Module, config: Any | None = None
    ) -> tuple[torch.dtype, str]:
        """
        Determines the appropriate compute dtype for a given module.
        Priority: Explicit config > compute_dtype (BNB) > parameter sample.

        Returns:
            tuple[torch.dtype, str]: The detected dtype and a human-readable source string.
        """
        # 1. Explicit Config Override
        if config is not None:
            explicit_dtype_str = getattr(config, "engram_dtype", None)
            if explicit_dtype_str is not None:
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                target_dtype = dtype_map.get(explicit_dtype_str)
                if target_dtype is not None:
                    return (
                        target_dtype,
                        f"Explicit config (engram_dtype='{explicit_dtype_str}')",
                    )

        # 2. BitsAndBytes compute_dtype detection
        if hasattr(target_module, "compute_dtype"):
            val = target_module.compute_dtype
            if isinstance(val, torch.dtype):
                return (
                    val,
                    f"Quantized compute_dtype ({target_module.__class__.__name__})",
                )

        # 3. Parameter Sampling
        try:
            example_param = next(target_module.parameters())
            sample_dtype = example_param.dtype

            # If the sampled dtype is floating point, we trust it
            if sample_dtype.is_floating_point:
                return sample_dtype, f"Parameter sample ({sample_dtype})"

            # If it's an integer type (e.g. uint8 in 4-bit models), we fallback
            return (
                torch.float32,
                f"Fallback from non-floating-point sample ({sample_dtype})",
            )
        except (StopIteration, AttributeError):
            # No parameters found or attribute error
            return torch.float32, "Default fallback (no parameters found)"

    @staticmethod
    def find_largest_module_list(model: nn.Module) -> str | None:
        """Heuristically finds the largest nn.ModuleList in the model tree."""
        candidates: list[tuple[str, int]] = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                if all(isinstance(m, torch.nn.Module) for m in module):
                    candidates.append((name, len(module)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)
        return candidates[0][0]
