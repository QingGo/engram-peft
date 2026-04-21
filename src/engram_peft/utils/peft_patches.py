import logging
from typing import Any

import peft.tuners.lora.model as lora_model

logger = logging.getLogger(__name__)


def apply_peft_patches() -> None:
    """
    Apply deep monkey patches to PEFT to support custom architectures like Gemma-4.
    """
    try:
        # Check if already patched
        if hasattr(lora_model.LoraModel, "_is_engram_patched"):
            return

        original_create_new_module = lora_model.LoraModel._create_new_module

        # We use a wrapper that matches the signature of the original staticmethod
        def patched_create_new_module(*args: Any, **kwargs: Any) -> Any:
            target = None
            if len(args) >= 3:
                target = args[2]
            elif "target" in kwargs:
                target = kwargs["target"]

            if target is not None:
                target_type_name = target.__class__.__name__
                if target_type_name == "Gemma4ClippableLinear" and hasattr(
                    target, "linear"
                ):
                    logger.info(
                        f"Intercepting PEFT injection for {target_type_name}, redirecting to .linear"
                    )
                    if len(args) >= 3:
                        new_args = list(args)
                        new_args[2] = target.linear
                        return original_create_new_module(*new_args, **kwargs)
                    else:
                        kwargs["target"] = target.linear
                        return original_create_new_module(*args, **kwargs)

            return original_create_new_module(*args, **kwargs)

        # In PEFT, _create_new_module is a staticmethod, but we can replace it
        # with our function and wrap it back in staticmethod()
        lora_model.LoraModel._create_new_module = staticmethod(
            patched_create_new_module
        )
        lora_model.LoraModel._is_engram_patched = True
        logger.info("Successfully applied PEFT deep patches for custom layers.")

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to apply PEFT patches: {e}")
