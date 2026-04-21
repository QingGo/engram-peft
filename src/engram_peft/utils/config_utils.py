import logging
from typing import Any

logger = logging.getLogger(__name__)


def patch_config(config: Any, tokenizer: Any | None = None) -> Any:
    """
    Automated architecture patching for modern Transformer models.
    Handles nested text_configs, missing vocab_size, and pad_token synchronization.
    """
    # 1. Sync attributes from text_config (common in Mistral/Gemma/multimodal models)
    if hasattr(config, "text_config") and config.text_config is not None:
        logger.info(
            "Detected nested 'text_config'. Synchronizing attributes to top-level..."
        )
        text_config_dict = config.text_config.to_dict()
        for attr, value in text_config_dict.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                setattr(config, attr, value)

    # 2. Ensure vocab_size property exists (required by PEFT and Trainer)
    if not hasattr(config, "vocab_size") or config.vocab_size is None:
        if tokenizer is not None:
            config._vocab_size = len(tokenizer)
            logger.info(f"Inferred vocab_size from tokenizer: {config._vocab_size}")
        else:
            logger.warning(
                "vocab_size is missing and no tokenizer provided for inference."
            )

    # 3. Dynamic property injection for PEFT compatibility
    config_class = config.__class__
    if not hasattr(config_class, "vocab_size"):
        logger.info(
            f"Injecting dynamic 'vocab_size' property into {config_class.__name__}"
        )

        def get_vocab_size(self: Any) -> int | None:
            return getattr(self, "_vocab_size", None) or (
                getattr(self.text_config, "vocab_size", None)
                if hasattr(self, "text_config")
                else None
            )

        config_class.vocab_size = property(get_vocab_size)

    # 4. Sync pad_token_id
    if (
        not hasattr(config, "pad_token_id") or config.pad_token_id is None
    ) and tokenizer is not None:
        config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"Synchronized pad_token_id: {config.pad_token_id}")

    return config
