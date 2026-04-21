from typing import Any, cast

import torch
from transformers import DataCollatorForLanguageModeling


class SmartDataCollator(DataCollatorForLanguageModeling):
    """
    Standardized data collator that ensures padding tokens are masked in labels.
    """

    def __call__(
        self, examples: list[dict[str, Any]], return_tensors: str | None = None
    ) -> dict[str, torch.Tensor]:
        batch = super().__call__(examples, return_tensors=return_tensors)

        # Ensure padding is masked (DataCollatorForLanguageModeling with mlm=False
        # already does some of this, but we want to be explicit)
        if "labels" in batch and self.tokenizer.pad_token_id is not None:
            # Mask padding
            batch["labels"][batch["input_ids"] == self.tokenizer.pad_token_id] = -100

        return cast("dict[str, torch.Tensor]", batch)


def get_dataset_template(model_type: str) -> str:
    """
    Returns a prompt template for the given model architecture.
    """
    templates = {
        "gemma": (
            "<start_of_turn>user\n{instruction}\n{input}<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        "mistral": "<s>[INST] {instruction} {input} [/INST]",
        "qwen": "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n",
        "llama": "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{instruction} {input} [/INST]",
    }

    # Try to fuzzy match if not exact
    for key, value in templates.items():
        if key in model_type.lower():
            return value

    return "{instruction} {input}"
