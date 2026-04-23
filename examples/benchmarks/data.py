# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
import copy
from typing import Any

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    subset_size: int,
    eval_size: int,
    max_length: int,
    num_proc: int = 4,
) -> tuple[Any, Any]:
    """
    Standardizes dataset preparation for TinyStories.
    """
    print(f"Loading TinyStories dataset (subset={subset_size})...")
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=False)

    # Deterministic selection
    train_ds = train_ds.select(range(subset_size))
    val_ds = val_ds.select(range(min(len(val_ds), eval_size)))

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized_dict = dict(tokenized)
        tokenized_dict["labels"] = copy.deepcopy(tokenized_dict["input_ids"])
        return tokenized_dict

    print(f"Tokenizing with {num_proc} processes...")
    train_dataset = train_ds.map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=num_proc
    )
    eval_dataset = val_ds.map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=num_proc
    )

    return train_dataset, eval_dataset
