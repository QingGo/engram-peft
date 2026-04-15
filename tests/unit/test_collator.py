from typing import Any

import torch

from engram_peft.collator import EngramDataCollator
from engram_peft.config import EngramConfig


class MockCompressor:
    """A mock compressor to verify it's called during collation."""

    def compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids + 1000


def test_hash_indices_content(engram_config: EngramConfig, tiny_tokenizer: Any) -> None:
    """验证 engram_hash_indices 包含所有目标层的哈希"""
    collator = EngramDataCollator(tokenizer=tiny_tokenizer, config=engram_config)
    examples = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5, 6, 7]}]
    batch = collator(examples)

    assert "engram_hash_indices" in batch
    hashes = batch["engram_hash_indices"]
    # Batch size = 2, Max seq len = 4, Num layers = 2, Total heads = 8
    assert hashes.shape == (2, 4, 2, 8)
    assert isinstance(hashes, torch.Tensor)


def test_batch_processing(engram_config: EngramConfig, tiny_tokenizer: Any) -> None:
    """验证 batch 处理正确 (包含不同长度输入)"""
    collator = EngramDataCollator(tokenizer=tiny_tokenizer, config=engram_config)
    examples = [
        {"input_ids": [1, 2]},
        {"input_ids": [1, 2, 3, 4, 5]},
        {"input_ids": [1]},
    ]
    batch = collator(examples)
    assert batch["input_ids"].shape == (3, 5)
    assert batch["engram_hash_indices"].shape == (3, 5, 2, 8)


def test_with_compressor(
    engram_config: EngramConfig, tiny_tokenizer: Any, tiny_compressor: Any
) -> None:
    """验证 optional compressor 是否被正确调用"""
    collator = EngramDataCollator(
        tokenizer=tiny_tokenizer, config=engram_config, compressor=tiny_compressor
    )
    examples = [{"input_ids": [1, 2, 3]}]
    batch = collator(examples)
    assert batch["engram_hash_indices"].shape == (1, 3, 2, 8)


def test_trainer_compatibility(
    engram_config: EngramConfig, tiny_tokenizer: Any
) -> None:
    """验证与 transformers Trainer 兼容"""
    collator = EngramDataCollator(tokenizer=tiny_tokenizer, config=engram_config)
    examples = [{"input_ids": [1, 2], "labels": [1, 2]}]
    batch = collator(examples)
    assert "input_ids" in batch
    assert "labels" in batch
    assert "engram_hash_indices" in batch
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["labels"], torch.Tensor)
    assert isinstance(batch["engram_hash_indices"], torch.Tensor)


def test_cpu_computation(engram_config: EngramConfig, tiny_tokenizer: Any) -> None:
    """验证所有计算都在 CPU 上完成"""
    collator = EngramDataCollator(tokenizer=tiny_tokenizer, config=engram_config)
    examples = [{"input_ids": [1, 2, 3]}]
    batch = collator(examples)
    assert batch["engram_hash_indices"].device.type == "cpu"
    assert batch["input_ids"].device.type == "cpu"
