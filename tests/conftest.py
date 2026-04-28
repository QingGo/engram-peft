from collections.abc import Iterable, Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn


class MockTokenizer:
    pad_token_id: int
    eos_token_id: int
    tokenizer_vocab_size: int
    vocab: dict[str, int]

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.tokenizer_vocab_size = 128  # reduced for fast unit testing (was 129280)
        self.vocab = {f"token_{i}": i for i in range(10)}

    @property
    def vocab_size(self) -> int:
        return self.tokenizer_vocab_size

    def __len__(self) -> int:
        return 10

    def __call__(self, text: str | list[str] | Any, **kwargs: Any) -> Any:
        # Mock call for unit tests
        import torch

        if isinstance(text, str):
            ids = [self.vocab.get(text, 0)]
        else:
            ids = [0]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([[1]])}

    def encode(self, text: str | list[str] | Any, **kwargs: Any) -> list[int]:
        return [0]

    def decode(
        self,
        token_ids: list[int] | Any,
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join([f"token_{t}" for t in token_ids])

    def convert_ids_to_tokens(self, tid: int) -> str:
        return f"token_{tid}"

    def pad(
        self, examples: list[dict[str, Any]], return_tensors: str = "pt", **kwargs: Any
    ) -> dict[str, Any]:
        """Basic padding simulation for unit tests without top-level torch."""
        import torch

        # Basic padding simulation for unit tests
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
        max_len = max(len(x) for x in input_ids)
        padded_ids = []
        for x in input_ids:
            p = torch.cat(
                [
                    x,
                    torch.full(
                        (max_len - len(x),), self.pad_token_id, dtype=torch.long
                    ),
                ]
            )
            padded_ids.append(p)

        batch = {"input_ids": torch.stack(padded_ids)}
        if "labels" in examples[0]:
            batch["labels"] = batch["input_ids"].clone()
        return batch


@pytest.fixture(scope="session")
def tokenizer_gpt2() -> Any:
    """Session-scoped GPT2 tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="session")
def tiny_tokenizer() -> MockTokenizer:
    """A minimal mock tokenizer with only 10 tokens for fast unit testing."""
    return MockTokenizer()


@pytest.fixture(scope="session")
def tiny_compressor(tiny_tokenizer: MockTokenizer) -> Any:
    """A CompressedTokenizer using the tiny mock tokenizer (fast)."""
    from engram_peft.compression import CompressedTokenizer

    # Create an actual CompressedTokenizer but with our mock tokenizer to skip loops
    return CompressedTokenizer("mock/tiny", tokenizer=tiny_tokenizer)


@pytest.fixture
def engram_config() -> Any:
    """Standard EngramConfig fixture."""
    from engram_peft.config import EngramConfig

    return EngramConfig(
        target_layers=[1, 14],
        ngram_sizes=[2, 3],
        n_head_per_ngram=4,
        engram_vocab_size_per_ngram=[100, 100],
        seed=42,
        hidden_size=32,
        compressed_vocab_size=129280,
        pad_id=0,
    )
