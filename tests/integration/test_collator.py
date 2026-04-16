from typing import Any

from engram_peft.collator import EngramDataCollator
from engram_peft.config import EngramConfig


def test_collator_gpt2_integration(tokenizer_gpt2: Any) -> None:
    """Integration Test: Verify collator with real GPT2 tokenizer (heavy vocab)."""
    # Using the standard large config that triggers prime calculations for 1.1M vocab

    config = EngramConfig(
        target_layers=[2, 15],
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        # Default capacity is large (~1.1M per ngram)
        seed=42,
        compressed_vocab_size=50257,  # GPT2 vocab size
        pad_id=tokenizer_gpt2.eos_token_id,
    )

    collator = EngramDataCollator(tokenizer=tokenizer_gpt2, config=config)
    examples = [{"input_ids": [1, 2, 3, 4, 5]}]
    batch = collator(examples)

    assert "engram_hash_indices" in batch
    # Batch size=1, SeqLen=5, Layers=2, Heads=16 (2 ngram * 8 heads)
    assert batch["engram_hash_indices"].shape == (1, 5, 2, 16)
