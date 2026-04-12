from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EngramConfig:
    """
    Configuration for Engram-PEFT.
    """

    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3])
    hash_heads: int = 8
    # 词表容量按 N-gram 阶数设置，值默认取 V3 词表的五倍 (如 [129280*5, 129280*5])
    memory_capacity_per_ngram: List[int] = field(
        default_factory=lambda: [129280 * 5, 129280 * 5]
    )
    embedding_dim_per_head: int = 128
    hidden_dim: int = 2560
    num_branches: int = 1
    kernel_size: int = 4
    dilation: Optional[int] = None
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    seed: int = 42

    def __post_init__(self) -> None:
        """Set default values if not provided."""
        if self.dilation is None:
            self.dilation = max(self.ngram_sizes) if self.ngram_sizes else 1

        # Ensure capacity list matches ngram_sizes count
        if len(self.memory_capacity_per_ngram) != len(self.ngram_sizes):
            # Fallback to default if mismatch
            default_cap = 129280 * 5
            self.memory_capacity_per_ngram = [default_cap] * len(self.ngram_sizes)

    @property
    def total_embedding_dim(self) -> int:
        """Total dimension of the concatenated Engram embeddings (e_t)."""
        return len(self.ngram_sizes) * self.hash_heads * self.embedding_dim_per_head
