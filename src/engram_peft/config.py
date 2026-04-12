from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EngramConfig:
    """
    Configuration for Engram-PEFT.
    """

    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    hash_heads: int = 8
    memory_capacity_per_head: Optional[List[int]] = None
    embedding_dim_per_head: int = 128
    hidden_dim: int = 2560
    num_branches: int = 1
    kernel_size: int = 4
    dilation: Optional[int] = None
    seed: int = 42

    @property
    def total_embedding_dim(self) -> int:
        """Total dimension of the concatenated Engram embeddings (e_t)."""
        return len(self.ngram_sizes) * self.hash_heads * self.embedding_dim_per_head
