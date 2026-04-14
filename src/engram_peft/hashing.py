from dataclasses import dataclass, field
from typing import Dict, List, Set, Union

import numpy as np
import torch
from sympy import nextprime  # type: ignore[import-untyped]


@dataclass
class NgramHashMapping:
    """
    Implements Multi-Head Hashing as described in the Engram paper section 2.2.
    Aligned with the official Engram demo implementation.
    """

    engram_vocab_size_per_ngram: List[int] = field(
        default_factory=lambda: [2262400 // 2, 2262400 // 2]
    )
    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3])
    max_ngram_size: int = 3
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [2, 15])
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    pad_id: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        self.max_ngram_size = max(self.ngram_sizes)
        self.all_multipliers: Dict[int, np.ndarray] = {}

        for layer_id in self.layer_ids:
            # Layer-specific seed
            PRIME_1 = 10007
            base_seed = int(self.seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)

            # Follow official demo's heuristic bound based on tokenizer_vocab_size
            tokenizer_vocab_size = 129280
            max_long = np.iinfo(np.int64).max
            M_max = int(max_long // tokenizer_vocab_size)
            half_bound = max(1, M_max // 2)

            # Generate max_ngram_size multipliers for this layer
            r = g.integers(
                low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64
            )
            # Must be odd numbers
            self.all_multipliers[layer_id] = r * 2 + 1

        self.prime_tables = self.calculate_vocab_size_across_layers()

    def find_next_prime(self, start: int, seen_primes: Set[int]) -> int:
        """Finds the next unused global prime number strictly greater than start."""
        p_val = nextprime(start)
        p = start + 1 if p_val is None else int(p_val)
        while p in seen_primes:
            p_val = nextprime(p)
            p = p + 1 if p_val is None else int(p_val)
        return p

    def calculate_vocab_size_across_layers(self) -> Dict[int, List[List[int]]]:
        """
        Calculates unique prime table sizes for all layers and heads.
        Matches the official demo's globally unique prime generation.
        Returns:
            Dict mapping layer_id -> List of Lists of primes [ngram_idx][head_idx]
        """
        seen_primes: Set[int] = set()
        primes_across_layers: Dict[int, List[List[int]]] = {}

        for layer_id in sorted(self.layer_ids):
            layer_primes = []
            for i in range(len(self.ngram_sizes)):
                head_primes = []
                # Distribute the total bucket capacity among the heads
                base_vocab_size = (
                    self.engram_vocab_size_per_ngram[i] // self.n_head_per_ngram
                )
                current_start = base_vocab_size - 1

                for _ in range(self.n_head_per_ngram):
                    p = self.find_next_prime(current_start, seen_primes)
                    seen_primes.add(p)
                    head_primes.append(p)
                    current_start = p

                layer_primes.append(head_primes)
            primes_across_layers[layer_id] = layer_primes

        return primes_across_layers

    def _get_ngram_indices(self, input_ids: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Internal implementation of multi-head hashing using NumPy vectorization.
        Extracts ngrams once and broadcasts multipliers and primes across layers and heads.
        """
        batch_size, seq_len = input_ids.shape
        max_n = self.max_ngram_size

        # 1. Pad with pad_id for left context
        padding = np.full((batch_size, max_n - 1), self.pad_id, dtype=np.int64)
        padded_tokens = np.concatenate([padding, input_ids], axis=1).astype(np.int64)

        # 2. Extract sliding windows for all unique n-gram sizes
        # ngrams_cache: Dict[n_size, np.ndarray] of shape [batch_size, seq_len, n_size]
        ngrams_cache = {}
        for n in set(self.ngram_sizes):
            view_shape = (batch_size, seq_len, n)
            strides = padded_tokens.strides + (padded_tokens.strides[-1],)
            ngrams_cache[n] = np.lib.stride_tricks.as_strided(
                padded_tokens, shape=view_shape, strides=strides
            )

        # 3. Pre-process multipliers and primes for efficient broadcasting
        # We process layer by layer but vectorize across heads.
        # Layer vectorization is possible but often memory-intensive if batch_size is large.
        # Head vectorization is the most critical for Engram (8x speedup).
        layer_results: Dict[int, np.ndarray] = {}

        for layer_id in self.layer_ids:
            all_head_hashes: List[np.ndarray] = []
            multipliers = self.all_multipliers[layer_id]
            layer_primes = self.prime_tables[layer_id]

            for i, n in enumerate(self.ngram_sizes):
                # shape: [B, L, n]
                ngrams = ngrams_cache[n]
                m = multipliers[:n]

                # weighted: [B, L, n]
                weighted = ngrams * m

                # mix: [B, L]
                # Using reduce for bitwise_xor for better performance
                mix = np.bitwise_xor.reduce(weighted, axis=-1)

                # Vectorize across all heads for this ngram size
                # primes: [n_head_per_ngram]
                primes = np.array(layer_primes[i], dtype=np.int64)

                # Broadcasting mix [B, L] against primes [H] -> [B, L, H]
                # h = (mix % p + p) % p
                h = np.mod(np.mod(mix[..., np.newaxis], primes) + primes, primes)
                all_head_hashes.append(h)

            # Stack all n-gram heads along the last dimension
            # Result: [B, L, total_heads]
            layer_results[layer_id] = np.concatenate(all_head_hashes, axis=-1)

        return layer_results

    def hash(self, input_ids: Union[torch.Tensor, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Compute hashes for all tracked layers using vectorized operations.

        Args:
            input_ids: Original compressed ids as torch Tensor or numpy Array.

        Returns:
            Dict mapping layer_id -> np.ndarray of hash indices [B, L, total_heads].
        """
        if isinstance(input_ids, torch.Tensor):
            arr = input_ids.cpu().numpy()
        else:
            arr = np.array(input_ids)

        return self._get_ngram_indices(arr)
