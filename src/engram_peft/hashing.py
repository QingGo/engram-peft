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
    hashing_mode: str = "left"
    stop_token_ids: List[int] = field(default_factory=list)
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
        pad_left, pad_right = 0, 0

        # 1. Padding tokens according to hashing_mode
        if self.hashing_mode == "centered":
            # For centered hashing of max_n tokens:
            # Shift tokens so that index t is in the middle of the window.
            # Example n=3: window is [t-1, t, t+1].
            # This requires floor(max_n/2) left padding and ceil(max_n/2)-1 right padding.
            pad_left = max_n // 2
            pad_right = (max_n - 1) - pad_left
            left_padding = np.full((batch_size, pad_left), self.pad_id, dtype=np.int64)
            right_padding = np.full(
                (batch_size, pad_right), self.pad_id, dtype=np.int64
            )
            padded_tokens = np.concatenate(
                [left_padding, input_ids, right_padding], axis=1
            ).astype(np.int64)
        else:
            # Default: left-context (suffix) padding
            padding = np.full((batch_size, max_n - 1), self.pad_id, dtype=np.int64)
            padded_tokens = np.concatenate([padding, input_ids], axis=1).astype(
                np.int64
            )

        # 2. Extract sliding windows for all unique n-gram sizes
        ngrams_cache = {}
        stop_mask_cache = {}
        for n in set(self.ngram_sizes):
            if self.hashing_mode == "centered":
                # For a specific n, we might need a different offset than max_n.
                # Window for t should be [t - floor(n/2), ..., t + ceil(n/2) - 1].
                # Calculate start offset in the padded_tokens.
                # padded_tokens[t+pad_left] was originally input_ids[t].
                # We want a window of size n starting at t+pad_left - n//2.
                start_offset = pad_left - (n // 2)
                end_offset = start_offset + seq_len
                # Use as_strided on the relevant slice
                view_shape = (batch_size, seq_len, n)
                strides = padded_tokens.strides + (padded_tokens.strides[-1],)
                tokens_slice = padded_tokens[:, start_offset : end_offset + n - 1]
                ngrams_cache[n] = np.lib.stride_tricks.as_strided(
                    tokens_slice, shape=view_shape, strides=strides
                )
            else:
                # Default suffix extraction
                view_shape = (batch_size, seq_len, n)
                strides = padded_tokens.strides + (padded_tokens.strides[-1],)
                tokens_slice = padded_tokens[:, max_n - n : seq_len + max_n - 1]
                ngrams_cache[n] = np.lib.stride_tricks.as_strided(
                    tokens_slice, shape=view_shape, strides=strides
                )

            # 3. Handle stop_token_ids boundary check
            if self.stop_token_ids:
                # Create a mask where any token in the n-gram is a stop token
                is_stop = np.isin(ngrams_cache[n], self.stop_token_ids)
                stop_mask_cache[n] = np.any(is_stop, axis=-1)

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

                # Apply stop_mask if present
                if n in stop_mask_cache:
                    # Set hash to a "virtual" invalid bucket if boundary crossed.
                    # We can use a deterministic but unlikely value, or just 0 (as masking).
                    # Using 0 is fine because the hashing logic handles it as a valid bucket,
                    # but if it matches many "invalid" n-grams, it effectively acts as a
                    # shared "null memory".
                    h[stop_mask_cache[n]] = 0

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
