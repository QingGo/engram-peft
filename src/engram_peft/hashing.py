from typing import Dict, List, Tuple, TypedDict

import numpy as np
import torch
from sympy import nextprime  # type: ignore[import-untyped]


class HashParams(TypedDict):
    p: int
    multipliers: torch.Tensor


def calculate_global_primes(
    layer_ids: List[int],
    ngram_sizes: List[int],
    hash_heads: int,
    memory_capacity_per_ngram: List[int],
) -> Dict[int, List[int]]:
    """
    Calculates unique prime table sizes for all layers and heads.
    Matches the official demo's calculate_vocab_size_across_layers logic.

    Args:
        layer_ids: List of all layers where Engram is applied.
        ngram_sizes: List of N-gram orders (e.g. [2, 3]).
        hash_heads: Number of hash heads per N-gram order.
        memory_capacity_per_ngram: Target base capacity for each N-gram order.

    Returns:
        A dictionary mapping layer_id to a flat list of primes for all its heads.
    """
    seen_primes = set()
    primes_across_layers = {}

    for layer_id in sorted(layer_ids):
        all_layer_primes = []
        for i, _ in enumerate(ngram_sizes):
            # For each ngram order, start from its specific base capacity
            base_vocab_size = memory_capacity_per_ngram[i]
            current_prime_search_start = base_vocab_size - 1

            for _ in range(hash_heads):
                p_val = nextprime(current_prime_search_start)
                if p_val is None:
                    p = current_prime_search_start + 1
                else:
                    p = int(p_val)

                # Ensure global uniqueness across layers and heads
                while p in seen_primes:
                    p_val = nextprime(p)
                    if p_val is None:
                        p += 1
                    else:
                        p = int(p_val)

                seen_primes.add(p)
                all_layer_primes.append(p)
                # Next head for the SAME ngram in SAME layer starts searching from current prime
                current_prime_search_start = p

        primes_across_layers[layer_id] = all_layer_primes

    return primes_across_layers


class MultiHeadHash:
    """
    Implements Multi-Head Hashing as described in the paper section 2.2.
    For each N-gram order n, we employ K distinct hash heads.
    Aligned with the official Engram demo implementation.
    """

    def __init__(
        self,
        layer_id: int,
        primes: List[int],
        ngram_sizes: List[int] = [2, 3],
        hash_heads: int = 8,
        seed: int = 42,
        tokenizer_vocab_size: int = 129280,
    ):
        """
        Initialize the MultiHeadHash with pre-calculated primes.

        Args:
            layer_id: The ID of the current layer.
            primes: List of pre-calculated primes for this layer (all heads).
            ngram_sizes: List of N-gram orders.
            hash_heads: Number of hash heads per N-gram order.
            seed: Random seed for generating hash parameters.
            tokenizer_vocab_size: Size of the compressed vocabulary.
        """
        self.layer_id = layer_id
        self.ngram_sizes = ngram_sizes
        self.hash_heads = hash_heads
        self.primes = primes
        self.seed = seed
        self.tokenizer_vocab_size = tokenizer_vocab_size

        # Generate layer-specific multipliers
        PRIME_1 = 10007
        base_seed = int(seed + PRIME_1 * int(layer_id))
        g = np.random.default_rng(base_seed)

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)

        max_ngram = max(ngram_sizes)
        r = g.integers(low=0, high=half_bound, size=(max_ngram,), dtype=np.int64)
        self.all_multipliers = torch.from_numpy(r * 2 + 1)

        self.hash_params: Dict[Tuple[int, int], HashParams] = {}
        prime_idx = 0
        for n in ngram_sizes:
            for k in range(hash_heads):
                p = self.primes[prime_idx]
                self.hash_params[(n, k)] = {
                    "p": p,
                    "multipliers": self.all_multipliers[:n],
                }
                prime_idx += 1

    def compute_hashes(
        self, compressed_token_ids: torch.Tensor
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Computes the hash indices using Multiplicative-XOR logic.
        """
        batch_size, seq_len = compressed_token_ids.shape
        device = compressed_token_ids.device
        results: Dict[Tuple[int, int], torch.Tensor] = {}

        max_n = max(self.ngram_sizes)
        padding = torch.zeros((batch_size, max_n - 1), dtype=torch.long, device=device)
        padded_tokens = torch.cat([padding, compressed_token_ids], dim=1)

        for n in self.ngram_sizes:
            ngrams = padded_tokens[:, : seq_len + n - 1].unfold(1, n, 1)
            m = self.all_multipliers[:n].to(device)
            weighted_tokens = ngrams * m

            mix = weighted_tokens[..., 0]
            for i in range(1, n):
                mix = torch.bitwise_xor(mix, weighted_tokens[..., i])

            for k in range(self.hash_heads):
                p = self.hash_params[(n, k)]["p"]
                h = (mix % p + p) % p
                results[(n, k)] = h.long()

        return results
