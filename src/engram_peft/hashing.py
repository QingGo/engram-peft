import random
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torch
from sympy import isprime, nextprime  # type: ignore[import-untyped]


class HashParams(TypedDict):
    p: int
    multipliers: torch.Tensor


class MultiHeadHash:
    """
    Implements Multi-Head Hashing as described in the paper section 2.2.
    For each N-gram order n, we employ K distinct hash heads.
    Aligned with the official Engram demo implementation:
    1. Uses Multiplicative-XOR hash.
    2. Supports layer-specific seeds.
    """

    def __init__(
        self,
        layer_id: int,
        ngram_sizes: List[int] = [2, 3],
        hash_heads: int = 8,
        memory_capacity_per_head: Optional[List[int]] = None,
        seed: int = 42,
        tokenizer_vocab_size: int = 129280,  # Default for DeepSeek-V3
    ):
        """
        Initialize the MultiHeadHash.

        Args:
            layer_id: The ID of the current layer (for layer-specific hashing).
            ngram_sizes: List of N-gram orders to consider.
            hash_heads: Number of hash heads per N-gram order.
            memory_capacity_per_head: List of target sizes for each hash head's embedding table.
            seed: Random seed for generating hash parameters.
            tokenizer_vocab_size: Size of the compressed vocabulary.
        """
        self.layer_id = layer_id
        self.ngram_sizes = ngram_sizes
        self.hash_heads = hash_heads
        self.seed = seed
        self.tokenizer_vocab_size = tokenizer_vocab_size

        num_total_heads = len(ngram_sizes) * hash_heads
        if memory_capacity_per_head is None:
            # 282809 is a prime size mentioned in the paper for some configurations
            # Defaulting to a reasonable prime if not provided
            self.memory_capacities = [282809] * num_total_heads
        else:
            if len(memory_capacity_per_head) != num_total_heads:
                if len(memory_capacity_per_head) == 1:
                    self.memory_capacities = memory_capacity_per_head * num_total_heads
                else:
                    raise ValueError(
                        f"Expected {num_total_heads} memory capacities, but got {len(memory_capacity_per_head)}"
                    )
            else:
                self.memory_capacities = memory_capacity_per_head

        # Generate unique primes for each head
        self.primes: List[int] = []
        seen_primes = set()
        for cap in self.memory_capacities:
            p_val = nextprime(cap - 1)
            if p_val is None:
                # Fallback or error handling
                p = cap  # very unlikely with nextprime
            else:
                p = int(p_val)

            while p in seen_primes:
                p_val = nextprime(p)
                if p_val is None:
                    p += 1
                else:
                    p = int(p_val)
            seen_primes.add(p)
            self.primes.append(p)

        # Generate layer-specific multipliers
        # Following demo: base_seed = seed + PRIME_1 * layer_id
        PRIME_1 = 10007
        base_seed = int(seed + PRIME_1 * int(layer_id))
        g = np.random.default_rng(base_seed)

        # half_bound calculation from demo
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)

        # Generate multipliers for each ngram order
        max_ngram = max(ngram_sizes)
        r = g.integers(low=0, high=half_bound, size=(max_ngram,), dtype=np.int64)
        # multipliers = r * 2 + 1 (ensures odd numbers)
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
        mix = (tokens[0] * m[0]) ^ (tokens[1] * m[1]) ^ ...
        """
        batch_size, seq_len = compressed_token_ids.shape
        device = compressed_token_ids.device
        results: Dict[Tuple[int, int], torch.Tensor] = {}

        # Pad tokens to handle beginning of sequence
        max_n = max(self.ngram_sizes)
        padding = torch.zeros((batch_size, max_n - 1), dtype=torch.long, device=device)
        # Use a pad token ID if available, here we just use 0 as a simple fallback
        padded_tokens = torch.cat([padding, compressed_token_ids], dim=1)

        for n in self.ngram_sizes:
            # Extract N-grams ending at each position
            # ngrams shape: [batch_size, seq_len, n]
            ngrams = padded_tokens[:, : seq_len + n - 1].unfold(1, n, 1)

            # multipliers: [n]
            m = self.all_multipliers[:n].to(device)

            # Compute mix = XOR_i(token_i * m_i)
            # token_i * m_i can overflow int64, which is expected for hashing
            weighted_tokens = ngrams * m  # [batch_size, seq_len, n]

            # PyTorch doesn't have a direct bitwise_xor reduce, so we do it manually
            mix = weighted_tokens[..., 0]
            for i in range(1, n):
                mix = torch.bitwise_xor(mix, weighted_tokens[..., i])

            for k in range(self.hash_heads):
                p = self.hash_params[(n, k)]["p"]
                # (mix % p) can be negative if mix is negative, so we do (mix % p + p) % p
                h = (mix % p + p) % p
                results[(n, k)] = h.long()

        return results
