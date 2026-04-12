import torch
import random
from typing import List, Dict, Tuple, Optional, TypedDict
from sympy import isprime, nextprime  # type: ignore[import-untyped]


class HashParams(TypedDict):
    p: int
    a: int
    b: int
    weights: torch.Tensor
    head_seed: int


class MultiHeadHash:
    """
    Implements Multi-Head Hashing as described in the paper section 2.2.
    For each N-gram order n, we employ K distinct hash heads.
    Each head maps a compressed context (N-gram) to an index in an embedding table.
    The hash is implemented as a multiplicative hash: (a * x + b) % prime.
    """

    def __init__(
        self,
        ngram_sizes: List[int] = [2, 3],
        hash_heads: int = 8,
        memory_capacity_per_head: Optional[List[int]] = None,
        seed: int = 42,
    ):
        """
        Initialize the MultiHeadHash.

        Args:
            ngram_sizes: List of N-gram orders to consider.
            hash_heads: Number of hash heads per N-gram order.
            memory_capacity_per_head: List of target sizes for each hash head's embedding table.
                                     If None, defaults to 282809 for all heads.
            seed: Random seed for generating hash parameters.
        """
        self.ngram_sizes = ngram_sizes
        self.hash_heads = hash_heads
        self.seed = seed

        num_total_heads = len(ngram_sizes) * hash_heads
        if memory_capacity_per_head is None:
            # 282809 is the prime size mentioned in the paper, derived from 2,262,400 / 8 heads
            self.memory_capacities = [282809] * num_total_heads
        else:
            if len(memory_capacity_per_head) != num_total_heads:
                # If only one value is provided, repeat it
                if len(memory_capacity_per_head) == 1:
                    self.memory_capacities = memory_capacity_per_head * num_total_heads
                else:
                    raise ValueError(
                        f"Expected {num_total_heads} memory capacities, but got {len(memory_capacity_per_head)}"
                    )
            else:
                self.memory_capacities = memory_capacity_per_head

        # Ensure all table sizes are primes
        # We need a prime for each head.
        self.primes = []
        for cap in self.memory_capacities:
            # Generate a unique prime for each head starting from the target capacity
            # Using a list to satisfy the _generate_prime_candidates requirement
            prime_list = self._generate_prime_candidates(cap)
            # Take a prime that hasn't been used yet if possible, or just the first one
            # To ensure uniqueness if needed, we could advance, but nextprime is deterministic.
            # For simplicity, we'll just use the first prime >= cap.
            # If the user wants distinct primes for each head, we can use different offsets.
            self.primes.append(prime_list[0])

        # Generate (a, b) parameters for each (ngram_size, head_idx)
        # We use a deterministic process based on the global seed and head index.
        self.hash_params: Dict[Tuple[int, int], HashParams] = {}
        prime_idx = 0
        for n in ngram_sizes:
            for k in range(hash_heads):
                p = self.primes[prime_idx]
                # Each head gets its own a, b coefficients.
                # Use a new RNG per head seeded by (global_seed, n, k) for stability.
                head_seed = hash((seed, n, k))
                rng = random.Random(head_seed)
                a = rng.randint(1, p - 1)
                b = rng.randint(0, p - 1)

                # Precompute weights for the vectorized computation in compute_hashes.
                # If x = sum(token_i * (base ** (n-1-i))), then
                # (a * x + b) % p = (sum(token_i * (a * base ** (n-1-i) % p)) + b) % p
                # We'll use base = 2**32 as a way to "treat N-gram as a big integer".
                base = 2**32
                weights = []
                for i in range(n):
                    # weight_i = (a * (base ** (n-1-i))) % p
                    w = (a * pow(base, n - 1 - i, p)) % p
                    weights.append(w)

                self.hash_params[(n, k)] = {
                    "p": p,
                    "a": a,
                    "b": b,
                    "weights": torch.tensor(weights, dtype=torch.long),
                    "head_seed": head_seed,
                }
                prime_idx += 1

    def _generate_prime_candidates(self, target_size: int) -> List[int]:
        """
        Generate a list of prime candidates starting from target_size.
        """
        primes = []
        curr = target_size
        # Just return the first prime >= target_size as a list of one element
        # as per the requirement for a "list of primes".
        if isprime(curr):
            primes.append(int(curr))
        else:
            p = nextprime(curr)
            primes.append(int(p))  # type: ignore
        return primes

    def _multiplicative_xor_hash(
        self, ngram: Tuple[int, ...], seed: int, prime: int
    ) -> int:
        """
        Implementation of the multiplicative hash: (a * x + b) % prime.
        Treats the N-gram tuple as a big integer by concatenating bits (base 2^32).

        Args:
            ngram: Tuple of token IDs.
            seed: Seed used to generate a and b.
            prime: The prime modulus (table size).
        """
        rng = random.Random(seed)
        a = rng.randint(1, prime - 1)
        b = rng.randint(0, prime - 1)

        # Treat N-gram as a big integer (base 2^32)
        x = 0
        for token in ngram:
            x = (x << 32) | (int(token) & 0xFFFFFFFF)

        return (a * x + b) % prime

    def compute_hashes(
        self, compressed_token_ids: torch.LongTensor
    ) -> Dict[Tuple[int, int], torch.LongTensor]:
        """
        Computes the hash indices for all N-grams and heads.

        Args:
            compressed_token_ids: Tensor of shape [batch_size, seq_len] containing token IDs.

        Returns:
            A dictionary mapping (ngram_size, head_idx) to a tensor of shape [batch_size, seq_len].
        """
        batch_size, seq_len = compressed_token_ids.shape
        device = compressed_token_ids.device
        results: Dict[Tuple[int, int], torch.LongTensor] = {}

        for n in self.ngram_sizes:
            # Extract N-grams for each position t.
            # The suffix N-gram at position t is (x'_{t-n+1}, ..., x'_t).
            # We pad the beginning of the sequence with 0s to handle t < n-1.
            padding = torch.zeros((batch_size, n - 1), dtype=torch.long, device=device)
            padded_tokens = torch.cat([padding, compressed_token_ids], dim=1)

            # ngrams shape: [batch_size, seq_len, n]
            # Each row i, t contains the N-gram ending at position t.
            ngrams = padded_tokens.unfold(1, n, 1)

            for k in range(self.hash_heads):
                params = self.hash_params[(n, k)]
                p = params["p"]
                b = params["b"]
                weights = params["weights"].to(device)

                # Compute (sum(token_i * weight_i) + b) % p
                # [batch_size, seq_len, n] * [n] -> [batch_size, seq_len]
                h = torch.sum(ngrams * weights, dim=-1)
                h = (h + b) % p

                results[(n, k)] = h.long()  # type: ignore

        return results
