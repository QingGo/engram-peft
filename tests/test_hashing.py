import torch
import numpy as np
from scipy.stats import chisquare  # type: ignore[import-untyped]
from sympy import isprime  # type: ignore[import-untyped]
from engram_peft.hashing import MultiHeadHash


def test_prime_table_sizes() -> None:
    """
    测试用例 1：验证所有哈希表大小都是质数
    """
    ngram_sizes = [2, 3, 4]
    hash_heads = 4
    mhh = MultiHeadHash(ngram_sizes=ngram_sizes, hash_heads=hash_heads)

    for p in mhh.primes:
        assert isprime(p), f"Table size {p} is not prime"

    # Test with custom capacities
    custom_caps = [100, 200, 300, 400, 500, 600, 700, 800]
    mhh_custom = MultiHeadHash(
        ngram_sizes=[2], hash_heads=8, memory_capacity_per_head=custom_caps
    )
    for p in mhh_custom.primes:
        assert isprime(p), f"Custom table size {p} is not prime"


def test_reproducibility() -> None:
    """
    测试用例 2：验证相同输入产生相同哈希（可复现性）
    """
    seed = 42
    mhh1 = MultiHeadHash(seed=seed)
    mhh2 = MultiHeadHash(seed=seed)

    # Use .long() to ensure it's a LongTensor
    tokens = torch.randint(0, 10000, (4, 32)).long()

    hashes1 = mhh1.compute_hashes(tokens)  # type: ignore
    hashes2 = mhh2.compute_hashes(tokens)  # type: ignore

    for key in hashes1:
        assert torch.equal(
            hashes1[key], hashes2[key]
        ), f"Hashes for {key} are not equal across instances with same seed"

    # Verify same instance produces same hash for same input
    hashes1_again = mhh1.compute_hashes(tokens)  # type: ignore
    for key in hashes1:
        assert torch.equal(
            hashes1[key], hashes1_again[key]
        ), f"Hashes for {key} are not consistent within same instance"


def test_uniform_distribution() -> None:
    """
    测试用例 3：验证哈希分布均匀（卡方检验 p>0.05）
    """
    # Use a smaller prime for better statistics with fewer samples
    p = 1009
    mhh = MultiHeadHash(
        ngram_sizes=[2], hash_heads=1, memory_capacity_per_head=[p], seed=42
    )

    # Generate many unique N-grams
    num_samples = 20000
    # To get many unique 2-grams, we can use sequential IDs
    # (0, 1), (0, 2), ..., (0, 100), (1, 0), (1, 1), ...
    tokens = torch.zeros((1, num_samples), dtype=torch.long).long()
    tokens[0] = torch.arange(num_samples).long()

    hashes = mhh.compute_hashes(tokens)  # type: ignore
    h_values = hashes[(2, 0)].flatten().cpu().numpy()

    # Count frequencies
    counts = np.bincount(h_values, minlength=p)

    # Chi-square test
    # Expected frequency for each bucket is num_samples / p
    # Since we have n-1-th padding, the first hash will be for (0, token[0])
    # The number of hashes is exactly num_samples.
    f_exp = np.full(p, num_samples / p)
    chi2, p_val = chisquare(counts, f_exp=f_exp)

    print(f"Chi-square p-value: {p_val}")
    assert p_val > 0.05, f"Hash distribution is not uniform (p={p_val})"


def test_batch_processing() -> None:
    """
    测试用例 4：验证 batch 处理正确
    """
    mhh = MultiHeadHash(ngram_sizes=[2, 3], hash_heads=2)

    batch_size = 4
    seq_len = 16
    tokens = torch.randint(0, 1000, (batch_size, seq_len)).long()

    hashes = mhh.compute_hashes(tokens)  # type: ignore

    for (n, k), h_tensor in hashes.items():
        assert h_tensor.shape == (batch_size, seq_len), f"Incorrect shape for {(n, k)}"

        # Verify each sequence in batch is processed independently
        for b in range(batch_size):
            single_token = tokens[b : b + 1].long()
            single_hash = mhh.compute_hashes(single_token)[(n, k)]  # type: ignore
            assert torch.equal(
                h_tensor[b : b + 1], single_hash
            ), f"Batch element {b} differs from independent processing for {(n, k)}"


def test_ngram_suffix_correctness() -> None:
    """
    测试用例 5：验证 N-gram 后缀提取和手动计算的一致性
    """
    mhh = MultiHeadHash(
        ngram_sizes=[2], hash_heads=1, memory_capacity_per_head=[1009], seed=42
    )

    # tokens = [10, 20, 30]
    # For n=2:
    # t=0: ngram=(0, 10)
    # t=1: ngram=(10, 20)
    # t=2: ngram=(20, 30)
    tokens = torch.tensor([[10, 20, 30]], dtype=torch.long).long()
    hashes = mhh.compute_hashes(tokens)  # type: ignore
    h_values = hashes[(2, 0)].flatten().tolist()

    # Manually compute
    params = mhh.hash_params[(2, 0)]
    seed = params["head_seed"]
    p = params["p"]

    expected_h0 = mhh._multiplicative_xor_hash((0, 10), seed, p)
    expected_h1 = mhh._multiplicative_xor_hash((10, 20), seed, p)
    expected_h2 = mhh._multiplicative_xor_hash((20, 30), seed, p)

    assert h_values == [
        expected_h0,
        expected_h1,
        expected_h2,
    ], "Vectorized hashing doesn't match manual hashing"
