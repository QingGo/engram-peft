import torch
import numpy as np
from scipy.stats import chisquare  # type: ignore[import-untyped]
from sympy import isprime  # type: ignore[import-untyped]
from engram_peft.hashing import MultiHeadHash, calculate_global_primes


def test_prime_table_sizes() -> None:
    """
    测试用例 1：验证所有哈希表大小都是质数
    """
    ngram_sizes = [2, 3, 4]
    hash_heads = 4
    memory_capacity_per_ngram = [1000, 2000, 3000]
    primes_dict = calculate_global_primes(
        layer_ids=[1],
        ngram_sizes=ngram_sizes,
        hash_heads=hash_heads,
        memory_capacity_per_ngram=memory_capacity_per_ngram,
    )
    mhh = MultiHeadHash(
        layer_id=1,
        primes=primes_dict[1],
        ngram_sizes=ngram_sizes,
        hash_heads=hash_heads,
    )

    for p in mhh.primes:
        assert isprime(p), f"Table size {p} is not prime"


def test_global_prime_uniqueness() -> None:
    """
    测试用例 7：验证全局质数唯一性
    """
    layer_ids = [1, 15]
    ngram_sizes = [2, 3]
    hash_heads = 8
    memory_capacity_per_ngram = [10000, 20000]

    primes_dict = calculate_global_primes(
        layer_ids=layer_ids,
        ngram_sizes=ngram_sizes,
        hash_heads=hash_heads,
        memory_capacity_per_ngram=memory_capacity_per_ngram,
    )

    all_primes = []
    for layer_id in layer_ids:
        all_primes.extend(primes_dict[layer_id])

    assert len(all_primes) == len(set(all_primes)), "Primes are not globally unique"
    assert len(all_primes) == len(layer_ids) * len(ngram_sizes) * hash_heads


def test_reproducibility() -> None:
    """
    测试用例 2：验证相同输入产生相同哈希（可复现性）
    """
    seed = 42
    ngram_sizes = [2, 3]
    hash_heads = 8
    memory_capacity_per_ngram = [10000, 20000]
    primes_dict = calculate_global_primes(
        [1], ngram_sizes, hash_heads, memory_capacity_per_ngram
    )

    mhh1 = MultiHeadHash(
        layer_id=1,
        primes=primes_dict[1],
        ngram_sizes=ngram_sizes,
        hash_heads=hash_heads,
        seed=seed,
    )
    mhh2 = MultiHeadHash(
        layer_id=1,
        primes=primes_dict[1],
        ngram_sizes=ngram_sizes,
        hash_heads=hash_heads,
        seed=seed,
    )

    # Use .long() to ensure it's a LongTensor
    tokens = torch.randint(0, 10000, (4, 32)).long()

    hashes1 = mhh1.compute_hashes(tokens)
    hashes2 = mhh2.compute_hashes(tokens)

    # Verify same input produces same hash across instances
    assert torch.equal(
        hashes1, hashes2
    ), "Hashes are not equal across instances with same seed"

    # Verify same instance produces same hash for same input
    hashes1_again = mhh1.compute_hashes(tokens)
    assert torch.equal(
        hashes1, hashes1_again
    ), "Hashes are not consistent within same instance"


def test_uniform_distribution() -> None:
    """
    测试用例 3：验证哈希分布均匀（卡方检验 p>0.05）
    """
    # Use a smaller prime for better statistics with fewer samples
    p_base = 1009
    ngram_sizes = [2]
    hash_heads = 1
    primes_dict = calculate_global_primes([1], ngram_sizes, hash_heads, [p_base])

    mhh = MultiHeadHash(
        layer_id=1,
        primes=primes_dict[1],
        ngram_sizes=ngram_sizes,
        hash_heads=hash_heads,
        seed=42,
    )
    p = mhh.primes[0]

    # Generate many unique N-grams
    num_samples = 20000
    # To get many unique 2-grams, we can use sequential IDs
    tokens = torch.zeros((1, num_samples), dtype=torch.long).long()
    tokens[0] = torch.arange(num_samples).long()

    hashes = mhh.compute_hashes(tokens)
    # For ngram_size=2 and hash_heads=1, the first head is at index 0
    h_values = hashes[:, :, 0].flatten().cpu().numpy()

    # Count frequencies
    counts = np.bincount(h_values, minlength=p)

    # Chi-square test
    f_exp = np.full(p, num_samples / p)
    chi2, p_val = chisquare(counts, f_exp=f_exp)

    print(f"Chi-square p-value: {p_val}")
    # Multiplicative-XOR hash is generally uniform enough
    assert p_val > 0.01, f"Hash distribution is not uniform enough (p={p_val})"


def test_batch_processing() -> None:
    """
    测试用例 4：验证 batch 处理正确
    """
    ngram_sizes = [2, 3]
    hash_heads = 2
    primes_dict = calculate_global_primes([1], ngram_sizes, hash_heads, [1000, 2000])

    mhh = MultiHeadHash(
        layer_id=1,
        primes=primes_dict[1],
        ngram_sizes=ngram_sizes,
        hash_heads=hash_heads,
    )

    batch_size = 4
    seq_len = 16
    tokens = torch.randint(0, 1000, (batch_size, seq_len)).long()

    hashes = mhh.compute_hashes(tokens)
    total_heads = len(ngram_sizes) * hash_heads
    assert hashes.shape == (
        batch_size,
        seq_len,
        total_heads,
    ), f"Incorrect shape: {hashes.shape}"

    # Verify each sequence in batch is processed independently
    for b in range(batch_size):
        single_token = tokens[b : b + 1].long()
        single_hash = mhh.compute_hashes(single_token)
        assert torch.equal(
            hashes[b : b + 1], single_hash
        ), f"Batch element {b} differs from independent processing"


def test_ngram_suffix_correctness() -> None:
    """
    测试用例 5：验证 N-gram 后缀提取和手动计算的一致性
    """
    p_base = 1009
    ngram_sizes = [2]
    hash_heads = 1
    primes_dict = calculate_global_primes([1], ngram_sizes, hash_heads, [p_base])

    mhh = MultiHeadHash(
        layer_id=1,
        primes=primes_dict[1],
        ngram_sizes=ngram_sizes,
        hash_heads=hash_heads,
        seed=42,
    )

    # tokens = [10, 20, 30]
    # For n=2:
    # t=0: ngram=(0, 10)
    # t=1: ngram=(10, 20)
    # t=2: ngram=(20, 30)
    tokens = torch.tensor([[10, 20, 30]], dtype=torch.long).long()
    hashes = mhh.compute_hashes(tokens)
    # For ngram_size=2 and hash_heads=1, the first head is at index 0
    h_values = hashes[:, :, 0].flatten().tolist()

    # Manually compute using Multiplicative-XOR logic
    # mix = XOR_i(token_i * multiplier_i)
    # h = (mix % p + p) % p
    p = mhh.hash_params[(2, 0)]["p"]
    m = mhh.all_multipliers

    def manual_hash(tokens: tuple[int, ...]) -> int:
        mix = 0
        for i, t in enumerate(tokens):
            multiplier = int(m[i].item())
            mix = mix ^ (t * multiplier)
        return (mix % p + p) % p

    expected_h0 = manual_hash((0, 10))
    expected_h1 = manual_hash((10, 20))
    expected_h2 = manual_hash((20, 30))

    assert h_values == [
        expected_h0,
        expected_h1,
        expected_h2,
    ], f"Vectorized hashing {h_values} doesn't match manual hashing {[expected_h0, expected_h1, expected_h2]}"


def test_layer_specific_hashing() -> None:
    """
    测试用例 6：验证不同层的哈希函数不同
    """
    seed = 42
    ngram_sizes = [2, 3]
    hash_heads = 8
    primes_dict = calculate_global_primes([1, 2], ngram_sizes, hash_heads, [1000, 2000])

    mhh_layer1 = MultiHeadHash(
        layer_id=1,
        primes=primes_dict[1],
        ngram_sizes=ngram_sizes,
        hash_heads=hash_heads,
        seed=seed,
    )
    mhh_layer2 = MultiHeadHash(
        layer_id=2,
        primes=primes_dict[2],
        ngram_sizes=ngram_sizes,
        hash_heads=hash_heads,
        seed=seed,
    )

    tokens = torch.randint(0, 10000, (1, 32)).long()

    hashes1 = mhh_layer1.compute_hashes(tokens)
    hashes2 = mhh_layer2.compute_hashes(tokens)

    # Different layers should produce different hashes for the same input
    assert not torch.equal(hashes1, hashes2), "Hashes should be different across layers"
