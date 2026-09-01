import numpy as np
from scipy.stats import chisquare
from sympy import isprime

from engram_peft.hashing import (
    NgramHashMapping,
    QwenPleHashMapping,
    create_hash_mapping,
)


def test_hash_table_sizes_are_prime() -> None:
    """测试用例 1：验证所有哈希表大小都是质数"""
    mapping = NgramHashMapping(
        engram_vocab_size_per_ngram=[1000, 2000],
        ngram_sizes=[2, 3],
        n_head_per_ngram=4,
        layer_ids=[1, 2],
        compressed_vocab_size=129280,
    )
    for layer_id in [1, 2]:
        layer_primes = mapping.prime_tables[layer_id]
        for ngram_primes in layer_primes:
            for p in ngram_primes:
                assert isprime(p), f"Size {p} is not prime"


def test_reproducibility() -> None:
    """测试用例 2：验证相同输入产生相同哈希（可复现性）"""
    mapping1 = NgramHashMapping(seed=42, compressed_vocab_size=129280)
    mapping2 = NgramHashMapping(seed=42, compressed_vocab_size=129280)

    tokens = np.random.randint(0, 10000, size=(4, 32))

    hashes1 = mapping1.hash(tokens)
    hashes2 = mapping2.hash(tokens)

    for layer_id in hashes1:
        np.testing.assert_array_equal(
            hashes1[layer_id],
            hashes2[layer_id],
            err_msg="Hashes must be exactly the same for same seed",
        )


def test_uniform_distribution() -> None:
    """测试用例 3：验证哈希分布均匀（卡方检验 p>0.05，样本量 10000）"""
    # use small prime limits to get more accurate histograms easily
    mapping = NgramHashMapping(
        engram_vocab_size_per_ngram=[101],
        ngram_sizes=[2],
        n_head_per_ngram=1,
        layer_ids=[1],
        compressed_vocab_size=129280,
        seed=42,
    )
    p_size = mapping.prime_tables[1][0][0]

    num_samples = 10000
    tokens = np.zeros((1, num_samples), dtype=np.int64)
    tokens[0] = np.arange(num_samples)

    hashes = mapping.hash(tokens)[1]
    # first head, n=2
    h_vals = hashes[0, :, 0]

    counts = np.bincount(h_vals, minlength=p_size)
    f_exp = np.full(p_size, num_samples / p_size)

    _, p_val = chisquare(counts, f_exp=f_exp)

    # Multiplicative-XOR hash should be relatively uniform, p>0.01 is acceptable generally due to PRNG
    assert p_val > 0.01, (
        f"Hash distribution failed chi-square test with p-value={p_val}"
    )


def test_batch_processing() -> None:
    """测试用例 4：验证 batch 处理正确"""
    mapping = NgramHashMapping(compressed_vocab_size=129280)
    batch_size = 4
    seq_len = 16

    tokens = np.random.randint(0, 10000, size=(batch_size, seq_len))
    layer_id = mapping.layer_ids[-1]
    hashes = mapping.hash(tokens)[layer_id]

    # total_heads = len(ngram_sizes) * n_head_per_ngram
    # Here ngrams are 2, 3 -> 2 orders * 8 = 16
    expected_heads = len(mapping.ngram_sizes) * mapping.n_head_per_ngram
    assert hashes.shape == (batch_size, seq_len, expected_heads)

    for b in range(batch_size):
        single_input = tokens[b : b + 1]
        single_hash = mapping.hash(single_input)[layer_id]
        np.testing.assert_array_equal(
            hashes[b : b + 1],
            single_hash,
            err_msg=f"Batch element {b} diverges from individual execution",
        )


def test_layer_independence() -> None:
    """测试用例 5：验证不同层有不同的哈希结果"""
    mapping = NgramHashMapping(layer_ids=[1, 2], compressed_vocab_size=129280)
    tokens = np.random.randint(0, 1000, size=(2, 10))

    hashes = mapping.hash(tokens)

    h1 = hashes[1]
    h2 = hashes[2]

    # they should be entirely different due to distinct random multipliers and prime tables
    assert not np.array_equal(h1, h2), "Hashes from different layers should differ"


def test_qwen_ple_default_prime_table() -> None:
    """测试用例 6：PLE_QWEN_V1 默认 16 个 head prime 全部为质数。"""
    mapping = QwenPleHashMapping(compressed_vocab_size=100)
    primes = [p for heads in mapping.prime_tables[1] for p in heads]

    assert len(primes) == 16
    assert all(isprime(p) for p in primes)
    # PLE 使用 bigram 8 heads + trigram 8 heads
    assert [len(heads) for heads in mapping.prime_tables[1]] == [8, 8]


def test_qwen_ple_hash_shape_and_determinism() -> None:
    """测试用例 7：qwen_ple 哈希输出 shape 正确且可复现。"""
    tokens = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)

    mapping1 = QwenPleHashMapping(compressed_vocab_size=100)
    mapping2 = QwenPleHashMapping(compressed_vocab_size=100)

    hashes1 = mapping1.hash(tokens)
    hashes2 = mapping2.hash(tokens)

    assert set(hashes1) == {1}
    assert hashes1[1].shape == (2, 5, 16)
    np.testing.assert_array_equal(hashes1[1], hashes2[1])
    # 本地索引应落在对应 head prime 范围内
    for head_idx, prime in enumerate(sum(mapping1.prime_tables[1], [])):
        assert hashes1[1][..., head_idx].max() < prime


def test_create_hash_mapping_defaults_to_deepseek() -> None:
    """测试用例 8：不设置 engine/table_spec 时必须保持旧 deepseek 路径。"""
    mapping = create_hash_mapping(
        compressed_vocab_size=100,
        engram_vocab_size_per_ngram=[100, 100],
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        layer_ids=[1],
        pad_id=2,
        seed=0,
    )
    assert isinstance(mapping, NgramHashMapping)
    assert not isinstance(mapping, QwenPleHashMapping)


def test_create_hash_mapping_qwen_ple_selection() -> None:
    """测试用例 9：engine='qwen_ple' 或 table_spec='PLE_QWEN_V1' 选择新映射。"""
    kwargs = dict(
        compressed_vocab_size=100,
        engram_vocab_size_per_ngram=[100, 100],
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        layer_ids=[1],
        pad_id=2,
        seed=0,
    )
    assert isinstance(
        create_hash_mapping(**kwargs, engine="qwen_ple"),
        QwenPleHashMapping,
    )
    assert isinstance(
        create_hash_mapping(**kwargs, table_spec="PLE_QWEN_V1"),
        QwenPleHashMapping,
    )
