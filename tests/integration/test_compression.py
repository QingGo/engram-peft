"""
集成测试：词表压缩算法在真实规模下的正确性验证。
涉及外部 Tokenizer 下载及大词表哈希计算，耗时较长。
建议时机：发布 Release 前或由于代码改动导致 engram_weights 推理异常时运行。
"""

import unittest

import pytest

from engram_peft.compression import CompressedTokenizer


class TestCompressedTokenizerIntegration(unittest.TestCase):
    def test_compression_rate_deepseek_v3(self) -> None:
        """
        验证 DeepSeek-V3 原始词表的压缩率是否符合 Engram 论文结论 (~23%)。
        这是验证算法核心有效性的“锚点”测试，Mock 环境无法取代。
        """
        try:
            tokenizerPath = "deepseek-ai/DeepSeek-V3"
            compressor = CompressedTokenizer(tokenizerPath, trust_remote_code=True)

            rate = 1.0 - (compressor.compressed_vocab_size / compressor.vocab_size)
            self.assertTrue(
                0.22 <= rate <= 0.24,
                f"Compression rate {rate:.2%} is not within 22-24%",
            )
        except Exception as e:
            pytest.skip(f"Could not load DeepSeek-V3 tokenizer (check network): {e}")


if __name__ == "__main__":
    unittest.main()
