import os
import tempfile
import unittest

from transformers import PretrainedConfig

from engram_peft.config import EngramConfig


class TestEngramConfig(unittest.TestCase):
    def test_default_config_matches_paper(self) -> None:
        """测试用例 1：验证默认配置与论文附录A完全一致"""
        config = EngramConfig()

        self.assertEqual(config.engram_vocab_size_per_ngram, [1131200, 1131200])
        self.assertEqual(config.max_ngram_size, 3)
        self.assertEqual(config.n_head_per_ngram, 8)
        self.assertEqual(config.embedding_dim, 1280)
        self.assertEqual(config.enable_tokenizer_compression, True)
        self.assertEqual(config.target_layers, [2, 15])
        self.assertEqual(config.hc_mult, 4)
        self.assertEqual(config.combine_mhc, True)
        self.assertEqual(config.conv_kernel_size, 4)
        self.assertEqual(config.conv_dilation, 3)  # default max_ngram_size
        self.assertEqual(config.conv_zero_init, True)
        self.assertEqual(config.learning_rate_multiplier, 5.0)
        self.assertEqual(config.weight_decay, 0.0)
        self.assertEqual(config.tokenizer_name_or_path, "deepseek-ai/DeepSeek-V3")
        self.assertEqual(config.pad_id, 2)
        self.assertEqual(config.seed, 0)
        self.assertEqual(config.model_type, "engram")

    def test_save_load_config(self) -> None:
        """测试用例 2：验证 save/load 功能正常"""
        config = EngramConfig(
            engram_vocab_size_per_ngram=[1000, 1000, 1000],
            ngram_sizes=[2, 3, 4],
            embedding_dim=256,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test save
            config.save_pretrained(tmp_dir)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "config.json")))

            # Test load
            loaded_config = EngramConfig.from_pretrained(tmp_dir)
            self.assertEqual(
                loaded_config.engram_vocab_size_per_ngram, [1000, 1000, 1000]
            )
            self.assertEqual(loaded_config.ngram_sizes, [2, 3, 4])
            self.assertEqual(loaded_config.max_ngram_size, 4)
            self.assertEqual(loaded_config.embedding_dim, 256)
            self.assertEqual(
                loaded_config.conv_dilation, 4
            )  # Falls back to new max_ngram_size

    def test_custom_config_override(self) -> None:
        """测试用例 3：验证自定义配置正确覆盖默认值"""
        config = EngramConfig(
            learning_rate_multiplier=10.0,
            weight_decay=0.1,
            target_layers=[5, 10, 15],
            conv_dilation=5,
        )

        self.assertEqual(config.learning_rate_multiplier, 10.0)
        self.assertEqual(config.weight_decay, 0.1)
        self.assertEqual(config.target_layers, [5, 10, 15])
        self.assertEqual(config.conv_dilation, 5)

        # Ensure unchanged items remain default
        self.assertEqual(config.embedding_dim, 1280)
        self.assertEqual(config.n_head_per_ngram, 8)

    def test_transformers_compatibility(self) -> None:
        """测试用例 4：验证与 transformers 的 PretrainedConfig 生态兼容"""
        config = EngramConfig(
            embedding_dim=512,
            return_dict=False,
            output_hidden_states=True,
            use_cache=False,
        )

        # Check dataclass attribute
        self.assertEqual(config.embedding_dim, 512)

        # Check base PretrainedConfig attributes
        self.assertEqual(config.return_dict, False)
        self.assertEqual(config.output_hidden_states, True)
        self.assertEqual(config.use_cache, False)

        # Check serialization contains both
        config_dict = config.to_dict()
        self.assertEqual(config_dict["embedding_dim"], 512)
        self.assertEqual(config_dict["return_dict"], False)
        self.assertEqual(config_dict["output_hidden_states"], True)

        # Check round-trip from dictionary via from_dict
        loaded_config = EngramConfig.from_dict(config_dict)
        self.assertEqual(loaded_config.embedding_dim, 512)
        self.assertEqual(loaded_config.return_dict, False)
        self.assertEqual(loaded_config.output_hidden_states, True)

        # Check instance is of PretrainedConfig
        self.assertIsInstance(loaded_config, PretrainedConfig)


if __name__ == "__main__":
    unittest.main()
