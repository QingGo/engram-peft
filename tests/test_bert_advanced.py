import unittest

import numpy as np
import torch
from transformers import BertConfig as HuggingFaceBertConfig
from transformers import BertModel

from engram_peft.config import EngramConfig
from engram_peft.hashing import NgramHashMapping
from engram_peft.model import EngramModel


class TestBertAdvanced(unittest.TestCase):
    def test_centered_hashing(self) -> None:
        """Verify that centered hashing produces different indices than left hashing."""
        input_ids = torch.tensor([[10, 20, 30, 40, 50]])

        config_left = EngramConfig(ngram_sizes=[3], hashing_mode="left", seed=42)
        hasher_left = NgramHashMapping(
            ngram_sizes=config_left.ngram_sizes,
            hashing_mode=config_left.hashing_mode,
            seed=config_left.seed,
            layer_ids=[0],
        )

        config_center = EngramConfig(ngram_sizes=[3], hashing_mode="centered", seed=42)
        hasher_center = NgramHashMapping(
            ngram_sizes=config_center.ngram_sizes,
            hashing_mode=config_center.hashing_mode,
            seed=config_center.seed,
            layer_ids=[0],
        )

        h_left = hasher_left.hash(input_ids)[0]
        h_center = hasher_center.hash(input_ids)[0]

        # They should differ because the sliding windows are shifted
        self.assertFalse(np.array_equal(h_left, h_center))
        print("Centered vs Left hashing verified.")

    def test_stop_tokens(self) -> None:
        """Verify that stop_token_ids invalidate n-grams crossing them."""
        # Tokens: [10, 20, 30, 40, 50], where 30 is a stop token (e.g., [SEP])
        input_ids = torch.tensor([[10, 20, 30, 40, 50]])
        stop_token_id = 30

        config = EngramConfig(ngram_sizes=[3], stop_token_ids=[stop_token_id], seed=42)
        hasher = NgramHashMapping(
            ngram_sizes=config.ngram_sizes,
            stop_token_ids=config.stop_token_ids,
            seed=config.seed,
            layer_ids=[0],
        )

        h = hasher.hash(input_ids)[0]
        # n=3 windows:
        # pos 0: [pad, pad, 10] -> OK
        # pos 1: [pad, 10, 20] -> OK
        # pos 2: [10, 20, 30] -> Contains 30 -> Should be 0
        # pos 3: [20, 30, 40] -> Contains 30 -> Should be 0
        # pos 4: [30, 40, 50] -> Contains 30 -> Should be 0

        self.assertTrue(np.all(h[:, 2:5, :] == 0))
        self.assertTrue(np.all(h[:, 0:2, :] != 0))
        print("Stop tokens verification successful.")

    def test_target_modules_resolution(self) -> None:
        """Verify that target_modules strings are correctly resolved to hooks."""
        hf_config = HuggingFaceBertConfig.from_dict(
            {
                "num_hidden_layers": 4,
                "hidden_size": 128,
                "num_attention_heads": 8,
            }
        )
        base_model = BertModel(hf_config)

        # Target only the 2nd layer using a regex-like name
        config = EngramConfig(
            target_modules=[r"encoder\.layer\.1"],  # 0-indexed, so 2nd layer
            target_layers=[1],  # MUST include 1 so that EngramLayer(1) is created
            hidden_size=128,
        )

        _model = EngramModel(base_model, config)

        # Check if layer 1 has hooks (it should have 2: one from target_layers, one from target_modules)
        self.assertEqual(len(base_model.encoder.layer[1]._forward_pre_hooks), 2)
        self.assertEqual(len(base_model.encoder.layer[0]._forward_pre_hooks), 0)
        self.assertEqual(len(base_model.encoder.layer[2]._forward_pre_hooks), 0)
        print("Target modules resolution verified.")


if __name__ == "__main__":
    unittest.main()
