from typing import Dict, Any, cast
import torch
import torch.nn as nn
from engram_peft.config import EngramConfig
from engram_peft.layer import EngramLayer
from engram_peft.compression import CompressedTokenizer
from engram_peft.hashing import NgramHashMapping


class EngramModel(nn.Module):
    """
    Mock model to demonstrate Engram integration with global prime calculation.
    """

    def __init__(self, config: EngramConfig, tokenizer: Any = None) -> None:
        super().__init__()
        self.config = config

        # 1. Initialize compressor if needed
        self.compressor = None
        if tokenizer is not None:
            if isinstance(tokenizer, str):
                self.compressor = CompressedTokenizer(tokenizer)
            else:
                self.compressor = CompressedTokenizer(
                    getattr(config, "tokenizer_name_or_path", "deepseek-ai/DeepSeek-V3")
                )
                # If they pass tokenizer object, we bypass. The previous fix made it take str.

        # 2. Initialize global Hash Mapping
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
            max_ngram_size=config.max_ngram_size,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.target_layers,
            tokenizer_name_or_path=getattr(
                config, "tokenizer_name_or_path", "deepseek-ai/DeepSeek-V3"
            ),
            seed=config.seed,
        )

        # 3. Initialize Engram layers
        self.engram_layers = nn.ModuleDict()
        for layer_id in config.target_layers:
            # Flatten primes for this layer
            flat_primes = sum(self.hash_mapping.prime_tables[layer_id], [])
            self.engram_layers[str(layer_id)] = EngramLayer(
                config=config,
                layer_id=layer_id,
                primes=flat_primes,
                compressor=self.compressor,
            )

    def forward(
        self, input_ids: torch.Tensor, hidden_states_dict: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Apply Engram layers to corresponding hidden states.
        """
        for layer_id_str, layer in self.engram_layers.items():
            layer_id = int(layer_id_str)
            if layer_id in hidden_states_dict:
                # Type cast for layer access if needed, but nn.ModuleDict values are modules
                engram_layer = cast(EngramLayer, layer)
                hidden_states_dict[layer_id] = engram_layer(
                    input_ids=input_ids, hidden_states=hidden_states_dict[layer_id]
                )
        return hidden_states_dict
