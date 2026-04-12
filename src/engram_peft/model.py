from typing import Dict, Any, cast
import torch
import torch.nn as nn
from engram_peft.config import EngramConfig
from engram_peft.layer import EngramLayer
from engram_peft.compression import TokenizerCompressor
from engram_peft.hashing import calculate_global_primes


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
            self.compressor = TokenizerCompressor(tokenizer)

        # 2. Calculate global primes for all layers
        self.primes_per_layer = calculate_global_primes(
            layer_ids=config.layer_ids,
            ngram_sizes=config.ngram_sizes,
            hash_heads=config.hash_heads,
            memory_capacity_per_ngram=config.memory_capacity_per_ngram,
        )

        # 3. Initialize Engram layers
        self.engram_layers = nn.ModuleDict()
        for layer_id in config.layer_ids:
            self.engram_layers[str(layer_id)] = EngramLayer(
                config=config,
                layer_id=layer_id,
                primes=self.primes_per_layer[layer_id],
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
