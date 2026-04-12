import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from tokenizers import Regex, normalizers  # type: ignore
from transformers import PreTrainedTokenizer


class TokenizerCompressor:
    """
    Tokenizer compression module based on normalized textual equivalence.
    Implements a surjective function P: V -> V' as described in Section 2.2 of the Engram paper.
    Aligned with the official Engram demo implementation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        """
        Initialize the TokenizerCompressor.

        Args:
            tokenizer: The Hugging Face tokenizer to compress.
        """
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)

        # Sentinel for space normalization as in official demo
        SENTINEL = "\ue000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )

        self.mapping: Dict[int, int] = {}
        self.compressed_vocab_size: int = 0
        self.lookup: torch.Tensor = torch.empty(0)
        self._build_mapping()

    def _build_mapping(self) -> None:
        """
        Pre-compute the surjective mapping from original token IDs to canonical IDs.
        Matches the official demo's _build_lookup_table logic.
        """
        old2new: Dict[int, int] = {}
        key2new: Dict[str, int] = {}
        new_tokens: List[str] = []

        for tid in range(self.vocab_size):
            # Decode token as in demo
            text = str(self.tokenizer.decode([tid], skip_special_tokens=False))

            if "�" in text:
                # Use token string directly if it contains replacement char
                key = str(self.tokenizer.convert_ids_to_tokens(tid))
            else:
                norm = str(self.normalizer.normalize_str(text))
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        self.mapping = old2new
        self.compressed_vocab_size = len(new_tokens)

        # Create a lookup tensor for faster mapping
        self.lookup = torch.zeros(self.vocab_size, dtype=torch.long)
        for tid in range(self.vocab_size):
            self.lookup[tid] = torch.tensor(old2new[tid], dtype=torch.long)

    def compress(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compress a sequence of original token IDs into canonical IDs.

        Args:
            token_ids: Original token IDs as a torch tensor.

        Returns:
            Compressed token IDs.
        """
        # Ensure lookup is on the same device as input
        if self.lookup.device != token_ids.device:
            self.lookup = self.lookup.to(token_ids.device)

        return self.lookup[token_ids]

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the mapping table to a JSON file.
        """
        os.makedirs(save_directory, exist_ok=True)
        config = {
            "compressed_vocab_size": self.compressed_vocab_size,
            "vocab_size": self.vocab_size,
            "mapping": {str(k): v for k, v in self.mapping.items()},
        }
        with open(os.path.join(save_directory, "compression_config.json"), "w") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def from_pretrained(
        cls, save_directory: str, tokenizer: PreTrainedTokenizer
    ) -> "TokenizerCompressor":
        """
        Load a TokenizerCompressor from a saved directory.
        """
        config_path = os.path.join(save_directory, "compression_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        instance = cls.__new__(cls)
        instance.tokenizer = tokenizer
        instance.compressed_vocab_size = config["compressed_vocab_size"]
        instance.vocab_size = config["vocab_size"]
        instance.mapping = {int(k): v for k, v in config["mapping"].items()}

        # Create a lookup tensor for faster mapping
        instance.lookup = torch.zeros(instance.vocab_size, dtype=torch.long)
        for oid, cid in instance.mapping.items():
            instance.lookup[oid] = cid

        return instance
