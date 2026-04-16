import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tokenizers import Regex, normalizers  # type: ignore
from transformers import AutoTokenizer


class CompressedTokenizer:
    """
    Tokenizer compression module based on normalized textual equivalence.
    Implements a surjective function P: V -> V' as described in Section 2.2 of the Engram paper.
    Aligned with the official Engram demo implementation.
    """

    def __init__(
        self,
        tokenizer_name_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
        tokenizer: Any = None,
    ) -> None:
        """
        Initialize the CompressedTokenizer.

        Args:
            tokenizer_name_or_path: The Hugging Face tokenizer path or name.
            trust_remote_code: Whether to trust remote code when loading tokenizer.
            tokenizer: Optional pre-loaded tokenizer instance to use instead of fetching from Hub.
        """
        self.tokenizer_name_or_path = tokenizer_name_or_path
        # Allow passing an existing path (e.g. locally) or load from HF
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self.vocab_size = len(self.tokenizer)
            # We still need the normalizer
            self.normalizer = self._get_default_normalizer()
            self.mapping: dict[int, int] = {}
            self.compressed_vocab_size = 0
            self.lookup = torch.empty(0)
            self._build_lookup_table()
        elif tokenizer_name_or_path is not None:
            if not os.path.isdir(tokenizer_name_or_path) or not os.path.exists(
                os.path.join(tokenizer_name_or_path, "compression_config.json")
            ):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path, trust_remote_code=trust_remote_code
                )
                self.vocab_size = len(self.tokenizer)
                self.normalizer = self._get_default_normalizer()
                self.mapping = {}
                self.compressed_vocab_size = 0
                self.lookup = torch.empty(0)
                self._build_lookup_table()
        else:
            raise ValueError(
                "Either tokenizer_name_or_path or a pre-loaded tokenizer must be provided."
            )

    def _get_default_normalizer(self) -> Any:
        # Sentinel for space normalization as in official demo
        SENTINEL = "\ue000"
        return normalizers.Sequence(
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

    def _build_lookup_table(self) -> None:
        """
        Pre-compute the surjective mapping from original token IDs to canonical IDs.
        Matches the official demo's logic.
        """
        old2new: Dict[int, int] = {}
        key2new: Dict[str, int] = {}
        new_tokens: List[str] = []

        for tid in range(self.vocab_size):
            text = str(self.tokenizer.decode([tid], skip_special_tokens=False))

            if "\ufffd" in text:
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
        # Optimization: Build list first, then create tensor once
        mapping_list = [old2new[tid] for tid in range(self.vocab_size)]
        self.lookup = torch.tensor(mapping_list, dtype=torch.long)

    def map_ids(
        self, input_ids: Union[torch.Tensor, np.ndarray[Any, Any]]
    ) -> Union[torch.Tensor, np.ndarray[Any, Any]]:
        """
        Maps a sequence of original token IDs to canonical (compressed) IDs.
        Handles negative IDs (like -100 for ignore_index) by preserving them.
        """
        is_numpy = isinstance(input_ids, np.ndarray)
        if is_numpy:
            tensor_ids = torch.from_numpy(input_ids).to(torch.long)
        else:
            tensor_ids = input_ids  # type: ignore

        if self.lookup.device != tensor_ids.device:
            self.lookup = self.lookup.to(tensor_ids.device)

        # 1. Create a mask for valid (non-negative) IDs
        mask = tensor_ids >= 0

        # 2. Clone input to preserve values in negative positions
        compressed = tensor_ids.clone()

        # 3. Only perform lookup for valid IDs
        valid_ids = tensor_ids[mask]
        if valid_ids.numel() > 0:
            compressed[mask] = self.lookup[valid_ids]

        if is_numpy:
            return compressed.cpu().numpy()
        return compressed

    def map_id(self, original_id: int) -> int:
        """Maps a single original token ID to compressed space."""
        if original_id < 0:
            return original_id
        if original_id >= len(self.lookup):
            return original_id
        return int(self.lookup[original_id])

    def compress(
        self, input_ids: Union[torch.Tensor, np.ndarray[Any, Any]]
    ) -> Union[torch.Tensor, np.ndarray[Any, Any]]:
        """Surjective mapping: collapses V tokens into V' tokens (V' < V)."""
        return self.map_ids(input_ids)

    def __len__(self) -> int:
        """Returns the size of the compressed vocabulary (V')."""
        return self.compressed_vocab_size

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the mapping table to a JSON file.
        """
        os.makedirs(save_directory, exist_ok=True)
        config = {
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "compressed_vocab_size": self.compressed_vocab_size,
            "vocab_size": self.vocab_size,
            "mapping": {str(k): v for k, v in self.mapping.items()},
        }
        with open(os.path.join(save_directory, "compression_config.json"), "w") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "CompressedTokenizer":
        """
        Load a CompressedTokenizer from a saved directory.
        """
        config_path = os.path.join(save_directory, "compression_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Initialize bypassing huggingface downloads
        instance = cls.__new__(cls)
        instance.tokenizer_name_or_path = config["tokenizer_name_or_path"]

        # Optionally, one could load the tokenizer here too if needed:
        # instance.tokenizer = AutoTokenizer.from_pretrained(...)

        instance.compressed_vocab_size = config["compressed_vocab_size"]
        instance.vocab_size = config["vocab_size"]
        instance.mapping = {int(k): v for k, v in config["mapping"].items()}

        instance.lookup = torch.zeros(instance.vocab_size, dtype=torch.long)
        for oid, cid in instance.mapping.items():
            instance.lookup[oid] = cid

        return instance
