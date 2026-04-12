import json
import os
import re
import unicodedata
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer


class TokenizerCompressor:
    """
    Tokenizer compression module based on normalized textual equivalence.
    Implements a surjective function P: V -> V' as described in Section 2.2 of the Engram paper.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        compression_ratio: float = 0.23,
    ) -> None:
        """
        Initialize the TokenizerCompressor.

        Args:
            tokenizer: The Hugging Face tokenizer to compress.
            compression_ratio: Target compression ratio (default 23% as in paper).
        """
        self.tokenizer = tokenizer
        self.compression_ratio = compression_ratio
        self.mapping: Dict[int, int] = {}
        self.vocab_size = len(tokenizer)
        self.compressed_vocab_size: int = 0
        self.lookup: torch.Tensor = torch.empty(0)
        self._build_mapping()

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text according to NFKC, lowercase, and space normalization.

        Args:
            text: Input token text.

        Returns:
            Normalized text.
        """
        # 1. NFKC normalization
        text = unicodedata.normalize("NFKC", text)
        # 2. Lowercasing
        text = text.lower()
        # 3. Merge space variants (\t, \n, \r, multiple spaces -> single space)
        text = re.sub(r"[\s\t\n\r]+", " ", text)
        # 4. Strip leading/trailing spaces
        stripped_text = text.strip()
        # If the text was only whitespace, the paper/prompt suggests it should be " "
        if not stripped_text and text:
            return " "
        return stripped_text

    def _build_mapping(self) -> None:
        """
        Pre-compute the surjective mapping from original token IDs to compressed IDs.
        """
        # Group original token IDs by their normalized text
        groups: Dict[str, List[int]] = {}
        for token_id in range(self.vocab_size):
            token_text = self.tokenizer.convert_ids_to_tokens(token_id)
            # Handle cases where convert_ids_to_tokens might return special objects or None
            if token_text is None:
                continue

            # Some tokenizers use special characters for spaces (e.g., Ġ or  )
            # We try to get the actual string representation
            if hasattr(self.tokenizer, "convert_tokens_to_string"):
                try:
                    text = self.tokenizer.convert_tokens_to_string([token_text])
                except Exception:
                    text = str(token_text)
            else:
                text = str(token_text)

            normalized_text = self._normalize_text(text)
            if normalized_text not in groups:
                groups[normalized_text] = []
            groups[normalized_text].append(token_id)

        # "按频率排序，保留 top (1-compression_ratio) 的 token，其余合并到 UNK"
        # We use vocab frequency (number of original tokens in each group) as the proxy.
        # Primary sort: frequency (descending), Secondary sort: min token id (ascending)
        sorted_groups = sorted(groups.items(), key=lambda x: (-len(x[1]), min(x[1])))

        target_size = int(self.vocab_size * (1 - self.compression_ratio))

        mapping: Dict[int, int] = {}
        current_compressed_id = 0

        # We'll use target_size - 1 as the UNK id if we truncate.
        unk_id = target_size - 1

        for i, (normalized_text, original_ids) in enumerate(sorted_groups):
            if i < target_size:
                for oid in original_ids:
                    mapping[oid] = current_compressed_id
                current_compressed_id += 1
            else:
                for oid in original_ids:
                    mapping[oid] = unk_id

        self.mapping = mapping
        self.compressed_vocab_size = min(len(groups), target_size)

        # Create a lookup tensor for faster mapping
        self.lookup = torch.zeros(self.vocab_size, dtype=torch.long)
        for oid, cid in self.mapping.items():
            self.lookup[oid] = torch.tensor(cid, dtype=torch.long)

    def compress(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compress a sequence of original token IDs into compressed IDs.

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

        Args:
            save_directory: Directory to save the mapping.
        """
        os.makedirs(save_directory, exist_ok=True)
        config = {
            "compression_ratio": self.compression_ratio,
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

        Args:
            save_directory: Directory containing the saved mapping.
            tokenizer: The tokenizer corresponding to the mapping.

        Returns:
            An initialized TokenizerCompressor.
        """
        config_path = os.path.join(save_directory, "compression_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        instance = cls.__new__(cls)
        instance.tokenizer = tokenizer
        instance.compression_ratio = config["compression_ratio"]
        instance.compressed_vocab_size = config["compressed_vocab_size"]
        instance.vocab_size = config["vocab_size"]
        instance.mapping = {int(k): v for k, v in config["mapping"].items()}

        # Create a lookup tensor for faster mapping
        instance.lookup = torch.zeros(instance.vocab_size, dtype=torch.long)
        for oid, cid in instance.mapping.items():
            instance.lookup[oid] = cid

        return instance
