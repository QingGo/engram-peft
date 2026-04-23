# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
from typing import Any, cast

import torch
from transformers import DataCollatorForLanguageModeling
from typing_extensions import override

from engram_peft.compression import CompressedTokenizer
from engram_peft.config import EngramConfig
from engram_peft.hashing import NgramHashMapping
from engram_peft.types import TokenizerProtocol
from engram_peft.utils import safe_from_numpy


class EngramDataCollator(DataCollatorForLanguageModeling):
    """
    Data Collator for Engram PEFT that precomputes hash indices on CPU.

    This collator inherits from transformers.DataCollatorForLanguageModeling.
    It extends the functionality by precalculating multi-head hash indices
    for all target layers specified in the EngramConfig. This precomputation
    is performed on the CPU during data loading to minimize GPU idle time.
    """

    config: EngramConfig
    compressor: CompressedTokenizer | None
    hash_mapping: NgramHashMapping

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        config: EngramConfig,
        compressor: CompressedTokenizer | None = None,
        mlm: bool = False,
        mlm_probability: float = 0.15,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the EngramDataCollator.

        Args:
            tokenizer: The Hugging Face PreTrainedTokenizer.
            config: EngramConfig containing PEFT hyperparameters.
            compressor: Optional CompressedTokenizer for token mapping.
            mlm: Whether to use masked language modeling.
            **kwargs: Additional arguments for DataCollatorForLanguageModeling.
        """
        # DataCollatorForLanguageModeling requires a nominal PreTrainedTokenizerBase
        # for its internal logic, so we cast to Any at the boundary.
        super().__init__(
            tokenizer=cast("Any", tokenizer),
            mlm=mlm,
            mlm_probability=mlm_probability,
            **kwargs,
        )
        self.config = config
        self.compressor = compressor

        # Map pad_id to compressed space for hashing consistency
        mapped_pad_id = config.pad_id
        if self.compressor is not None:
            assert config.pad_id is not None
            mapped_pad_id = self.compressor.map_id(config.pad_id)

        assert config.compressed_vocab_size is not None
        assert mapped_pad_id is not None

        # Initialize the global hashing mapping
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
            ngram_sizes=config.ngram_sizes,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.target_layers,
            compressed_vocab_size=config.compressed_vocab_size,
            pad_id=mapped_pad_id,
            seed=config.seed,
        )

    @override
    def __call__(
        self, features: list[dict[str, Any]], return_tensors: str | None = None
    ) -> dict[str, Any]:
        """
        Processes a batch of examples and adds precomputed engram hash indices.

        Args:
            features: List of tokenized examples from the dataset.
            return_tensors: The type of tensors to return.

        Returns:
            Dict[str, Any]: A dictionary containing input_ids, labels,
                and engram_hash_indices.
        """
        # Step 1: Call parent to get basic LM batch (handles padding, labels, etc.)
        batch = cast(
            "dict[str, Any]", super().__call__(features, return_tensors=return_tensors)
        )
        input_ids = cast("torch.Tensor", batch["input_ids"])

        # Step 1.5: Explicitly mask padding in labels (critical for training stability)
        if "labels" in batch and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id
            if isinstance(pad_id, int):
                batch["labels"][input_ids == pad_id] = -100

        # Step 2: Handle tokenizer compression if enabled and compressor is provided
        # If config says compression is enabled but no compressor is provided,
        # we fallback to raw IDs
        if self.compressor is not None:
            hash_input_ids = self.compressor.compress(input_ids)
        else:
            # If no compressor, we use the original input_ids for hashing
            hash_input_ids = input_ids

        # Step 3: Precompute engram_hash_indices for all target layers on CPU
        # hash_mapping.hash returns a Dict[int, np.ndarray] mapping layer_id -> indices
        hashes_dict = self.hash_mapping.hash(hash_input_ids)

        # Step 4: Consolidate indices into a single tensor [B, L, num_layers, total_heads]
        layer_hashes: list[torch.Tensor] = []
        for layer_id in self.config.target_layers:
            h = hashes_dict[layer_id]
            # h is a numpy array on CPU
            layer_hashes.append(safe_from_numpy(h))

        # Stack along a new dimension for target layers
        # Result shape: [batch_size, seq_len, num_target_layers, total_heads]
        batch["engram_hash_indices"] = torch.stack(layer_hashes, dim=2)

        return batch
