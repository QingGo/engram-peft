from typing import Any, Dict, Optional, cast

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from engram_peft.compression import CompressedTokenizer
from engram_peft.config import EngramConfig
from engram_peft.hashing import NgramHashMapping


class EngramDataCollator(DataCollatorForLanguageModeling):
    """
    Data Collator for Engram PEFT that precomputes hash indices on CPU.

    This collator inherits from transformers.DataCollatorForLanguageModeling.
    It extends the functionality by precalculating multi-head hash indices
    for all target layers specified in the EngramConfig. This precomputation
    is performed on the CPU during data loading to minimize GPU idle time.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: EngramConfig,
        compressor: Optional[CompressedTokenizer] = None,
        mlm: bool = False,
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
        super().__init__(tokenizer=tokenizer, mlm=mlm, **kwargs)
        self.config = config
        self.compressor = compressor

        # Initialize the global hashing mapping
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
            ngram_sizes=config.ngram_sizes,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.target_layers,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            pad_id=config.pad_id,
            seed=config.seed,
        )

    def __call__(
        self, examples: Any, return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        and engram_hash_indices.
        """
        # Capture raw input_ids before masking if we are doing MLM and want stable hashes
        raw_input_ids = None
        if self.mlm and isinstance(examples, list) and "input_ids" in examples[0]:
            # This is a heuristic to get unmasked IDs if they are available
            import torch

            # Use the same logic as transformers to pad the raw IDs
            raw_ids_list = [torch.tensor(e["input_ids"]) for e in examples]
            raw_input_ids = torch.nn.utils.rnn.pad_sequence(
                raw_ids_list,
                batch_first=True,
                padding_value=float(self.tokenizer.pad_token_id or 0),
            )

        # Step 1: Call parent to get basic LM batch (handles padding, labels, etc.)
        batch = cast(
            Dict[str, Any], super().__call__(examples, return_tensors=return_tensors)
        )

        # Step 2: Determine which IDs to use for hashing
        # If we captured raw IDs and this is MLM, use them to avoid hashing [MASK]
        input_ids = batch["input_ids"]
        if raw_input_ids is not None:
            batch["unmasked_input_ids"] = raw_input_ids.to(input_ids.device)
            hash_input_ids = raw_input_ids
        else:
            hash_input_ids = input_ids

        # Step 3: Compute and attach hash indices
        batch = self.compute_and_attach_hashes(batch, hash_input_ids)
        return batch

    def compute_and_attach_hashes(
        self, batch: Dict[str, Any], input_ids: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Helper method to compute hash indices and attach them to the batch.
        Can be used by other collators or directly.
        """
        # 1. Handle tokenizer compression if enabled and compressor is provided
        if self.compressor is not None:
            hash_input_ids = self.compressor.compress(input_ids)
        else:
            hash_input_ids = input_ids

        # 2. Precompute engram_hash_indices for all target layers on CPU
        hashes_dict = self.hash_mapping.hash(hash_input_ids)

        # 3. Consolidate indices into a single tensor [B, L, num_layers, total_heads]
        layer_hashes = []
        for layer_id in self.config.target_layers:
            h = hashes_dict[layer_id]
            # h is a numpy array on CPU
            layer_hashes.append(torch.from_numpy(h))

        # Stack along a new dimension for target layers
        # Result shape: [batch_size, seq_len, num_target_layers, total_heads]
        batch["engram_hash_indices"] = torch.stack(layer_hashes, dim=2)
        return batch
