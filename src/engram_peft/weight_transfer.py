import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm  # type: ignore

from engram_peft.config import EngramConfig
from engram_peft.hashing import NgramHashMapping

logger = logging.getLogger(__name__)


def check_compatibility(src_config: EngramConfig, target_config: EngramConfig) -> None:
    """Checks for fundamental configuration mismatches."""
    if src_config.seed != target_config.seed:
        warnings.warn(
            f"Hash seeds mismatch (src={src_config.seed}, target={target_config.seed}). "
            "Buckets will not align semantically. Direct transfer is NOT recommended "
            "unless followed by corpus-based remapping."
        )

    if src_config.tokenizer_name_or_path != target_config.tokenizer_name_or_path:
        warnings.warn(
            f"Tokenizers mismatch ({src_config.tokenizer_name_or_path} vs {target_config.tokenizer_name_or_path}). "
            "Buckets will not align semantically."
        )

    if src_config.hidden_size != target_config.hidden_size:
        raise ValueError(
            f"Hidden size mismatch ({src_config.hidden_size} vs {target_config.hidden_size}). "
            "Weights are structurally incompatible."
        )


def get_layer_mapping(
    src_layers: List[int],
    target_layers: List[int],
    manual_mapping: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    """Determines how source layers map to target layers."""
    if manual_mapping:
        return manual_mapping

    # Default: Map by index
    mapping = {}
    for i, src_id in enumerate(src_layers):
        if i < len(target_layers):
            mapping[src_id] = target_layers[i]
    return mapping


def align_embedding_table(
    src_weight: torch.Tensor,
    src_config: EngramConfig,
    target_config: EngramConfig,
    src_layer_id: int,
    target_layer_id: int,
    target_mapper: NgramHashMapping,
    src_mapper: NgramHashMapping,
) -> torch.Tensor:
    """
    Slices and pads the embedding table between different configurations.
    Handles n-gram size changes and bucket capacity changes.
    """
    src_primes = src_mapper.prime_tables[src_layer_id]
    target_primes = target_mapper.prime_tables[target_layer_id]

    # Calculate embedding dim per head
    src_total_heads = len(src_config.ngram_sizes) * src_config.n_head_per_ngram
    target_total_heads = len(target_config.ngram_sizes) * target_config.n_head_per_ngram

    src_dim_per_head = src_config.embedding_dim // src_total_heads
    target_dim_per_head = target_config.embedding_dim // target_total_heads

    if src_dim_per_head != target_dim_per_head:
        raise ValueError(
            f"Embedding dim per head mismatch ({src_dim_per_head} vs {target_dim_per_head}). "
            "Cannot align tables without re-projection."
        )

    # Calculate flat primes and offsets
    src_flat_primes = sum(src_primes, [])
    target_flat_primes = sum(target_primes, [])

    src_offsets = [0]
    for p in src_flat_primes[:-1]:
        src_offsets.append(src_offsets[-1] + p)

    target_offsets = [0]
    for p in target_flat_primes[:-1]:
        target_offsets.append(target_offsets[-1] + p)

    total_target_capacity = sum(target_flat_primes)
    new_weight = torch.zeros(
        (total_target_capacity, target_dim_per_head),
        dtype=src_weight.dtype,
        device=src_weight.device,
    )

    # Map shared n-grams
    for i, n_size in enumerate(target_config.ngram_sizes):
        if n_size in src_config.ngram_sizes:
            src_n_idx = src_config.ngram_sizes.index(n_size)
            # Both have this n-gram size, now map heads
            for head_idx in range(
                min(src_config.n_head_per_ngram, target_config.n_head_per_ngram)
            ):
                src_head_pos = src_n_idx * src_config.n_head_per_ngram + head_idx
                target_head_pos = i * target_config.n_head_per_ngram + head_idx

                src_start = src_offsets[src_head_pos]
                src_p = src_flat_primes[src_head_pos]

                target_start = target_offsets[target_head_pos]
                target_p = target_flat_primes[target_head_pos]

                # Copy minimum possible size
                copy_size = min(src_p, target_p)
                new_weight[target_start : target_start + copy_size] = src_weight[
                    src_start : src_start + copy_size
                ]

    return new_weight


def remap_weights_from_corpus(
    target_model: torch.nn.Module,
    src_state_dict: Dict[str, torch.Tensor],
    src_config: EngramConfig,
    corpus: Union[List[str], List[int], np.ndarray, torch.Tensor],
    layer_mapping: Optional[Dict[int, int]] = None,
    tokenizer: Optional[Any] = None,
    batch_size: int = 1024,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Best-effort remapping of weights for different seeds/tokenizers.
    Builds an index-to-index translation table using the provided corpus.

    If corpus is List[str], it performs character-level alignment between
    the source tokenizer and the target tokenizer.
    """
    from transformers import AutoTokenizer

    from engram_peft.model import EngramModel

    if not isinstance(target_model, EngramModel):
        raise ValueError("target_model must be an EngramModel instance.")

    target_config = target_model.config
    mapping = get_layer_mapping(
        src_config.target_layers, target_config.target_layers, layer_mapping
    )

    # Setup Source Mapper
    # If source used compression, we need to map the pad_id accordingly
    src_mapped_pad_id = src_config.pad_id
    if getattr(src_config, "enable_tokenizer_compression", True):
        # We need a temporary compressor for the source to resolve the pad_id mapping
        from engram_peft.compression import CompressedTokenizer

        try:
            src_compressor = CompressedTokenizer(src_config.tokenizer_name_or_path)
            assert src_config.pad_id is not None
            src_mapped_pad_id = src_compressor.map_id(src_config.pad_id)
        except Exception as e:
            logger.warning(
                f"Could not load source compressor for pad_id mapping: {e}. Using raw pad_id."
            )

    assert src_config.compressed_vocab_size is not None, (
        "source compressed_vocab_size must be set"
    )
    assert src_mapped_pad_id is not None, "source pad_id must be set"

    src_mapper = NgramHashMapping(
        engram_vocab_size_per_ngram=src_config.engram_vocab_size_per_ngram,
        ngram_sizes=src_config.ngram_sizes,
        n_head_per_ngram=src_config.n_head_per_ngram,
        layer_ids=list(mapping.keys()),
        compressed_vocab_size=src_config.compressed_vocab_size,
        pad_id=src_mapped_pad_id,
        seed=src_config.seed,
    )
    # target_model's internal hash_mapping is already initialized

    # Prepare index mapping buffers
    # We use a dictionary or sparse tensor for mapping to avoid massive memory usage
    # mapping_table[layer_id][head_idx] -> {target_idx: (sum_src_weight, count)}
    index_maps: Dict[int, List[Dict[int, Tuple[torch.Tensor, int]]]] = {}
    for src_id, target_id in mapping.items():
        total_heads = len(target_config.ngram_sizes) * target_config.n_head_per_ngram
        index_maps[src_id] = [{} for _ in range(total_heads)]

    # 1. Scan Corpus and Build Map
    if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], str):
        # Case: Raw text corpus (Cross-Tokenizer)
        logger.info("Performing cross-tokenizer alignment using raw text corpus...")
        src_tokenizer = AutoTokenizer.from_pretrained(src_config.tokenizer_name_or_path)
        # But since we are in a best-effort, we'll try to load it from config.
        target_tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            target_config.tokenizer_name_or_path or src_config.tokenizer_name_or_path
        )

        for item in tqdm(corpus, desc="Processing text", disable=not show_progress):
            if not isinstance(item, str):
                continue
            src_hashes, target_hashes = _get_aligned_hashes(
                item, src_tokenizer, target_tokenizer, src_mapper, target_model
            )
            _update_index_maps(
                index_maps,
                mapping,
                src_hashes,
                target_hashes,
                src_state_dict,
                src_mapper,
                target_config,
            )
    else:
        # Case: Token ID corpus (Same-Tokenizer, different seeds)
        if torch.is_tensor(corpus):
            tokens_np = corpus.cpu().numpy()  # type: ignore
        else:
            tokens_np = np.array(corpus)

        if tokens_np.ndim == 1:
            tokens_np = tokens_np.reshape(1, -1)

        steps = (tokens_np.shape[1] + batch_size - 1) // batch_size
        pbar = tqdm(total=steps, desc="Building Index Map", disable=not show_progress)

        for i in range(0, tokens_np.shape[1], batch_size):
            batch = tokens_np[:, i : i + batch_size]
            src_hashes = src_mapper.hash(batch)
            if target_model.compressor:
                c_ids = target_model.compressor.compress(
                    torch.from_numpy(batch).to(target_model.base_model.device)
                )
                c_ids_np = (
                    c_ids.cpu().numpy() if isinstance(c_ids, torch.Tensor) else c_ids
                )
                target_hashes = target_model.hash_mapping.hash(c_ids_np)
            else:
                target_hashes = target_model.hash_mapping.hash(batch)

            _update_index_maps(
                index_maps,
                mapping,
                src_hashes,
                target_hashes,
                src_state_dict,
                src_mapper,
                target_config,
            )
            pbar.update(1)
        pbar.close()

    # 2. Transfer Weights
    final_state_dict = {}
    for src_id, target_id in mapping.items():
        total_target_heads = (
            len(target_config.ngram_sizes) * target_config.n_head_per_ngram
        )
        target_layer = target_model.engram_layers[str(target_id)]
        target_emb_weight = (
            target_layer.multi_head_embedding.embedding.weight.data.clone()
        )

        # Target offsets
        target_primes = sum(target_model.hash_mapping.prime_tables[target_id], [])
        target_offsets = [0]
        for p in target_primes[:-1]:
            target_offsets.append(target_offsets[-1] + p)

        mapped_count = 0
        for head_idx in range(min(len(index_maps[src_id]), total_target_heads)):
            head_map = index_maps[src_id][head_idx]
            offset = target_offsets[head_idx]
            for t_idx, (sum_val, count) in head_map.items():
                target_emb_weight[offset + t_idx] = sum_val / count
                mapped_count += 1

        final_state_dict[f"{target_id}.multi_head_embedding.embedding.weight"] = (
            target_emb_weight
        )
        logger.info(
            f"Layer {src_id}->{target_id}: Mapped {mapped_count} buckets using corpus."
        )

    return final_state_dict


def _update_index_maps(
    index_maps: Dict[int, List[Dict[int, Tuple[torch.Tensor, int]]]],
    mapping: Dict[int, int],
    src_hashes: Dict[int, np.ndarray],
    target_hashes: Dict[int, np.ndarray],
    src_state_dict: Dict[str, torch.Tensor],
    src_mapper: NgramHashMapping,
    target_config: EngramConfig,
) -> None:
    """Helper to update index_maps from hashes."""
    for src_id, target_id in mapping.items():
        h_src = src_hashes[src_id]  # [B, L, K]
        h_target = target_hashes[target_id]  # [B, L, K]

        src_emb_key = f"{src_id}.multi_head_embedding.embedding.weight"
        src_weight = src_state_dict[src_emb_key]

        # Find offsets for src
        src_primes = sum(src_mapper.prime_tables[src_id], [])
        src_offsets = [0]
        for p in src_primes[:-1]:
            src_offsets.append(src_offsets[-1] + p)

        # Align length if needed (in cross-tokenizer case, L might differ but we zip them)
        L_src = h_src.shape[1]
        L_target = h_target.shape[1]
        L = min(L_src, L_target)

        for head_idx in range(min(h_src.shape[2], h_target.shape[2])):
            s_indices = h_src[:, :L, head_idx].flatten()
            t_indices = h_target[:, :L, head_idx].flatten()

            # Offset src indices
            s_indices_shifted = s_indices + src_offsets[head_idx]

            for s_idx, t_idx in zip(s_indices_shifted, t_indices):
                s_idx_val = int(s_idx)
                t_idx_val = int(t_idx)

                val = src_weight[s_idx_val]
                if t_idx_val not in index_maps[src_id][head_idx]:
                    index_maps[src_id][head_idx][t_idx_val] = (val.clone(), 1)
                else:
                    prev_val, count = index_maps[src_id][head_idx][t_idx_val]
                    index_maps[src_id][head_idx][t_idx_val] = (
                        prev_val + val,
                        count + 1,
                    )


def _get_aligned_hashes(
    text: str,
    src_tokenizer: Any,
    target_tokenizer: Any,
    src_mapper: NgramHashMapping,
    target_model: Any,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Tokenizes text with two tokenizers and aligns them using character offsets.
    Returns parallel hashes.
    """
    src_enc = src_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    target_enc = target_tokenizer(
        text, return_offsets_mapping=True, add_special_tokens=False
    )

    src_ids = src_enc["input_ids"]
    src_offsets = src_enc["offset_mapping"]

    target_ids = target_enc["input_ids"]
    target_offsets = target_enc["offset_mapping"]

    # Map characters to token indices
    char_to_src = {}
    for i, (start, end) in enumerate(src_offsets):
        for c in range(start, end):
            char_to_src[c] = i

    char_to_target = {}
    for i, (start, end) in enumerate(target_offsets):
        for c in range(start, end):
            char_to_target[c] = i

    # We want to find a sequence of (src_token_idx, target_token_idx)
    # Since multiple characters map to one token, we sample at the END of each target token.
    aligned_src_ids = []
    aligned_target_ids = []

    for i, (start, end) in enumerate(target_offsets):
        # Sample at the end of the target token
        sample_char = end - 1
        if sample_char in char_to_src:
            aligned_target_ids.append(target_ids[i])
            aligned_src_ids.append(src_ids[char_to_src[sample_char]])

    if not aligned_src_ids:
        return src_mapper.hash(
            np.zeros((1, 1), dtype=np.int64)
        ), target_model.hash_mapping.hash(np.zeros((1, 1), dtype=np.int64))

    # Convert to batches
    src_batch = np.array([aligned_src_ids], dtype=np.int64)
    target_batch = np.array([aligned_target_ids], dtype=np.int64)

    src_hashes = src_mapper.hash(src_batch)

    # Handle compressor for target if needed
    if target_model.compressor:
        t_batch_torch = torch.from_numpy(target_batch).to(
            target_model.base_model.device
        )
        c_ids = target_model.compressor.compress(t_batch_torch)
        target_hashes = target_model.hash_mapping.hash(c_ids.cpu().numpy())
    else:
        target_hashes = target_model.hash_mapping.hash(target_batch)

    return src_hashes, target_hashes
