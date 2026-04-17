import logging
import os
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from engram_peft.compression import CompressedTokenizer
from engram_peft.config import EngramConfig
from engram_peft.discovery import ArchitectureResolver
from engram_peft.hashing import NgramHashMapping
from engram_peft.layer import EngramLayer
from engram_peft.utils import get_optimizer, get_scheduler
from engram_peft.weight_transfer import (
    align_embedding_table,
    check_compatibility,
    get_layer_mapping,
    remap_weights_from_corpus,
)

logger = logging.getLogger(__name__)

TrainMode = Literal["engram_only", "preserve_trainable", "full_finetune"]

# Architecture Layer Mapping moved to discovery.py


class EngramModel(nn.Module):
    """
    Wrapper model that injects EngramLayer into a PreTrainedModel via forward hooks.
    """

    _is_engram_model = True

    def __init__(
        self,
        base_model: PreTrainedModel | nn.Module,
        config: EngramConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """
        Initialize the Engram model.

        Args:
            base_model: The base model (nn.Module or PreTrainedModel) to wrap.
            config: EngramConfig containing hyperparameters.
            tokenizer: Optional tokenizer for extracting vocabulary size or mapping.
        """
        super().__init__()
        self.base_model = base_model
        self.train_mode: TrainMode = "engram_only"
        # hidden_size is now an explicit field in EngramConfig.
        # It should be set correctly when the config is created.
        self.config = config

        # 1. Initialize compressor if needed
        self.compressor = None
        if config.enable_tokenizer_compression:
            tokenizer_name = config.tokenizer_name_or_path
            self.compressor = CompressedTokenizer(tokenizer_name, tokenizer=tokenizer)
            # Synchronize the actual compressed vocab size to config
            config.compressed_vocab_size = len(self.compressor)
        else:
            # If no compression and not already set, we expect it to be in config
            if config.compressed_vocab_size is None:
                # This shouldn't normally happen if initialized via get_engram_model
                raise ValueError(
                    "compressed_vocab_size must be set in config if compression is disabled."
                )

        # 2. Map pad_id to compressed space for hashing consistency
        mapped_pad_id = config.pad_id
        if self.compressor is not None:
            assert config.pad_id is not None, (
                "pad_id must be set for compression mapping"
            )
            mapped_pad_id = self.compressor.map_id(config.pad_id)

        # 3. Initialize global Hash Mapping with resolved metadata
        assert config.compressed_vocab_size is not None, (
            "compressed_vocab_size must be set"
        )
        assert mapped_pad_id is not None, "pad_id must be set"

        self.hash_mapping = NgramHashMapping(
            engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
            ngram_sizes=config.ngram_sizes,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.target_layers,
            compressed_vocab_size=config.compressed_vocab_size,
            pad_id=mapped_pad_id,
            seed=config.seed,
        )

        # 3. Named adapter management & Initialization
        self.active_adapter = "default"
        self.peft_config = {"default": config}
        self.adapters = nn.ModuleDict()

        # Initialize Engram layers for the default adapter
        default_layers = nn.ModuleDict()
        for layer_id in config.target_layers:
            flat_primes = sum(self.hash_mapping.prime_tables[layer_id], [])
            default_layers[str(layer_id)] = EngramLayer(
                config=config,
                layer_id=layer_id,
                primes=flat_primes,
                compressor=self.compressor,
            )
        self.adapters["default"] = default_layers

        self._hook_handles: list[Any] = []
        self._current_hash_indices: dict[int, Any] | torch.Tensor | None = None
        self._engram_enabled = True

        # Attach hooks immediately
        self.load_engram()

        # Note: We do NOT unconditionally cast to base_model.dtype here.
        # Keeping Engram layers in float32 is essential for training stability
        # when the backbone is float16/bfloat16 (Mixed Precision).
        # Trainer/autocast will handle the precision transitions during forward/backward.
        self.adapters.float()

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        backbone_all = sum(
            param.numel() for _, param in self.base_model.named_parameters()
        )
        backbone_trainable = sum(
            param.numel()
            for _, param in self.base_model.named_parameters()
            if param.requires_grad
        )
        engram_all = sum(param.numel() for _, param in self.adapters.named_parameters())
        engram_trainable = sum(
            param.numel()
            for _, param in self.adapters.named_parameters()
            if param.requires_grad
        )
        trainable_params = backbone_trainable + engram_trainable
        all_param = backbone_all + engram_all

        print(
            f"trainable params: {trainable_params:,} "
            f"(backbone: {backbone_trainable:,}, engram: {engram_trainable:,}) || "
            f"all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    @property
    def engram_layers(self) -> nn.ModuleDict:
        """Dynamic shortcut to the active adapter's Engram layers."""
        return cast(nn.ModuleDict, self.adapters[self.active_adapter])

    def set_adapter(self, adapter_name: str) -> None:
        """
        Sets the active adapter.
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        self.config = self.peft_config[adapter_name]
        # Reinstate hooks to ensure they point to the correct layers if needed,
        # but since hooks use self.engram_layers[str(layer_id)], and we just
        # updated self.engram_layers, they should pick up the new modules.
        # However, it's safer to just set engram_enabled if it was disabled.
        self._engram_enabled = True

    def add_adapter(self, adapter_name: str, config: EngramConfig) -> None:
        """
        Adds a new adapter with the given name and config.
        """
        if adapter_name in self.adapters:
            raise ValueError(f"Adapter {adapter_name} already exists.")

        self.peft_config[adapter_name] = config
        new_layers = nn.ModuleDict()
        for layer_id in config.target_layers:
            # Note: We reuse the same hash_mapping logic or create new ones?
            # For simplicity, we create new layers.
            # We might need a separate hash_mapping per adapter if configs differ.
            # But hashing is usually tied to the backbone/tokenizer.
            flat_primes = sum(self.hash_mapping.prime_tables[layer_id], [])
            new_layers[str(layer_id)] = EngramLayer(
                config=config,
                layer_id=layer_id,
                primes=flat_primes,
                compressor=self.compressor,
            )
        new_layers.float()
        self.adapters[adapter_name] = new_layers

    def _find_transformer_layers(self) -> nn.ModuleList:
        """Find the main transformer layers module list using ArchitectureResolver."""
        container_path = self.config.layer_container_path
        if container_path is None:
            # This should have been resolved by the factory function, but as fallback:
            container_path = ArchitectureResolver.find_largest_module_list(
                self.base_model
            )
            if container_path is None:
                raise ValueError(
                    "Could not detect transformer layers. Specify layer_container_path."
                )

        container = ArchitectureResolver.get_submodule_by_path(
            self.base_model, container_path
        )
        if not isinstance(container, nn.ModuleList):
            raise ValueError(f"Path '{container_path}' is not a nn.ModuleList.")
        return container

    def _create_pre_hook(self, layer_id: int) -> Callable[[nn.Module, tuple], tuple]:
        """Creates a pre-forward hook for a specific layer."""

        def pre_hook(module: nn.Module, args: tuple) -> tuple:
            if not self._engram_enabled or self._current_hash_indices is None:
                return args

            hidden_states = args[0]
            # Safety check: hidden_states is usually [B, L, D]
            if hidden_states.dim() < 2:
                return args

            curr_seq_len = hidden_states.size(1)

            if isinstance(self._current_hash_indices, dict):
                indices_np = self._current_hash_indices[layer_id]
                # Check if sequence length matches
                if indices_np.shape[1] != curr_seq_len:
                    return args

                engram_hash_indices = torch.from_numpy(indices_np).to(
                    hidden_states.device
                )
            else:
                # Assume it's the stacked tensor [B, L, num_layers, total_heads]
                try:
                    # Check if sequence length matches
                    if self._current_hash_indices.size(1) != curr_seq_len:
                        return args

                    layer_idx = self.config.target_layers.index(layer_id)
                    engram_hash_indices = self._current_hash_indices[
                        :, :, layer_idx, :
                    ].to(hidden_states.device)
                except (ValueError, IndexError, AttributeError):
                    return args

            engram_layer = cast("EngramLayer", self.engram_layers[str(layer_id)])
            modified_hidden_states = engram_layer(
                hidden_states=hidden_states, engram_hash_indices=engram_hash_indices
            )

            # Replace args[0] with modified hidden states
            return (modified_hidden_states,) + args[1:]

        return pre_hook

    def _create_model_pre_hook(self) -> Callable:
        """Creates a pre-forward hook for the main base model to capture input_ids."""

        def model_pre_hook(
            module: nn.Module, args: tuple, kwargs: dict
        ) -> tuple | None:
            if not self._engram_enabled:
                return None

            # PERFORMANCE OPTIMIZATION:
            # If indices are already precomputed (e.g. by DataLoader or previous hook), Skip!
            if self._current_hash_indices is not None:
                return None

            # Try to get input_ids from kwargs then args
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0:
                input_ids = args[0]

            if input_ids is not None:
                # Precompute global hash indices
                if self.compressor:
                    c_ids = self.compressor.compress(input_ids)
                    if isinstance(c_ids, torch.Tensor):
                        c_ids = c_ids.cpu().numpy()
                    self._current_hash_indices = self.hash_mapping.hash(c_ids)
                else:
                    input_ids_np = (
                        input_ids.cpu().numpy()
                        if isinstance(input_ids, torch.Tensor)
                        else input_ids
                    )
                    self._current_hash_indices = self.hash_mapping.hash(input_ids_np)
            elif "hidden_states" not in kwargs and not (
                len(args) > 0 and torch.is_tensor(args[0]) and args[0].dim() >= 3
            ):
                # If this is a top-level call without input_ids, reset indices to avoid stale values
                self._current_hash_indices = None

            return None

        return model_pre_hook

    def load_engram(self, engram_path: str | None = None) -> None:
        """
        Dynamically loads and attaches Engram hooks.
        If engram_path is specified, optionally loads weights.
        """
        self.unload_engram()

        if engram_path is not None:
            weights_path = os.path.join(engram_path, "engram_weights.pt")
            if os.path.exists(weights_path):
                state_dict = torch.load(
                    weights_path, map_location="cpu", weights_only=True
                )
                self.engram_layers.load_state_dict(state_dict)

        layers = self._find_transformer_layers()

        # 1. Attach hook to the base model itself to capture input_ids automatically
        # Priority: run this before layer hooks
        model_hook = self.base_model.register_forward_pre_hook(
            self._create_model_pre_hook(), with_kwargs=True
        )
        self._hook_handles.append(model_hook)

        # 2. Attach hooks to target transformer layers or specific modules
        logger.info(f"[Engram-PEFT] Attaching Engram layers to {len(layers)} blocks...")

        # Current layer-based targeting
        for layer_id in self.config.target_layers:
            if layer_id < len(layers):
                target_module = layers[layer_id]
                module_class = type(target_module).__name__

                # Determine target device (from transformer block)
                target_device = "unknown"
                try:
                    target_device = str(next(target_module.parameters()).device)

                    # Align the entire adapter module to target device (including embeddings)
                    engram_layer = cast(
                        "EngramLayer", self.engram_layers[str(layer_id)]
                    )
                    engram_layer.to(target_device)
                except (StopIteration, AttributeError):
                    pass

                hook_handle = target_module.register_forward_pre_hook(
                    self._create_pre_hook(layer_id)
                )
                self._hook_handles.append(hook_handle)
                logger.info(
                    f"  - [Injected] Layer {layer_id} -> {module_class} "
                    f"(device: {target_device})"
                )

        self._engram_enabled = True

    def unload_engram(self) -> None:
        """Removes all Engram hooks and effectively disables Engram operations."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._engram_enabled = False

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        engram_hash_indices: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Forward pass delegating to base_model. Intercepts input_ids to precompute
        hash indices globally before the transformer blocks process them.
        """
        # Note: With the new model_pre_hook, manual hash computation here is
        # technically redundant but kept for cases where forward is called
        # directly on the EngramModel wrapper.
        if self._engram_enabled:
            # Crucial: Reset stale indices before setting new ones
            self._current_hash_indices = None

            input_ids_to_hash = input_ids
            if engram_hash_indices is not None:
                self._current_hash_indices = engram_hash_indices
            elif input_ids_to_hash is not None:
                if self.compressor:
                    c_ids = self.compressor.compress(input_ids_to_hash)
                    if isinstance(c_ids, torch.Tensor):
                        c_ids = c_ids.cpu().numpy()
                    self._current_hash_indices = self.hash_mapping.hash(c_ids)
                else:
                    input_ids_np = (
                        input_ids_to_hash.cpu().numpy()
                        if isinstance(input_ids_to_hash, torch.Tensor)
                        else input_ids_to_hash
                    )
                    self._current_hash_indices = self.hash_mapping.hash(input_ids_np)

        return self.base_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            **kwargs,
        )

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegates generation to the underlying base_model."""
        generate_func = self.base_model.generate
        return generate_func(*args, **kwargs)

    def create_optimizer(
        self, base_learning_rate: float = 4e-4, **optimizer_kwargs: Any
    ) -> Any:
        """
        Helper to create the MixedOptimizer for this model.
        """
        return get_optimizer(
            self, base_learning_rate=base_learning_rate, **optimizer_kwargs
        )

    def create_scheduler(
        self, optimizer: Any, num_training_steps: int, warmup_steps: int = 0
    ) -> Any:
        """
        Helper to create the Step Decay scheduler for this model.
        """
        return get_scheduler(
            optimizer, num_training_steps=num_training_steps, warmup_steps=warmup_steps
        )

    def save_pretrained(self, save_directory: str) -> None:
        """
        Saves only the Engram configurations and explicitly the Engram layers' weights.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        weights_path = os.path.join(save_directory, "engram_weights.pt")
        torch.save(self.engram_layers.state_dict(), weights_path)

    @classmethod
    def from_pretrained(
        cls, base_model: PreTrainedModel, engram_path: str
    ) -> "EngramModel":
        """
        Load an Engram model from a directory.
        """
        config = EngramConfig.from_pretrained(engram_path)
        model = cls(base_model, config, tokenizer=None)
        model.load_engram(engram_path)
        return model

    def load_weights_flexible(
        self,
        checkpoint_path: str,
        source_config_path: str | None = None,
        layer_mapping: dict[int, int] | None = None,
        reuse_structural: bool = False,
    ) -> None:
        """
        Loads weights from a checkpoint even if configurations differ.
        If configurations differ (e.g. layers, ngram_sizes), it performs
        best-effort alignment.
        """
        if source_config_path is None:
            source_config_dir = os.path.dirname(checkpoint_path)
            source_config_path = os.path.join(source_config_dir, "config.json")

        if not os.path.exists(source_config_path):
            raise FileNotFoundError(f"Source config not found at {source_config_path}")

        src_config = EngramConfig.from_pretrained(os.path.dirname(source_config_path))
        check_compatibility(src_config, self.config)
        assert src_config.compressed_vocab_size is not None, (
            "source compressed_vocab_size must be set"
        )

        # Initialize src_mapper with full target_layers to ensure deterministic prime generation
        src_mapper = NgramHashMapping(
            engram_vocab_size_per_ngram=src_config.engram_vocab_size_per_ngram,
            ngram_sizes=src_config.ngram_sizes,
            n_head_per_ngram=src_config.n_head_per_ngram,
            layer_ids=src_config.target_layers,
            compressed_vocab_size=src_config.compressed_vocab_size,
            pad_id=getattr(src_config, "pad_id", 2),
            seed=src_config.seed,
        )
        target_mapper = self.hash_mapping

        src_state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        mapping = get_layer_mapping(
            src_config.target_layers, self.config.target_layers, layer_mapping
        )

        target_state_dict = self.engram_layers.state_dict()

        for src_id, target_id in mapping.items():
            logger.info(
                f"Mapping weights: Source Layer {src_id} -> Target Layer {target_id}"
            )

            # 1. Align Embedding Table
            src_emb_key = f"{src_id}.multi_head_embedding.embedding.weight"
            target_emb_key = f"{target_id}.multi_head_embedding.embedding.weight"

            if src_emb_key in src_state_dict:
                target_state_dict[target_emb_key] = align_embedding_table(
                    src_state_dict[src_emb_key],
                    src_config,
                    self.config,
                    src_id,
                    target_id,
                    target_mapper=target_mapper,
                    src_mapper=src_mapper,
                )

            # 2. Structural modules (Gating, Conv)
            if reuse_structural:
                # Try to copy weights if shapes match
                for key in src_state_dict:
                    if (
                        key.startswith(f"{src_id}.")
                        and "multi_head_embedding" not in key
                    ):
                        suffix = key[len(str(src_id)) + 1 :]
                        target_key = f"{target_id}.{suffix}"
                        if target_key in target_state_dict:
                            if (
                                src_state_dict[key].shape
                                == target_state_dict[target_key].shape
                            ):
                                target_state_dict[target_key] = src_state_dict[
                                    key
                                ].clone()
                            else:
                                logger.warning(
                                    f"Shape mismatch for {target_key}, skipping structural reuse."
                                )

        self.engram_layers.load_state_dict(target_state_dict)
        logger.info("Flexible weight transfer complete.")

    def remap_from_corpus(
        self,
        corpus: list[str] | list[int] | np.ndarray | torch.Tensor,
        checkpoint_path: str,
        source_config_path: str | None = None,
        layer_mapping: dict[int, int] | None = None,
        tokenizer: Any | None = None,
        batch_size: int = 1024,
    ) -> None:
        """
        Remaps weights using a corpus to handle seed or tokenizer mismatches.
        If corpus is List[str], it performs cross-tokenizer alignment.
        """
        if source_config_path is None:
            source_config_dir = os.path.dirname(checkpoint_path)
            source_config_path = os.path.join(source_config_dir, "config.json")

        src_config = EngramConfig.from_pretrained(os.path.dirname(source_config_path))
        src_state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )

        new_emb_weights = remap_weights_from_corpus(
            self,
            src_state_dict,
            src_config,
            corpus,
            layer_mapping=layer_mapping,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

        # Load into existing engram_layers
        current_state_dict = self.engram_layers.state_dict()
        current_state_dict.update(new_emb_weights)
        self.engram_layers.load_state_dict(current_state_dict)
        logger.info("Corpus-based remapping complete.")


def get_engram_model(
    model: PreTrainedModel | nn.Module,
    config: EngramConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
    wrap_peft: bool = False,
    train_mode: TrainMode | None = None,
) -> EngramModel:
    """
    Wraps a base model with Engram layers.

    Args:
        model: The base model (nn.Module or PreTrainedModel) to wrap.
        config: EngramConfig for the Engram layers.
        tokenizer: Optional tokenizer.
        wrap_peft: Backward-compatible alias for train_mode="preserve_trainable".
        train_mode: Controls which backbone parameters remain trainable:
                   "engram_only" freezes the backbone,
                   "preserve_trainable" preserves already-trainable params,
                   "full_finetune" enables the full backbone.
    """
    if train_mode is None:
        resolved_train_mode: TrainMode = (
            "preserve_trainable" if wrap_peft else "engram_only"
        )
    else:
        resolved_train_mode = train_mode
        if wrap_peft and resolved_train_mode != "preserve_trainable":
            raise ValueError(
                "wrap_peft=True is only compatible with "
                'train_mode="preserve_trainable".'
            )

    if resolved_train_mode == "preserve_trainable":
        # Record parameters that were already trainable (e.g., from LoRA)
        trainable_before = [p for p in model.parameters() if p.requires_grad]
        # Freeze the entire model
        model.requires_grad_(False)
        # Restore requires_grad=True for those parameters
        for p in trainable_before:
            p.requires_grad_(True)
    elif resolved_train_mode == "full_finetune":
        model.requires_grad_(True)
    elif resolved_train_mode == "engram_only":
        # Standard vanilla model, freeze everything
        model.requires_grad_(False)
    else:
        raise ValueError(f"Unsupported train_mode: {resolved_train_mode}")

    # --- Auto-discovery and configuration synchronization ---
    metadata = ArchitectureResolver.resolve(model, tokenizer, config)

    # Update config with resolved values to ensure persistence in save_pretrained
    config.hidden_size = metadata.hidden_size
    config.pad_id = metadata.pad_token_id
    config.layer_container_path = metadata.layer_container_path

    # If compression is disabled, the 'compressed' size is just the original vocab size
    if not config.enable_tokenizer_compression:
        config.compressed_vocab_size = metadata.original_vocab_size

    engram_model = EngramModel(model, config, tokenizer)
    engram_model.train_mode = resolved_train_mode

    # Note: Engram layers were instantiated and they inherently have requires_grad=True
    # by PyTorch default. Verify just in case:
    for param in engram_model.engram_layers.parameters():
        param.requires_grad_(True)

    return engram_model
