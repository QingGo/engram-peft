import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from engram_peft.compression import CompressedTokenizer
from engram_peft.config import EngramConfig
from engram_peft.hashing import NgramHashMapping
from engram_peft.layer import EngramLayer


class EngramModel(nn.Module):
    """
    Wrapper model that injects EngramLayer into a PreTrainedModel via forward hooks.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        config: EngramConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        """
        Initialize the Engram model.

        Args:
            base_model: The Hugging Face PreTrainedModel to wrap.
            config: EngramConfig containing hyperparameters.
            tokenizer: Optional tokenizer for extracting vocabulary size or mapping.
        """
        super().__init__()
        self.base_model = base_model
        # hidden_size is now an explicit field in EngramConfig.
        # It should be set correctly when the config is created.
        self.config = config

        # 1. Initialize compressor if needed
        self.compressor = None
        if config.enable_tokenizer_compression:
            tokenizer_name = config.tokenizer_name_or_path
            self.compressor = CompressedTokenizer(tokenizer_name)

        # 2. Initialize global Hash Mapping
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
            ngram_sizes=config.ngram_sizes,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.target_layers,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            seed=config.seed,
        )

        # 3. Initialize Engram layers
        self.engram_layers = nn.ModuleDict()
        for layer_id in config.target_layers:
            flat_primes = sum(self.hash_mapping.prime_tables[layer_id], [])
            self.engram_layers[str(layer_id)] = EngramLayer(
                config=config,
                layer_id=layer_id,
                primes=flat_primes,
                compressor=self.compressor,
            )

        self._hook_handles: List[Any] = []
        self._current_hash_indices: Optional[Union[Dict[int, Any], torch.Tensor]] = None
        self._engram_enabled = True

        # Named adapter management
        self.active_adapter = "default"
        self.peft_config = {"default": config}
        self.adapters = nn.ModuleDict()
        self.adapters["default"] = self.engram_layers

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
        trainable_params = 0
        all_param = 0
        for _, param in self.base_model.named_parameters():
            all_param += param.numel()
        for _, param in self.adapters.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"trainable params: {trainable_params:,} || all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def set_adapter(self, adapter_name: str) -> None:
        """
        Sets the active adapter.
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        self.engram_layers = cast(nn.ModuleDict, self.adapters[adapter_name])
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
        """Find the main transformer layers module list."""
        if hasattr(self.base_model, "model") and hasattr(
            self.base_model.model, "layers"
        ):
            return cast(nn.ModuleList, self.base_model.model.layers)
        if hasattr(self.base_model, "transformer") and hasattr(
            self.base_model.transformer, "h"
        ):
            return cast(nn.ModuleList, self.base_model.transformer.h)
        raise ValueError("Could not find transformer layers in the base model.")

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

            engram_layer = cast(EngramLayer, self.engram_layers[str(layer_id)])
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
        ) -> Optional[tuple]:
            if not self._engram_enabled:
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

    def load_engram(self, engram_path: Optional[str] = None) -> None:
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
        if self.config.target_modules is not None:
            # Module-based targeting
            target_names = (
                [self.config.target_modules]
                if isinstance(self.config.target_modules, str)
                else self.config.target_modules
            )
            import re

            for name, module in self.base_model.named_modules():
                for target_name in target_names:
                    if re.fullmatch(target_name, name) or target_name == name:
                        # We need a way to track these modules and assign an internal ID
                        # For simplicity, we use the name as ID if it's not a layer index
                        # But EngramLayer currently expects layer_id: int.
                        # This is a limitation we should address if we want true generic targeting.
                        # For now, we only support target_modules that match transformer blocks
                        # to avoid breaking the hashing/primes logic which is layer-indexed.
                        pass

        # Current layer-based targeting
        for layer_id in self.config.target_layers:
            if layer_id < len(layers):
                target_module = layers[layer_id]

                # Determine target device (from transformer block)
                try:
                    target_device = next(target_module.parameters()).device

                    # Align the entire adapter module to target device (including embeddings)
                    engram_layer = cast(EngramLayer, self.engram_layers[str(layer_id)])
                    engram_layer.to(target_device)
                except (StopIteration, AttributeError):
                    pass

                hook_handle = target_module.register_forward_pre_hook(
                    self._create_pre_hook(layer_id)
                )
                self._hook_handles.append(hook_handle)

        self._engram_enabled = True

    def unload_engram(self) -> None:
        """Removes all Engram hooks and effectively disables Engram operations."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._engram_enabled = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        engram_hash_indices: Optional[torch.Tensor] = None,
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
        generate_func = getattr(self.base_model, "generate")
        return generate_func(*args, **kwargs)

    def create_optimizer(self, base_learning_rate: float = 4e-4) -> Any:
        """
        Helper to create the MixedOptimizer for this model.
        """
        from engram_peft.utils import get_optimizer

        return get_optimizer(self, base_learning_rate=base_learning_rate)

    def create_scheduler(
        self, optimizer: Any, num_training_steps: int, warmup_steps: int = 0
    ) -> Any:
        """
        Helper to create the Step Decay scheduler for this model.
        """
        from engram_peft.utils import get_scheduler

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


def get_engram_model(
    model: PreTrainedModel,
    config: EngramConfig,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> EngramModel:
    """
    Wraps a base model with Engram layers.
    Freezes the base model and only leaves Engram Layer parameters trainable.
    """
    # Freeze the base model completely
    model.requires_grad_(False)

    # Ensure config has the correct hidden_size if it was using default
    base_config = getattr(model, "config", None)
    if base_config is not None:
        # Check various common attribute names for hidden size
        h_size = None
        for attr in ["hidden_size", "d_model", "dim", "n_embd"]:
            if hasattr(base_config, attr):
                h_size = getattr(base_config, attr)
                break

        if h_size is not None:
            config.hidden_size = h_size
    elif hasattr(model, "hidden_size"):
        # Fallback for models without a separate config object but with a direct attribute
        config.hidden_size = getattr(model, "hidden_size")

    engram_model = EngramModel(model, config, tokenizer)

    # Note: Engram layers were instantiated and they inherently have requires_grad=True
    # by PyTorch default. Verify just in case:
    for param in engram_model.engram_layers.parameters():
        param.requires_grad_(True)

    return engram_model
