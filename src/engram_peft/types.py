# pyright: reportUnknownVariableType=none, reportUnusedImport=none
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sized
from typing import (
    Any,
    Protocol,
    TypedDict,
    TypeVar,
    cast,
    override,
    runtime_checkable,
)

import torch
import torch.nn as nn

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def jaxtyped(fn: F) -> F:
    """
    Dynamic decorator that enables jaxtyping + typeguard runtime checks
    ONLY if ENGRAM_DEBUG_SHAPES=1 is set. Otherwise, it's a no-op (identity).
    This eliminates runtime overhead in production/normal development.
    """
    import os  # noqa: PLC0415

    if os.environ.get("ENGRAM_DEBUG_SHAPES") == "1":
        from jaxtyping import jaxtyped as _jaxtyped  # noqa: PLC0415

        try:
            from typeguard import typechecked  # noqa: PLC0415

            return cast("F", _jaxtyped(typechecked(fn)))
        except ImportError:
            # Fallback if typeguard is not installed
            return cast("F", _jaxtyped(fn))
    return fn


@runtime_checkable
class ToDictProtocol(Protocol):
    """Protocol for objects that have a to_dict method (like BatchEncoding)."""

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class ModelWithTags(Protocol):
    """Protocol for models that support add_model_tags (TRL integration)."""

    def add_model_tags(self, tags: list[str]) -> None: ...


@runtime_checkable
class HasComputeDtype(Protocol):
    """Protocol for modules that expose a compute_dtype (like quantized models)."""

    compute_dtype: torch.dtype


@runtime_checkable
class ModelOutputProtocol(Protocol):
    """Protocol for Hugging Face model outputs."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    past_key_values: tuple[Any, ...] | None
    hidden_states: tuple[torch.Tensor, ...] | None
    attentions: tuple[torch.Tensor, ...] | None


@runtime_checkable
class CausalLMOutputProtocol(ModelOutputProtocol, Protocol):
    """Protocol specifically for causal language model outputs."""

    pass


@runtime_checkable
class SizedEncoding(Iterable[int], Sized, Protocol):
    """
    Protocol for tokenizers.Encoding to fix missing __len__ and __iter__ in stubs.
    """

    @override
    def __len__(self) -> int: ...

    @override
    def __iter__(self) -> Iterator[int]: ...

    @property
    def ids(self) -> list[int]: ...

    @property
    def attention_mask(self) -> list[int]: ...


@runtime_checkable
class ConfigProtocol(Protocol):
    """
    Protocol for Hugging Face-style configuration objects.
    Covers common fields used across transformers and engram-peft.
    """

    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    pad_token_id: int | None

    # Optional nested configs
    text_config: Any | None = None

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class ModelProtocol(Protocol):
    """
    Comprehensive structural protocol for model interactions.
    Standardizes access to common HF and PyTorch model attributes/methods.
    """

    @property
    def config(self) -> ConfigProtocol | Any: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def training(self) -> bool: ...

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def generate(self, *args: Any, **kwargs: Any) -> Any: ...

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None: ...

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None: ...

    def gradient_checkpointing_disable(self, **kwargs: Any) -> None: ...

    def tie_weights(self) -> None: ...

    def to(self, *args: Any, **kwargs: Any) -> nn.Module: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]: ...

    def named_modules(
        self, memo: Any | None = None, prefix: str = "", remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Module]]: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def register_forward_pre_hook(self, hook: Any, **kwargs: Any) -> Any: ...

    def add_model_tags(self, tags: list[str]) -> None: ...


class SafeTrainingArguments(TypedDict, total=False):
    """
    TypedDict for common TrainingArguments to avoid Unknown parameter errors.
    Covers the most frequently used flags in engram-peft.
    """

    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: float
    max_steps: int
    lr_scheduler_type: str
    warmup_steps: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    evaluation_strategy: str
    save_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    report_to: str | list[str]
    bf16: bool
    fp16: bool
    push_to_hub: bool
    hub_model_id: str
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    remove_unused_columns: bool


class OptimizerGroupDict(TypedDict, total=False):
    """
    Typed representation of a PyTorch optimizer parameter group.
    Used for group-wise learning rate and weight decay control.
    """

    params: list[torch.nn.Parameter] | Iterable[torch.nn.Parameter]
    lr: float
    weight_decay: float
    name: str
    is_sparse: bool
    initial_lr: float


@runtime_checkable
class OptimizerFactory(Protocol):
    """Protocol for callables that produce a torch Optimizer."""

    def __call__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        **kwargs: Any,
    ) -> torch.optim.Optimizer: ...


OptimizerSpec = str | type[torch.optim.Optimizer] | OptimizerFactory


@runtime_checkable
class SchedulerFactory(Protocol):
    """Protocol for callables that produce a learning rate scheduler."""

    def __call__(self, optimizer: torch.optim.Optimizer, **kwargs: Any) -> Any: ...


@runtime_checkable
class PeftUnloadable(Protocol):
    """Protocol for PEFT models that support unload()."""

    def unload(self) -> nn.Module: ...


@runtime_checkable
class EngramComponentProtocol(Protocol):
    """
    Unified structural interface for EngramModel and EngramLayer.
    Ensures both have access to configuration and optional layer identification.
    """

    @property
    def config(self) -> Any: ...

    @property
    def layer_id(self) -> int | None: ...

    # Hash mapping and optional compressor
    hash_mapping: Any
    compressor: Any | None

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class EngramModelProtocol(EngramComponentProtocol, Protocol):
    """
    Structural protocol specifically for EngramModel discovery and access.
    Focuses on engram-specific attributes rather than full nn.Module replication.
    """

    base_model: nn.Module

    @property
    def engram_layers(self) -> nn.ModuleDict: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...


@runtime_checkable
class TokenizerProtocol(Protocol):
    """
    Structural protocol for Hugging Face-style tokenizers.
    Allows for decoupled interactions with AutoTokenizers.
    """

    def __len__(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    # Attributes instead of properties for better stub compatibility
    pad_token_id: int | None
    pad_token: str | None
    eos_token: str | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def encode(self, *args: Any, **kwargs: Any) -> list[int] | Any: ...

    def decode(self, *args: Any, **kwargs: Any) -> str | Any: ...

    def convert_ids_to_tokens(
        self, *args: Any, **kwargs: Any
    ) -> str | list[str] | Any: ...
