from collections.abc import Iterable, Iterator, Sized
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch
import torch.nn as nn

T = TypeVar("T")


@runtime_checkable
class ToDictProtocol(Protocol):
    """Protocol for objects that have a to_dict method (like BatchEncoding)."""

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class ModelWithTags(Protocol):
    """Protocol for models that support add_model_tags (TRL integration)."""

    def add_model_tags(self, tags: list[str]) -> None: ...


@runtime_checkable
class SizedEncoding(Iterable[int], Sized, Protocol):
    """
    Protocol for tokenizers.Encoding to fix missing __len__ and __iter__ in stubs.
    """

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    @property
    def ids(self) -> list[int]: ...
    @property
    def attention_mask(self) -> list[int]: ...


@runtime_checkable
class GenerativeProtocol(Protocol):
    """Protocol for models that support generation."""

    def generate(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class PeftUnloadable(Protocol):
    """Protocol for PEFT models that support unload()."""

    def unload(self) -> nn.Module: ...


@runtime_checkable
class HFConfigProtocol(Protocol):
    """
    Protocol for Hugging Face-style configuration objects.
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
class HFModelProtocol(Protocol):
    """
    Refined structural protocol for Hugging Face-style models.
    """

    def generate(self, *args: Any, **kwargs: Any) -> Any: ...
    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None: ...
    def gradient_checkpointing_enable(self, **kwargs: Any) -> None: ...
    def gradient_checkpointing_disable(self, **kwargs: Any) -> None: ...

    @property
    def training(self) -> bool: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def config(self) -> HFConfigProtocol | Any: ...

    def tie_weights(self) -> None: ...

    def to(self, *args: Any, **kwargs: Any) -> nn.Module: ...
    def parameters(self, recurse: bool = True) -> Any: ...
    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def register_forward_pre_hook(self, hook: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class EngramComponentProtocol(Protocol):
    """
    Unified interface for EngramModel and EngramLayer.
    """

    @property
    def config(self) -> Any: ...

    @property
    def layer_id(self) -> int | None: ...

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
