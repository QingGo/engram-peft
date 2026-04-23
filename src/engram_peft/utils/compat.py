# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

import torch
from huggingface_hub import snapshot_download as _snapshot_download
from safetensors.torch import load_file as _load_file
from safetensors.torch import save_file as _save_file
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

    from datasets import Dataset
    from torch import Tensor

    from engram_peft.types import (
        CausalLMOutputProtocol,
        ConfigProtocol,
        DatasetDictProtocol,
        DatasetProtocol,
        GenerativeModelProtocol,
        ModelOutputProtocol,
        ModelProtocol,
        SafeTrainingArguments,
        TokenizerProtocol,
    )


def safe_load_dataset(
    path: str, **kwargs: Any
) -> DatasetProtocol | DatasetDictProtocol:
    """
    Type-safe wrapper for datasets.load_dataset.
    """
    from datasets import load_dataset as _load_dataset  # noqa: PLC0415

    return cast("DatasetProtocol | DatasetDictProtocol", _load_dataset(path, **kwargs))


def safe_dataset_map(
    dataset: Dataset, function: Callable[..., Any], **kwargs: Any
) -> Dataset:
    """
    Type-safe wrapper for Dataset.map.
    """
    return dataset.map(function, **kwargs)


def safe_trainer_train(trainer: Any, **kwargs: Any) -> Any:
    """
    Type-safe wrapper for trainer.train().
    """
    return trainer.train(**kwargs)


def safe_load_file(
    filename: str | os.PathLike[Any], device: str = "cpu"
) -> dict[str, Tensor]:
    """
    Type-safe wrapper for safetensors.torch.load_file.
    Eliminates 'Unknown' return type warnings.
    """
    return _load_file(filename, device=device)


def safe_save_file(
    tensors: dict[str, Tensor],
    filename: str | os.PathLike[Any],
    metadata: dict[str, str] | None = None,
) -> None:
    """
    Type-safe wrapper for safetensors.torch.save_file.
    """
    _save_file(tensors, filename, metadata=metadata)


def safe_load(
    f: str | os.PathLike[Any],
    map_location: Any = None,
    weights_only: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Type-safe wrapper for torch.load.
    Defaults to weights_only=True for security.
    """
    return torch.load(f, map_location=map_location, weights_only=weights_only, **kwargs)


def safe_save(obj: Any, f: str | os.PathLike[Any], **kwargs: Any) -> None:
    """
    Type-safe wrapper for torch.save.
    """
    torch.save(obj, f, **kwargs)


def safe_model_from_pretrained(model_name_or_path: str, **kwargs: Any) -> ModelProtocol:
    """
    Type-safe wrapper for AutoModel.from_pretrained.
    Ensures the returned model adheres to ModelProtocol.
    """
    model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
    return cast("ModelProtocol", model)


def safe_causal_lm_from_pretrained(
    model_name_or_path: str, **kwargs: Any
) -> GenerativeModelProtocol:
    """
    Type-safe wrapper for AutoModelForCausalLM.from_pretrained.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return cast("GenerativeModelProtocol", model)


def safe_config_from_pretrained(
    model_name_or_path: str, **kwargs: Any
) -> ConfigProtocol:
    """
    Type-safe wrapper for AutoConfig.from_pretrained.
    Ensures the returned config adheres to ConfigProtocol.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
    return cast("ConfigProtocol", config)


def safe_tokenizer_from_pretrained(
    model_name_or_path: str, **kwargs: Any
) -> TokenizerProtocol:
    """
    Type-safe wrapper for AutoTokenizer.from_pretrained.
    Ensures the returned tokenizer adheres to TokenizerProtocol.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    return cast("TokenizerProtocol", tokenizer)


def safe_tokenizer_decode(
    tokenizer: TokenizerProtocol,
    token_ids: int | list[int] | torch.Tensor,
    skip_special_tokens: bool = False,
    **kwargs: Any,
) -> str:
    """
    Type-safe wrapper for tokenizer.decode.
    """
    return tokenizer.decode(
        token_ids, skip_special_tokens=skip_special_tokens, **kwargs
    )


def safe_from_numpy(ndarray: Any) -> torch.Tensor:
    """
    Type-safe wrapper for torch.from_numpy.
    Ensures the return type is a strongly-typed torch.Tensor.
    """
    return torch.from_numpy(ndarray)


def safe_norm(
    input_tensor: Any,
    p: float | str | None = 2,
    dim: int | list[int] | None = None,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Type-safe wrapper for torch.norm.
    """
    return cast(
        "torch.Tensor",
        torch.norm(input_tensor, p=p, dim=dim, keepdim=keepdim, dtype=dtype),
    )


def safe_stack(
    tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    dim: int = 0,
) -> torch.Tensor:
    """
    Type-safe wrapper for torch.stack.
    """
    return torch.stack(tensors, dim=dim)


def safe_cuda_is_available() -> bool:
    """
    Type-safe wrapper for torch.cuda.is_available.
    """
    return bool(torch.cuda.is_available())


def safe_cuda_is_bf16_supported() -> bool:
    """
    Type-safe wrapper for torch.cuda.is_bf16_supported.
    """
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def safe_snapshot_download(
    repo_id: str,
    **kwargs: Any,
) -> str:
    """
    Type-safe wrapper for snapshot_download.
    """
    res = _snapshot_download(repo_id=repo_id, **kwargs)
    return cast("str", res)


def as_scalar(val: Any) -> float:
    """
    Safely converts a value to a float scalar.
    Handles Tensor (via .item()) and float/int values.
    """
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def create_safe_training_args(
    args_dict: SafeTrainingArguments | None = None, **kwargs: Any
) -> TrainingArguments:
    """
    Type-safe wrapper for TrainingArguments.
    Automatically filters out non-standard arguments to avoid runtime errors
    with older transformers versions or custom configuration dictionaries.
    """
    # 1. Combine inputs
    combined: dict[str, Any] = {}
    if args_dict is not None:
        combined.update(args_dict)
    combined.update(kwargs)

    # 2. Filter against TrainingArguments.__init__ signature
    # This is the most robust way to ensure we don't pass invalid kwargs
    sig = inspect.signature(TrainingArguments.__init__)  # pyright: ignore[reportUnknownArgumentType]
    valid_keys = set(sig.parameters.keys())

    # Remove 'self' and 'kwargs' if present in signature
    valid_keys.discard("self")
    valid_keys.discard("kwargs")

    filtered = {k: v for k, v in combined.items() if k in valid_keys}

    return TrainingArguments(**filtered)


def safe_model_forward(
    model: nn.Module | ModelProtocol, **kwargs: Any
) -> ModelOutputProtocol:
    """
    Type-safe wrapper for model.forward.
    Casts the result to ModelOutputProtocol to ensure concrete member types.
    """
    res = model.forward(**kwargs)
    return cast("ModelOutputProtocol", res)


def safe_causal_lm_forward(
    model: nn.Module | ModelProtocol, **kwargs: Any
) -> CausalLMOutputProtocol:
    """
    Type-safe wrapper for causal LM forward pass.
    Ensures the result has concrete logits.
    """
    res = model.forward(**kwargs)
    return cast("CausalLMOutputProtocol", res)


def get_dim(tensor: torch.Tensor, dim: int) -> int:
    """
    Type-safe retrieval of tensor dimension.
    Prevents 'Unknown' warnings from direct .shape[i] access.
    """
    return int(tensor.shape[dim])


def get_batch_size(tensor: torch.Tensor) -> int:
    """Returns the batch size (dimension 0) of a tensor."""
    return get_dim(tensor, 0)


def get_seq_len(tensor: torch.Tensor) -> int:
    """Returns the sequence length (dimension 1) of a tensor."""
    return get_dim(tensor, 1)


def wash_tokenizer(tokenizer: Any) -> TokenizerProtocol:
    """
    Casts a tokenizer to TokenizerProtocol to 'wash' type errors
    caused by incomplete third-party stubs.
    """
    return cast("TokenizerProtocol", tokenizer)


def wash_model(model: Any) -> ModelProtocol:
    """
    Casts a model to ModelProtocol to 'wash' type errors
    caused by complex inheritance or incomplete stubs.
    """
    return cast("ModelProtocol", model)


def get_config_attr(config: Any, attr: str, default: Any = None) -> Any:
    """
    Type-safe retrieval of configuration attributes.
    Helps avoid 'Unknown' warnings from direct attribute access on HF configs.
    """
    return getattr(config, attr, default)
