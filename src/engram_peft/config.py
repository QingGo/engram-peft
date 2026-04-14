import dataclasses
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import PretrainedConfig


@dataclass(kw_only=True)
class EngramConfig(PretrainedConfig):
    """Configuration class for Engram PEFT module.

    This class inherits from transformers.PretrainedConfig and uses python
    dataclasses for defining Engram-specific hyperparameters exactly matching
    the specifications in the Engram paper Appendix A Table 5.
    """

    model_type = "engram"

    engram_vocab_size_per_ngram: List[int] = field(
        default_factory=lambda: [1131200, 1131200],
        metadata={
            "help": "Total number of hash buckets for each N-gram size (e.g., [2, 3]). "
            "This total capacity is distributed among the K heads."
        },
    )
    ngram_sizes: List[int] = field(
        default_factory=lambda: [2, 3],
        metadata={"help": "Explicit list of N-gram sizes to use (e.g., [2, 3])."},
    )
    max_ngram_size: int = 3
    n_head_per_ngram: int = 8
    embedding_dim: int = 1280
    enable_tokenizer_compression: bool = True
    target_layers: List[int] = field(default_factory=lambda: [2, 15])
    target_modules: Optional[Union[List[str], str]] = None
    hc_mult: int = 4
    combine_mhc: bool = True
    hidden_size: int = 2048
    conv_kernel_size: int = 4
    conv_dilation: Optional[int] = None
    conv_zero_init: bool = True
    gating_zero_init: bool = False
    learning_rate_multiplier: float = 5.0
    weight_decay: float = 0.0
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    pad_id: int = 2
    seed: int = 0

    def __init__(
        self,
        engram_vocab_size_per_ngram: Optional[List[int]] = None,
        ngram_sizes: Optional[List[int]] = None,
        n_head_per_ngram: int = 8,
        embedding_dim: int = 1280,
        enable_tokenizer_compression: bool = True,
        target_layers: Optional[List[int]] = None,
        hc_mult: int = 4,
        combine_mhc: bool = True,
        conv_kernel_size: int = 4,
        conv_dilation: Optional[int] = None,
        conv_zero_init: bool = True,
        gating_zero_init: bool = False,
        learning_rate_multiplier: float = 5.0,
        weight_decay: float = 0.0,
        tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3",
        pad_id: int = 2,
        seed: int = 0,
        hidden_size: int = 2048,
        **kwargs: Any,
    ):
        """Constructs EngramConfig."""
        self.engram_vocab_size_per_ngram = (
            engram_vocab_size_per_ngram
            if engram_vocab_size_per_ngram is not None
            else [1131200, 1131200]
        )
        self.ngram_sizes = ngram_sizes if ngram_sizes is not None else [2, 3]
        self.max_ngram_size = max(self.ngram_sizes)
        self.n_head_per_ngram = n_head_per_ngram
        self.embedding_dim = embedding_dim
        self.enable_tokenizer_compression = enable_tokenizer_compression
        self.target_layers = target_layers if target_layers is not None else [2, 15]
        self.hc_mult = hc_mult
        self.combine_mhc = combine_mhc
        self.conv_kernel_size = conv_kernel_size

        # dynamic default for conv_dilation
        self.conv_dilation = (
            conv_dilation if conv_dilation is not None else self.max_ngram_size
        )

        self.conv_zero_init = conv_zero_init
        self.gating_zero_init = gating_zero_init
        self.learning_rate_multiplier = learning_rate_multiplier
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.pad_id = pad_id
        self.seed = seed

        super().__init__(**kwargs)
        # Ensure extra kwargs are set, as PretrainedConfig might miss them in some environments
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs: Any) -> "EngramConfig":
        """Instantiates an EngramConfig from a python dictionary."""
        return super().from_dict(config_dict, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        # PretrainedConfig.to_dict handles basic fields but we need to ensure
        # all dataclass fields are included appropriately
        output = super().to_dict()

        # Override output with dataclass explicit values to capture our specific attributes
        for field_def in dataclasses.fields(self):
            output[field_def.name] = getattr(self, field_def.name)

        output["model_type"] = self.model_type
        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[Any],
        cache_dir: str | os.PathLike[Any] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> "EngramConfig":
        """Loads from JSON file/pretrained repo."""
        return super().from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[Any],
        push_to_hub: bool = False,
        **kwargs: Any,
    ) -> None:
        """Saves as JSON file."""
        import os

        os.makedirs(save_directory, exist_ok=True)
        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
