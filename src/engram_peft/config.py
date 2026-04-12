import dataclasses
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

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
        default_factory=lambda: [1131200, 1131200]
    )
    max_ngram_size: int = 3
    n_head_per_ngram: int = 8
    embedding_dim: int = 1280
    enable_tokenizer_compression: bool = True
    target_layers: List[int] = field(default_factory=lambda: [2, 15])
    hc_mult: int = 4
    combine_mhc: bool = True
    conv_kernel_size: int = 4
    conv_dilation: Optional[int] = None
    conv_zero_init: bool = True
    learning_rate_multiplier: float = 5.0
    weight_decay: float = 0.0
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    pad_id: int = 2
    seed: int = 0

    def __init__(
        self,
        engram_vocab_size_per_ngram: Optional[List[int]] = None,
        max_ngram_size: int = 3,
        n_head_per_ngram: int = 8,
        embedding_dim: int = 1280,
        enable_tokenizer_compression: bool = True,
        target_layers: Optional[List[int]] = None,
        hc_mult: int = 4,
        combine_mhc: bool = True,
        conv_kernel_size: int = 4,
        conv_dilation: Optional[int] = None,
        conv_zero_init: bool = True,
        learning_rate_multiplier: float = 5.0,
        weight_decay: float = 0.0,
        tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3",
        pad_id: int = 2,
        seed: int = 0,
        **kwargs: Any,
    ):
        """Constructs EngramConfig."""
        self.engram_vocab_size_per_ngram = (
            engram_vocab_size_per_ngram
            if engram_vocab_size_per_ngram is not None
            else [1131200, 1131200]
        )
        self.max_ngram_size = max_ngram_size
        self.n_head_per_ngram = n_head_per_ngram
        self.embedding_dim = embedding_dim
        self.enable_tokenizer_compression = enable_tokenizer_compression
        self.target_layers = target_layers if target_layers is not None else [2, 15]
        self.hc_mult = hc_mult
        self.combine_mhc = combine_mhc
        self.conv_kernel_size = conv_kernel_size

        # dynamic default for conv_dilation
        self.conv_dilation = (
            conv_dilation if conv_dilation is not None else max_ngram_size
        )

        self.conv_zero_init = conv_zero_init
        self.learning_rate_multiplier = learning_rate_multiplier
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
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "EngramConfig":
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
        cls, pretrained_model_name_or_path: str, **kwargs
    ) -> "EngramConfig":
        """Loads from JSON file/pretrained repo."""
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Saves as JSON file."""
        import os

        os.makedirs(save_directory, exist_ok=True)
        super().save_pretrained(save_directory, **kwargs)
