# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none
from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, cast, override

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.optim.optimizer import Optimizer

    from engram_peft.types import TokenizerProtocol

from transformers import PreTrainedModel, TrainingArguments
from transformers.modeling_utils import unwrap_model
from trl import SFTConfig, SFTTrainer

from engram_peft.collator import EngramDataCollator
from engram_peft.model import EngramModel
from engram_peft.trainer import _is_deepspeed_enabled, _warn_distributed_sparse
from engram_peft.utils import (
    apply_group_wise_clipping,
    compute_grad_norm,
    get_trainable_param_groups,
)

logger = logging.getLogger(__name__)


def prepare_engram_for_sft(
    model: EngramModel,
    use_gradient_checkpointing: bool = True,
) -> EngramModel:
    """
    Prepare an EngramModel for supervised fine-tuning (SFT).

    This function performs essential model adjustments:
    1. Disables `use_cache` in the backbone config.
    2. Enables gradient checkpointing if requested.
    3. Sets the model to training mode.
    4. Logs trainable parameters.

    Args:
        model: The EngramModel to prepare.
        use_gradient_checkpointing: Whether to enable gradient checkpointing.

    Returns:
        EngramModel: The prepared model.
    """
    # 1. Disable cache for training (essential for gradient computation in SFT)

    if isinstance(model.base_model, PreTrainedModel):
        model.base_model.config.use_cache = False
        logger.info("Disabled base model KV cache for SFT.")
    elif hasattr(model.base_model, "config") and hasattr(
        model.base_model.config, "use_cache"
    ):
        # Fallback for dynamic config
        config = model.base_model.config
        if isinstance(config, PreTrainedModel):
            config = config.config
        object.__setattr__(config, "use_cache", False)
        logger.info("Disabled base model KV cache for SFT (dynamic).")
    else:
        logger.warning("Could not disable use_cache on base model config")

    # 2. Enable gradient checkpointing if requested
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing.")

    # 3. Ensure model is in training mode
    model.train()

    # 4. Print trainable parameters for user visibility
    model.print_trainable_parameters()

    return model


class EngramCompatibleSFTTrainer(SFTTrainer):
    """
    A compatibility wrapper for SFTTrainer.

    This class extends `trl.SFTTrainer` while performing safety checks
    for `EngramModel` during initialization.
    """

    optimizer_kwargs: dict[str, Any]
    optimizer: Optimizer | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the trainer with EngramModel compatibility checks.
        """
        self.optimizer_kwargs = kwargs.pop("optimizer_kwargs", {})
        # Ensure model is an EngramModel
        model = kwargs.get("model")
        if model is not None and not isinstance(model, EngramModel):
            logger.warning(
                "Model passed to EngramCompatibleSFTTrainer is not an EngramModel. %s",
                "The compatibility layer might not be necessary.",
            )

        super().__init__(*args, **kwargs)
        if model is not None:
            _warn_distributed_sparse(model, getattr(self, "args", None))

    @override
    def create_optimizer(self, model: Any = None) -> Optimizer:
        """
        Creates the MixedOptimizer for EngramModels to support sparse gradients.

        When DeepSpeed is enabled, the MixedOptimizer is skipped because DeepSpeed
        manages its own optimizer internally.
        """
        if self.optimizer is None:
            if _is_deepspeed_enabled(self.args):
                print(
                    "[Engram-PEFT] DeepSpeed detected in EngramCompatibleSFTTrainer: "
                    + "skipping MixedOptimizer. DeepSpeed will manage the optimizer. "
                    + "Set use_sparse_embeddings=False in EngramConfig for compatibility."
                )
                self.optimizer = super().create_optimizer(model)
            else:
                model_to_unwrap = model if model is not None else self.model
                if model_to_unwrap is None:
                    return super().create_optimizer()
                unwrapped_model = unwrap_model(cast("nn.Module", model_to_unwrap))
                if isinstance(unwrapped_model, EngramModel) and hasattr(
                    unwrapped_model, "create_optimizer"
                ):
                    self.optimizer = unwrapped_model.create_optimizer(
                        base_learning_rate=self.args.learning_rate,
                        **self.optimizer_kwargs,
                    )
                else:
                    self.optimizer = super().create_optimizer()
        assert self.optimizer is not None
        return self.optimizer

    @override
    def _clip_grad_norm(
        self, model: torch.nn.Module, max_norm: float | None = None
    ) -> torch.Tensor | None:
        """
        Custom gradient clipping that handles sparse tensors.
        Overrides the default behavior of transformers.Trainer.
        """
        if max_norm is None:
            # We must ensure max_norm is not None for comparison
            _max_norm_val: float = self.args.max_grad_norm or 0.0
            max_norm = _max_norm_val

        if max_norm <= 0:
            return None

        unwrapped_model = unwrap_model(model)
        total_norm = compute_grad_norm(model.parameters())

        use_per_group = (
            getattr(unwrapped_model.config, "clip_grad_per_group", False)
            if isinstance(unwrapped_model, EngramModel)
            else False
        )

        if use_per_group and isinstance(unwrapped_model, EngramModel):
            groups = get_trainable_param_groups(unwrapped_model)
            apply_group_wise_clipping(groups, max_norm)
        else:
            # Standard Global Norm Clipping with Sparse Support
            if total_norm is None:
                return None

            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        g = p.grad
                        if g.is_sparse:
                            # Manual clipping for sparse tensors
                            g.coalesce().values().detach().mul_(clip_coef.to(g.device))
                        else:
                            g.detach().mul_(clip_coef.to(g.device))
        return total_norm


def create_engram_sft_trainer(
    model: EngramModel,
    tokenizer: TokenizerProtocol,
    train_dataset: Any,
    eval_dataset: Any | None = None,
    args: TrainingArguments | SFTConfig | None = None,
    **kwargs: Any,
) -> SFTTrainer:
    """
    High-level factory function to create an SFTTrainer compatible with EngramModel.

    This function automates the most common setup steps for fine-tuning Engram models:
    - Model preparation (cache disabling, gradient checkpointing).
    - Automatic creation of EngramDataCollator if none is provided.
    - Initialization of SFTTrainer.

    Args:
        model: The EngramModel instance.
        tokenizer: The tokenizer instance.
        train_dataset: The training dataset.
        eval_dataset: Optional evaluation dataset.
        args: TrainingArguments or SFTConfig for the trainer.
        **kwargs: Additional arguments passed directly to SFTTrainer.

    Returns:
        trl.SFTTrainer: An initialized SFTTrainer instance.
    """
    # 1. Automatically prepare model for SFT
    prepare_engram_for_sft(model)

    # 2. Handle data collator
    data_collator = kwargs.pop("data_collator", None)
    if data_collator is None:
        # Use EngramDataCollator by default to ensure hash indices are precomputed
        data_collator = EngramDataCollator(tokenizer=tokenizer, config=model.config)
        logger.info("Using default EngramDataCollator for SFT integration.")

    # 3. Handle trl>=1.2.0 SFTConfig migration
    # max_seq_length was renamed to max_length and moved to SFTConfig
    max_seq_length = kwargs.pop("max_seq_length", None)
    if max_seq_length is not None:
        if isinstance(args, SFTConfig):
            args.max_length = max_seq_length
        elif isinstance(args, TrainingArguments):
            # Convert TrainingArguments to SFTConfig to support max_length
            args_dict = args.to_dict()
            args_dict["max_length"] = max_seq_length
            args = SFTConfig(**args_dict)
        else:
            # Create a default SFTConfig
            args = SFTConfig("outputs/sft_output", max_length=max_seq_length)

    # 4. Instantiate EngramCompatibleSFTTrainer
    # This class supports sparse gradient clipping and is mandatory for EngramModel
    # Detect trl version: processing_class (trl>=1.2.0) vs tokenizer (trl<1.2.0)
    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sft_params:
        return EngramCompatibleSFTTrainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            **kwargs,
        )
    else:
        return EngramCompatibleSFTTrainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs,
        )
