import logging
from typing import Any, cast

import torch
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedTokenizerBase, TrainingArguments
from transformers.modeling_utils import unwrap_model
from trl import SFTConfig, SFTTrainer

from engram_peft.collator import EngramDataCollator
from engram_peft.model import EngramModel
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
    if hasattr(model.base_model, "config") and hasattr(
        model.base_model.config, "use_cache"
    ):
        model.base_model.config.use_cache = False
        logger.info("Disabled base model KV cache for SFT.")

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the trainer with EngramModel compatibility checks.
        """
        self.optimizer_kwargs = kwargs.pop("optimizer_kwargs", {})
        # Ensure model is an EngramModel
        model = kwargs.get("model")
        if model is not None and not isinstance(model, EngramModel):
            logger.warning(
                "Model passed to EngramCompatibleSFTTrainer is not an EngramModel. "
                "The compatibility layer might not be necessary."
            )

        super().__init__(*args, **kwargs)

    def create_optimizer(self) -> Optimizer:
        """
        Creates the MixedOptimizer for EngramModels to support sparse gradients.
        """
        if self.optimizer is None:
            unwrapped_model = unwrap_model(self.model)
            if isinstance(unwrapped_model, EngramModel) and hasattr(
                unwrapped_model, "create_optimizer"
            ):
                self.optimizer = unwrapped_model.create_optimizer(
                    base_learning_rate=self.args.learning_rate,
                    **self.optimizer_kwargs,
                )
            else:
                self.optimizer = super().create_optimizer()
        return self.optimizer

    def _clip_grad_norm(
        self, model: torch.nn.Module, max_norm: float | None = None
    ) -> torch.Tensor | None:
        """
        Custom gradient clipping that handles sparse tensors.
        Overrides the default behavior of transformers.Trainer.
        """
        if max_norm is None:
            max_norm = self.args.max_grad_norm

        if max_norm is None or max_norm <= 0:
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
    tokenizer: PreTrainedTokenizerBase,
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
            args = SFTConfig(output_dir="outputs/sft_output", max_length=max_seq_length)

    # 4. Instantiate EngramCompatibleSFTTrainer
    # This class supports sparse gradient clipping and is mandatory for EngramModel
    return EngramCompatibleSFTTrainer(
        model=cast("Any", model),
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        **kwargs,
    )
