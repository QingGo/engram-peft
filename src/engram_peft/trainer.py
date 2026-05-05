# trainer.py
# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportAttributeAccessIssue=none, reportIndexIssue=none, reportArgumentType=none, reportCallIssue=none
import logging
import os
from collections.abc import Iterable
from typing import Any, cast, override

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.trainer import TRAINING_ARGS_NAME, _is_peft_model

from engram_peft.model import EngramModel
from engram_peft.saving import load_engram_weights

logger = logging.getLogger(__name__)
from engram_peft.utils import (
    apply_group_wise_clipping,
    as_dict,
    as_module,
    compute_grad_norm,
    compute_telemetry_stats,
    get_scheduler,
    get_trainable_param_groups,
    iter_parameters,
)
from engram_peft.utils.compat import get_config_attr, wash_model


def _is_deepspeed_enabled(args: Any) -> bool:
    """Check if DeepSpeed is configured in the training arguments."""
    return getattr(args, "deepspeed", None) is not None


def _is_distributed() -> bool:
    """Check if running in any distributed mode (DDP or DeepSpeed via torchrun)."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _warn_distributed_sparse(model: Any, _args: Any = None) -> None:
    """Warn if distributed mode is incompatible with sparse embeddings.

    DDP (DistributedDataParallel) does not support sparse gradients regardless
    of the backend (NCCL or Gloo), because DDP flattens all gradients into
    dense buckets before all_reduce. The same applies to DeepSpeed ZeRO.
    Use single-GPU training for sparse embeddings + SparseAdam.
    """
    if model is None:
        return
    try:
        unwrapped = unwrap_model(model)
    except Exception:
        return
    if not isinstance(unwrapped, EngramModel):
        return
    if not get_config_attr(unwrapped.config, "use_sparse_embeddings"):
        return

    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not is_distributed:
        return

    print(
        "[Engram-PEFT] Warning: Distributed training detected with "
        + "use_sparse_embeddings=True. DDP converts all gradients to dense "
        + "during all_reduce, making SparseAdam unusable. The MixedOptimizer "
        + "will be skipped; a standard AdamW optimizer will be used instead. "
        + "For sparse embedding memory savings, train on a single GPU."
    )


class EngramTrainer(Trainer):
    """
    Custom Trainer that handles sparse gradient clipping and norm calculation.

    Standard torch.nn.utils.clip_grad_norm_ (and by extension accelerate's default implementation
    used in SFTTrainer) does not support SparseCPU/SparseCUDA tensors. This trainer
    overrides the clipping logic to correctly handle sparse parameters, making it the
    preferred choice when `use_sparse_embeddings=True` is set in the EngramConfig.

    .. note::
        When DeepSpeed is enabled, the custom MixedOptimizer is skipped because
        DeepSpeed manages its own optimizer internally. SparseAdam is not compatible
        with DeepSpeed; set ``use_sparse_embeddings=False`` in EngramConfig.
    """

    optimizer_kwargs: dict[str, Any]
    _initial_weights: dict[int, torch.Tensor]
    _last_ce_loss: float
    _last_entropy_loss: float
    optimizer: Optimizer | None = None
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def __init__(
        self,
        *args: Any,
        optimizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer_kwargs = optimizer_kwargs or {}
        self._initial_weights = {}
        # Metrics for Fair Comparison
        self._last_ce_loss = 0.0
        self._last_entropy_loss = 0.0

        if self.model is not None:
            self._handle_initial_freezing()
            self._capture_initial_weights()
            _warn_distributed_sparse(self.model, self.args)

    def _handle_initial_freezing(self) -> None:
        """Initially freezes the backbone if backbone_freeze_steps > 0."""
        if self.model is None:
            return
        unwrapped = unwrap_model(as_module(self.model))
        if not isinstance(unwrapped, EngramModel):
            return

        freeze_steps = int(
            get_config_attr(unwrapped.config, "backbone_freeze_steps") or 0
        )
        if freeze_steps > 0:
            print(
                f"[Engram-PEFT] Adapter-First Mode: Freezing backbone for {freeze_steps} steps."
            )
            # Only freeze the parameters that were originally intended to be trainable
            trainable_backbone_names: set[str] | None = getattr(
                unwrapped, "trainable_backbone_names", None
            )
            if trainable_backbone_names is not None:
                for name, param in unwrapped.base_model.named_parameters():
                    if name in trainable_backbone_names:
                        param.requires_grad = False

    def _capture_initial_weights(self) -> None:
        """Captures initial weights of Engram modules on CPU for drift calculation."""
        if self.model is None:
            return
        unwrapped = unwrap_model(as_module(self.model))
        if isinstance(unwrapped, EngramModel) and get_config_attr(
            unwrapped.config, "enable_telemetry"
        ):
            print("[Engram-PEFT] Capturing initial weights for telemetry...")
            groups = get_trainable_param_groups(unwrapped)
            # We only track drift for Engram groups to save memory
            for group_name in ["engram_dense", "engram_sparse"]:
                for p in groups.get(group_name, []):
                    self._initial_weights[id(p)] = p.detach().cpu().clone()

    @override
    def create_optimizer(self, model: Any = None) -> Optimizer:
        """
        Creates the MixedOptimizer if not provided.

        MixedOptimizer (SparseAdam + Adam) is skipped when using:
        - DeepSpeed: manages its own optimizer internally.
        - DDP / torchrun: DDP converts all gradients to dense during all_reduce,
          making SparseAdam unusable. A standard optimizer is used instead.
        """
        if self.optimizer is None:
            if _is_deepspeed_enabled(self.args) or _is_distributed():
                reason = (
                    "DeepSpeed"
                    if _is_deepspeed_enabled(self.args)
                    else "DDP (torchrun)"
                )
                print(
                    f"[Engram-PEFT] {reason} detected: skipping MixedOptimizer. "
                    + "DDP converts sparse gradients to dense during all_reduce, "
                    + "making SparseAdam unusable. A standard AdamW optimizer is used instead."
                )
                self.optimizer = super().create_optimizer(model)
            else:
                model = wash_model(self.model)
                if isinstance(model, EngramModel) and hasattr(
                    model, "create_optimizer"
                ):
                    self.optimizer = model.create_optimizer(
                        base_learning_rate=self.args.learning_rate,
                        **self.optimizer_kwargs,
                    )
                else:
                    # Non-Engram models (e.g. plain HF models or LoRA baselines) should
                    # use the standard Transformers optimizer creation path.
                    self.optimizer = super().create_optimizer(model)
        assert self.optimizer is not None
        return self.optimizer

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optimizer | None = None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Creates the scheduler. Uses standard Transformers schedulers if requested,
        otherwise falls back to Engram-specific Step Decay.
        """
        if self.lr_scheduler is None:
            # If the user explicitly requested a specific scheduler type (like cosine or wsd), use it.
            # We only use our custom Step Decay fallback if it's 'constant' or 'constant_with_warmup'.
            print(f"[Engram-PEFT] Creating scheduler: {self.args.lr_scheduler_type}")
            if self.args.lr_scheduler_type not in ["constant", "constant_with_warmup"]:
                return super().create_scheduler(num_training_steps, optimizer)

            if optimizer is None:
                optimizer = self.create_optimizer()

            model = wash_model(self.model)
            if isinstance(model, EngramModel) and hasattr(model, "create_scheduler"):
                self.lr_scheduler = model.create_scheduler(
                    optimizer,
                    num_training_steps=num_training_steps,
                    warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
            else:
                self.lr_scheduler = get_scheduler(
                    optimizer,
                    num_training_steps=num_training_steps,
                    warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
        assert self.lr_scheduler is not None
        return self.lr_scheduler

    @override
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """
        Overrides compute_loss to add an entropy penalty during training,
        but returns pure ce_loss during evaluation for fair comparison.
        """
        if num_items_in_batch is not None:
            ce_loss, ce_outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            ce_loss, ce_outputs = super().compute_loss(
                model, inputs, return_outputs=True
            )

        self._last_ce_loss = ce_loss.item()

        # Calculate Entropy Penalty
        unwrapped = unwrap_model(model)
        total_loss = ce_loss
        self._last_entropy_loss = 0.0

        if isinstance(unwrapped, EngramModel):
            alpha = float(
                get_config_attr(unwrapped.config, "entropy_loss_weight") or 0.0
            )
            if alpha > 0:
                entropy = unwrapped.get_total_gating_entropy()
                penalty = alpha * entropy
                self._last_entropy_loss = penalty.item()

                # IMPORTANT: Only add penalty to return loss during training
                if model.training:
                    total_loss = ce_loss + penalty
                else:
                    # In evaluation, we return the pure CE loss for fair baseline comparison
                    total_loss = ce_loss

        if return_outputs:
            if not isinstance(ce_outputs, dict):
                # Fallback for older transformers or unexpected return types
                return (total_loss, {"ce_outputs": ce_outputs})
            return (total_loss, as_dict(ce_outputs))

        return total_loss

    def _compute_total_norm(
        self, parameters: Iterable[nn.Parameter]
    ) -> torch.Tensor | None:
        """Computes the total gradient norm across dense and sparse parameters."""
        return compute_grad_norm(parameters)

    @override
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step. Injected for stage-wise training (backbone unfreezing).
        """
        if self.model is None:
            return super().training_step(
                model, inputs, num_items_in_batch=num_items_in_batch
            )
        unwrapped = unwrap_model(as_module(self.model))
        if isinstance(unwrapped, EngramModel):
            freeze_steps = int(
                get_config_attr(unwrapped.config, "backbone_freeze_steps") or 0
            )
            if freeze_steps > 0 and self.state.global_step == freeze_steps:
                print(
                    f"\n[Engram-PEFT] Step {freeze_steps}: Unfreezing backbone for joint training."
                )
                # Only unfreeze parameters that should be trainable according to the train_mode
                for name, param in unwrapped.base_model.named_parameters():
                    if name in unwrapped.trainable_backbone_names:
                        param.requires_grad = True

        try:
            # Try with num_items_in_batch (Transformers 4.46+)
            loss = super().training_step(
                model, inputs, num_items_in_batch=num_items_in_batch
            )
        except TypeError:
            # Fallback for older versions
            loss = super().training_step(model, inputs)

        # Capture telemetry AFTER backward but BEFORE optimizer zero_grad/step finishes
        # This ensures we capture gradients accurately.
        # Note: In Distributed/Gradient Accumulation, this runs every substep,
        # but _collect_telemetry is only logging-triggered.
        self._collect_telemetry()

        return loss

    @override
    def _clip_grad_norm(
        self, model: nn.Module, max_norm: float | None = None
    ) -> torch.Tensor | None:
        if max_norm is None:
            max_norm = cast("float | None", self.args.max_grad_norm)

        if max_norm is None or max_norm <= 0:
            return None

        unwrapped_model = unwrap_model(model)
        total_norm = self._compute_total_norm(model.parameters())

        # Standard Transformers Trainer behavior: if no gradients, return None early
        if total_norm is None:
            return None

        use_per_group = (
            unwrapped_model.config.clip_grad_per_group
            if isinstance(unwrapped_model, EngramModel)
            else False
        )

        if use_per_group:
            assert isinstance(unwrapped_model, EngramModel)
            groups = get_trainable_param_groups(unwrapped_model)
            apply_group_wise_clipping(groups, max_norm)
        else:
            # Standard Global Norm Clipping
            clip_coef = torch.tensor(max_norm, device=total_norm.device) / (
                total_norm + 1e-6
            )
            if clip_coef < 1.0:
                for p in iter_parameters(model):
                    if p.grad is not None:
                        g = p.grad
                        if g.is_sparse:
                            g.coalesce().values().detach().mul_(clip_coef.to(g.device))
                        else:
                            g.detach().mul_(clip_coef.to(g.device))
        return total_norm

    @override
    def _get_grad_norm(
        self, model: nn.Module, grad_norm: float | None = None
    ) -> torch.Tensor | None:
        """Override _get_grad_norm to avoid SparseCPU NotImplementedError during logging."""
        if grad_norm is not None:
            if torch.is_tensor(grad_norm):
                return grad_norm.detach().clone()
            return torch.tensor(grad_norm)

        return self._compute_total_norm(iter_parameters(model))

    def _collect_telemetry(self) -> dict[str, float]:
        """Performs deep telemetry collection across parameter groups."""
        if self.model is None:
            return {}
        unwrapped = unwrap_model(as_module(self.model))
        if not isinstance(unwrapped, EngramModel) or not get_config_attr(
            unwrapped.config, "enable_telemetry"
        ):
            return {}

        if (
            self.state.global_step == 0
            or self.state.global_step % self.args.logging_steps != 0
        ):
            return {}

        groups = get_trainable_param_groups(unwrapped)
        model_stats = unwrapped.get_telemetry_stats()

        return compute_telemetry_stats(
            groups=groups,
            initial_weights=self._initial_weights,
            model_telemetry_stats=model_stats,
            last_ce_loss=self._last_ce_loss,
            last_entropy_loss=self._last_entropy_loss,
        )

    @override
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Injects deep telemetry into the logs before they are sent to callbacks."""
        # Only collect telemetry if we are logging (i.e. at logging_steps)
        # We don't want to slow down every step
        extra_stats = self._collect_telemetry()
        logs.update(extra_stats)
        super().log(logs, start_time=start_time)

    @override
    def _save(
        self, output_dir: str | None = None, state_dict: dict[str, Any] | None = None
    ) -> None:
        """
        Override _save to delegate to EngramModel.save_pretrained() when the model
        is an EngramModel, avoiding safetensors shared-tensor errors from weight tying.

        Unwraps the model first (DDP / DeepSpeed wrapper) before checking isinstance,
        so the custom save path is used in distributed training too.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        dir_path: str = cast("str", output_dir)
        os.makedirs(dir_path, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        if isinstance(unwrapped, EngramModel):
            unwrapped.save_pretrained(dir_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(dir_path)
            elif self.data_collator is not None:
                tokenizer = getattr(self.data_collator, "tokenizer", None)
                if tokenizer is not None:
                    tokenizer.save_pretrained(dir_path)
            torch.save(self.args, os.path.join(dir_path, TRAINING_ARGS_NAME))
            return

        super()._save(dir_path, state_dict=state_dict)

    @override
    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: nn.Module | None = None
    ) -> None:
        """
        Override to handle composite checkpoints from EngramModel + PeftModel (LoRA).

        The base Trainer falls through to ``load_sharded_checkpoint`` when the model
        is neither a PreTrainedModel nor a PeftModel, but EngramModel wraps a PeftModel
        so neither branch triggers and the load fails.
        """
        if model is None:
            model = cast("nn.Module", self.model)
        unwrapped = self.accelerator.unwrap_model(model)

        if isinstance(unwrapped, EngramModel):
            base = unwrapped.base_model

            if _is_peft_model(base):
                if hasattr(base, "active_adapters") and hasattr(base, "load_adapter"):
                    active_adapters = base.active_adapters
                    if len(active_adapters) > 1:
                        logger.warning(
                            "Multiple active adapters detected; loading only the first."
                        )
                    active_adapter = active_adapters[0]
                    base.load_adapter(
                        resume_from_checkpoint, active_adapter, is_trainable=True
                    )
                else:
                    logger.warning(
                        "Base model is a PeftModel but lacks load_adapter/active_adapters."
                    )

            load_engram_weights(unwrapped, resume_from_checkpoint)
            return

        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
