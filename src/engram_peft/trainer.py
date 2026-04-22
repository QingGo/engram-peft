from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from transformers import Trainer
from transformers.modeling_utils import unwrap_model

from engram_peft.model import EngramModel
from engram_peft.utils import (
    apply_group_wise_clipping,
    compute_grad_norm,
    compute_telemetry_stats,
    get_scheduler,
    get_trainable_param_groups,
)


class EngramTrainer(Trainer):
    """
    Custom Trainer that handles sparse gradient clipping and norm calculation.

    Standard torch.nn.utils.clip_grad_norm_ (and by extension accelerate's default implementation
    used in SFTTrainer) does not support SparseCPU/SparseCUDA tensors. This trainer
    overrides the clipping logic to correctly handle sparse parameters, making it the
    preferred choice when `use_sparse_embeddings=True` is set in the EngramConfig.
    """

    def __init__(
        self,
        *args: Any,
        optimizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer_kwargs = optimizer_kwargs or {}
        self._initial_weights: dict[int, torch.Tensor] = {}
        # Metrics for Fair Comparison
        self._last_ce_loss: float = 0.0
        self._last_entropy_loss: float = 0.0

        if self.model is not None:
            self._handle_initial_freezing()
            self._capture_initial_weights()

    def _handle_initial_freezing(self) -> None:
        """Initially freezes the backbone if backbone_freeze_steps > 0."""
        if self.model is None:
            return
        unwrapped = unwrap_model(self.model)
        if not isinstance(unwrapped, EngramModel):
            return

        freeze_steps = getattr(unwrapped.config, "backbone_freeze_steps", 0)
        if freeze_steps > 0:
            print(
                f"[Engram-PEFT] Adapter-First Mode: Freezing backbone for {freeze_steps} steps."
            )
            for param in unwrapped.base_model.parameters():
                param.requires_grad = False

    def _capture_initial_weights(self) -> None:
        """Captures initial weights of Engram modules on CPU for drift calculation."""
        if self.model is None:
            return
        unwrapped = unwrap_model(self.model)
        if isinstance(unwrapped, EngramModel) and unwrapped.config.enable_telemetry:
            print("[Engram-PEFT] Capturing initial weights for telemetry...")
            groups = get_trainable_param_groups(unwrapped)
            # We only track drift for Engram groups to save memory
            for group_name in ["engram_dense", "engram_sparse"]:
                for p in groups.get(group_name, []):
                    self._initial_weights[id(p)] = p.detach().cpu().clone()

    def create_optimizer(self, model: Any = None) -> Optimizer:
        """
        Creates the MixedOptimizer if not provided.
        """
        if self.optimizer is None:
            # Check if model has the helper

            if isinstance(self.model, EngramModel) and hasattr(
                self.model, "create_optimizer"
            ):
                self.optimizer = self.model.create_optimizer(
                    base_learning_rate=self.args.learning_rate,
                    **self.optimizer_kwargs,
                )
            else:
                # Non-Engram models (e.g. plain HF models or LoRA baselines) should
                # use the standard Transformers optimizer creation path.
                self.optimizer = super().create_optimizer(model)
        return self.optimizer

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

            if isinstance(self.model, EngramModel) and hasattr(
                self.model, "create_scheduler"
            ):
                self.lr_scheduler = self.model.create_scheduler(
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
        return self.lr_scheduler

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
            alpha = getattr(unwrapped.config, "entropy_loss_weight", 0.0)
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
            return (total_loss, ce_outputs)

        return total_loss

    def _compute_total_norm(self, parameters: Any) -> torch.Tensor | None:
        """Computes the total gradient norm across dense and sparse parameters."""
        return compute_grad_norm(parameters)

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
        unwrapped = unwrap_model(self.model)
        if isinstance(unwrapped, EngramModel):
            freeze_steps = getattr(unwrapped.config, "backbone_freeze_steps", 0)
            if freeze_steps > 0 and self.state.global_step == freeze_steps:
                print(
                    f"\n[Engram-PEFT] Step {freeze_steps}: Unfreezing backbone for joint training."
                )
                for param in unwrapped.base_model.parameters():
                    param.requires_grad = True

        try:
            loss = super().training_step(
                model, inputs, num_items_in_batch=num_items_in_batch
            )
        except TypeError:
            loss = super().training_step(model, inputs)

        # Capture telemetry AFTER backward but BEFORE optimizer zero_grad/step finishes
        # This ensures we capture gradients accurately.
        # Note: In Distributed/Gradient Accumulation, this runs every substep,
        # but _collect_telemetry is only logging-triggered.
        self._collect_telemetry()

        return loss

    def _clip_grad_norm(
        self, model: nn.Module, max_norm: float | None = None
    ) -> torch.Tensor | None:
        if max_norm is None:
            max_norm = self.args.max_grad_norm

        if max_norm is None or max_norm <= 0:
            return None

        unwrapped_model = unwrap_model(model)
        total_norm = self._compute_total_norm(model.parameters())

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
            if total_norm is None:
                return None

            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        g = p.grad
                        if g.is_sparse:
                            g.coalesce().values().detach().mul_(clip_coef.to(g.device))
                        else:
                            g.detach().mul_(clip_coef.to(g.device))
        return total_norm

    def _get_grad_norm(
        self, model: nn.Module, grad_norm: float | None = None
    ) -> torch.Tensor | None:
        """Override _get_grad_norm to avoid SparseCPU NotImplementedError during logging."""
        if grad_norm is not None:
            return (
                torch.tensor(grad_norm)
                if not isinstance(grad_norm, torch.Tensor)
                else grad_norm
            )

        return self._compute_total_norm(model.parameters())

    def _collect_telemetry(self) -> dict[str, float]:
        """Performs deep telemetry collection across parameter groups."""
        if self.model is None:
            return {}
        unwrapped = unwrap_model(self.model)
        if (
            not isinstance(unwrapped, EngramModel)
            or not unwrapped.config.enable_telemetry
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

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Injects deep telemetry into the logs before they are sent to callbacks."""
        # Only collect telemetry if we are logging (i.e. at logging_steps)
        # We don't want to slow down every step
        extra_stats = self._collect_telemetry()
        logs.update(extra_stats)
        super().log(logs, start_time=start_time)
