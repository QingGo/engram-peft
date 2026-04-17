from typing import Any, cast

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from transformers import Trainer
from transformers.modeling_utils import unwrap_model

from engram_peft.model import EngramModel
from engram_peft.utils import get_scheduler, get_trainable_param_groups


class EngramTrainer(Trainer):
    """
    Custom Trainer that handles sparse gradient clipping and norm calculation.

    Standard torch.nn.utils.clip_grad_norm_ (and by extension accelerate's default implementation)
    does not support SparseCPU tensors. This trainer overrides both clipping and norm
    calculation to correctly handle SparseCPU/SparseCUDA tensors.
    """

    def __init__(
        self,
        *args: Any,
        optimizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer_kwargs = optimizer_kwargs or {}

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

    def _compute_total_norm(self, parameters: Any) -> torch.Tensor | None:
        """Computes the total gradient norm across dense and sparse parameters."""
        grads = []
        for p in parameters:
            if p.grad is not None:
                if p.grad.is_sparse:
                    grads.append(p.grad.coalesce().values())
                else:
                    grads.append(p.grad)

        if not grads:
            return None

        # Compute the global L2 norm
        device = grads[0].device
        norms = [torch.norm(g.detach().float(), 2).to(device) for g in grads]
        return cast("torch.Tensor", torch.norm(torch.stack(norms), 2))

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Standard training step handles both old and new Transformers Trainer signatures."""
        try:
            return super().training_step(
                model, inputs, num_items_in_batch=num_items_in_batch
            )
        except TypeError:
            return super().training_step(model, inputs)

    def _clip_grad_norm(
        self, model: nn.Module, max_norm: float | None = None
    ) -> torch.Tensor | None:
        if max_norm is None:
            max_norm = self.args.max_grad_norm

        if max_norm is None or max_norm <= 0:
            return None

        # Compatibility with non-Engram models is preserved so EngramTrainer can act as a
        # robust replacement for Trainer in the benchmark suite (handling sparse grads if present).
        unwrapped_model = unwrap_model(model)
        # Compute the original total norm BEFORE clipping for accurate logging
        total_norm = self._compute_total_norm(model.parameters())

        use_per_group = (
            unwrapped_model.config.clip_grad_per_group
            if isinstance(unwrapped_model, EngramModel)
            else False
        )

        if use_per_group:
            # Group-wise clipping: each major group (Backbone, Engram) is clipped independently
            groups = get_trainable_param_groups(cast("EngramModel", unwrapped_model))
            for _group_name, params in groups.items():
                if not params:
                    continue

                group_norm = self._compute_total_norm(params)
                if group_norm is None:
                    continue

                clip_coef = max_norm / (group_norm + 1e-6)
                if clip_coef < 1.0:
                    for p in params:
                        if p.grad is not None:
                            g = p.grad
                            if g.is_sparse:
                                g.coalesce().values().detach().mul_(
                                    clip_coef.to(g.device)
                                )
                            else:
                                g.detach().mul_(clip_coef.to(g.device))

            # Return the cached original total norm for logging consistency
            return total_norm
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
