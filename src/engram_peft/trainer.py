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
        self._initial_weights: dict[int, torch.Tensor] = {}
        if self.model is not None:
            self._capture_initial_weights()

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
                for p in groups[group_name]:
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

        stats: dict[str, float] = {}
        groups = get_trainable_param_groups(unwrapped)

        for group_name, params in groups.items():
            if not params:
                continue

            # 1. Parameter Stats
            all_p = []
            for p in params:
                all_p.append(p.detach().float().flatten())

            p_concat = torch.cat(all_p)
            p_norm = torch.norm(p_concat, 2).item()
            p_max = p_concat.abs().max().item()
            p_zero_rate = (p_concat.abs() < 1e-7).float().mean().item()

            stats[f"telemetry/{group_name}/param_norm"] = p_norm
            stats[f"telemetry/{group_name}/param_max"] = p_max
            stats[f"telemetry/{group_name}/param_zero_rate"] = p_zero_rate

            # 2. Gradient Stats
            all_g = []
            for p in params:
                if p.grad is not None:
                    g = p.grad
                    if g.is_sparse:
                        all_g.append(g.coalesce().values().detach().float().flatten())
                    else:
                        all_g.append(g.detach().float().flatten())

            if all_g:
                g_concat = torch.cat(all_g)
                g_norm = torch.norm(g_concat, 2).item()
                g_max = g_concat.abs().max().item()
                g_zero_rate = (g_concat.abs() < 1e-9).float().mean().item()

                stats[f"telemetry/{group_name}/grad_norm"] = g_norm
                stats[f"telemetry/{group_name}/grad_max"] = g_max
                stats[f"telemetry/{group_name}/grad_zero_rate"] = g_zero_rate
                stats[f"telemetry/{group_name}/grad_to_param_ratio"] = g_norm / (
                    p_norm + 1e-6
                )

            # 3. Weight Drift (relative to initial)
            if group_name in ["engram_dense", "engram_sparse"]:
                drift_norms = []
                init_norms = []
                for p in params:
                    p_id = id(p)
                    if p_id in self._initial_weights:
                        w0 = self._initial_weights[p_id].to(p.device)
                        drift_norms.append(
                            torch.norm(p.detach().float() - w0.float(), 2)
                        )
                        init_norms.append(torch.norm(w0.float(), 2))

                if drift_norms:
                    total_drift = torch.norm(torch.stack(drift_norms), 2).item()
                    total_init = torch.norm(torch.stack(init_norms), 2).item()
                    stats[f"telemetry/{group_name}/weight_drift"] = total_drift / (
                        total_init + 1e-6
                    )

        # 4. Activation Stats from Model
        model_stats = unwrapped.get_telemetry_stats()
        for k, v in model_stats.items():
            stats[f"telemetry/{k}"] = v

        return stats

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Injects deep telemetry into the logs before they are sent to callbacks."""
        # Only collect telemetry if we are logging (i.e. at logging_steps)
        # We don't want to slow down every step
        extra_stats = self._collect_telemetry()
        logs.update(extra_stats)
        super().log(logs, start_time=start_time)
