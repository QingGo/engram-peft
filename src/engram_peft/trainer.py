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

        return (total_loss, ce_outputs) if return_outputs else total_loss

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

        # Guard: only run on logging steps to avoid significant training slowdown
        # Also skip step 0 as gradients are not yet populated/stable
        if (
            self.state.global_step == 0
            or self.state.global_step % self.args.logging_steps != 0
        ):
            return {}

        stats: dict[str, float] = {}
        groups = get_trainable_param_groups(unwrapped)
        backbone_grad_norm = 0.0
        engram_grads_list = []

        for group_name, params in groups.items():
            if not params:
                continue

            device = next(unwrapped.parameters()).device
            sum_sq_p = torch.tensor(0.0, device=device)
            max_p = torch.tensor(0.0, device=device)
            total_elements = 0
            total_zeros = 0.0

            for p in params:
                p_detached = p.detach().float()
                sum_sq_p += p_detached.pow(2).sum()
                max_p = torch.max(max_p, p_detached.abs().max())
                total_elements += p.numel()
                total_zeros += (p_detached.abs() < 1e-7).sum().item()

            p_norm = torch.sqrt(sum_sq_p).item()
            stats[f"telemetry/{group_name}/param_norm"] = p_norm
            stats[f"telemetry/{group_name}/param_max"] = max_p.item()
            stats[f"telemetry/{group_name}/param_zero_rate"] = total_zeros / max(
                1, total_elements
            )

            # 2. Gradient Stats - Iterative calculation
            sum_sq_g = torch.tensor(0.0, device=device)
            has_g = False
            for p in params:
                if p.grad is not None:
                    # Handle sparse or dense grads
                    g = p.grad.detach().float()
                    if g.is_sparse:
                        g = g.coalesce().values()
                    sum_sq_g += g.pow(2).sum()
                    has_g = True

            if has_g:
                g_norm = torch.sqrt(sum_sq_g).item()
                stats[f"telemetry/{group_name}/grad_norm"] = g_norm
                if group_name == "backbone":
                    backbone_grad_norm = g_norm
                elif "engram" in group_name:
                    # Engram groups are small enough to cat if we really need conflict analysis,
                    # but let's keep it safe and just store the already computed norm
                    engram_grads_list.append(g_norm)

        # 3. Weight Drift - (Keep as-is or optimize if needed, usually smaller)
        # Weight drift is skipped for backbone to save time/memory
        # 3. Weight Drift - Engram only
        # We compare current weights to the CPU-cached initial weights
        for group_name, params in groups.items():
            if group_name != "backbone":
                total_drift = 0.0
                drift_count = 0
                for p in params:
                    pid = id(p)
                    if pid in self._initial_weights:
                        init_p = self._initial_weights[pid]
                        # Calculate on CPU or ensures it's safe
                        total_drift += torch.norm(
                            p.detach().cpu().float() - init_p.float(), 2
                        ).item()
                        drift_count += 1
                if drift_count > 0:
                    stats[f"telemetry/{group_name}/weight_drift"] = (
                        total_drift / drift_count
                    )

        # 4. Deep Diagnostic: Gradient Analysis
        stats["telemetry/diagnostics/heartbeat"] = 1.0
        if engram_grads_list:
            # We already stored g_norm (floats) in the list to save space
            # Total L2 norm of concatenated vectors = sqrt(sum(norms^2))
            total_e_norm = (sum(ng**2 for ng in engram_grads_list)) ** 0.5
            if backbone_grad_norm > 0:
                stats["telemetry/diagnostics/grad_norm_ratio"] = total_e_norm / (
                    backbone_grad_norm + 1e-8
                )

        # 5. Model Activation Stats (Entropy, Contribution)
        model_stats = unwrapped.get_telemetry_stats()
        for k, v in model_stats.items():
            stats[f"telemetry/{k}"] = v

        # 6. Fair Loss Stats
        stats["telemetry/loss/ce"] = self._last_ce_loss
        stats["telemetry/loss/entropy_penalty"] = self._last_entropy_loss

        return stats

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Injects deep telemetry into the logs before they are sent to callbacks."""
        # Only collect telemetry if we are logging (i.e. at logging_steps)
        # We don't want to slow down every step
        extra_stats = self._collect_telemetry()
        logs.update(extra_stats)
        super().log(logs, start_time=start_time)
