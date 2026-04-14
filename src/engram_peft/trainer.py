import torch
import torch.nn as nn
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

if TYPE_CHECKING:
    from torch import Tensor


class EngramTrainer(Trainer):
    """
    Custom Trainer that handles sparse gradient clipping and norm calculation.

    Standard torch.nn.utils.clip_grad_norm_ (and by extension accelerate's default implementation)
    does not support SparseCPU tensors. This trainer overrides both clipping and norm
    calculation to correctly handle SparseCPU/SparseCUDA tensors.
    """

    def _compute_total_norm(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Computes the total gradient norm across dense and sparse parameters."""
        grads = []
        for _, p in model.named_parameters():
            if p.grad is not None:
                if p.grad.is_sparse:
                    grads.append(p.grad.coalesce().values())
                else:
                    grads.append(p.grad)

        if not grads:
            return None

        # Compute the global L2 norm
        device = grads[0].device
        norms = [torch.norm(g.detach(), 2).to(device) for g in grads]
        return cast(torch.Tensor, torch.norm(torch.stack(norms), 2))

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
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
        self, model: nn.Module, max_norm: Optional[float] = None
    ) -> Optional[torch.Tensor]:
        if max_norm is None:
            max_norm = self.args.max_grad_norm

        if max_norm is None or max_norm <= 0:
            return None

        total_norm = self._compute_total_norm(model)
        if total_norm is None:
            return None

        # Apply in-place scaling if norm exceeds max_norm
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad
                    if g.is_sparse:
                        # For sparse tensors, we must scale the values directly
                        g.coalesce().values().detach().mul_(clip_coef.to(g.device))
                    else:
                        g.detach().mul_(clip_coef.to(g.device))

        return total_norm

    def _get_grad_norm(
        self, model: nn.Module, grad_norm: Optional[float] = None
    ) -> Optional[torch.Tensor]:
        """Override _get_grad_norm to avoid SparseCPU NotImplementedError during logging."""
        if grad_norm is not None:
            return (
                torch.tensor(grad_norm)
                if not isinstance(grad_norm, torch.Tensor)
                else grad_norm
            )

        return self._compute_total_norm(model)
