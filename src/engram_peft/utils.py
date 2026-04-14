from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, overload

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.sparse_adam import SparseAdam
from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from engram_peft import EngramModel


class MixedOptimizer(Optimizer):
    """
    Utility optimizer that wraps multiple optimizers (e.g., SparseAdam and Adam).
    This is necessary because PyTorch does not support mixing sparse and dense
    parameters in a single Adam optimizer instance.
    """

    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers
        # Combine parameter groups for transparency and compatibility
        param_groups = []
        for i, opt in enumerate(optimizers):
            param_groups.extend(opt.param_groups)
        # We call super().__init__ to ensure the base Optimizer class
        # is correctly initialized (state, etc.)
        super().__init__(param_groups, {})

        # Ensure defaults are populated from sub-optimizers for transparency
        for opt in optimizers:
            self.defaults.update(opt.defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step, ensuring hyperparams are synced."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # IMPORTANT: Sync hyperparams from MixedOptimizer.param_groups to sub-optimizers.
        # This is CRITICAL when using Accelerate, which might deep-copy param_groups.
        current_idx = 0
        for opt in self.optimizers:
            for sub_pg in opt.param_groups:
                parent_pg = self.param_groups[current_idx]
                # Sync all key hyperparameters
                for key in ["lr", "weight_decay", "betas", "eps"]:
                    if key in parent_pg:
                        sub_pg[key] = parent_pg[key]
                current_idx += 1

        for i, opt in enumerate(self.optimizers):
            # Check if gradients exist in this sub-optimizer
            has_grads = False
            for pg in opt.param_groups:
                for p in pg["params"]:
                    if p.grad is not None:
                        has_grads = True
                        break
                if has_grads:
                    break

            if has_grads:
                opt.step()

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clears the gradients of all optimized parameters."""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dict."""
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the optimizer state."""
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


def get_optimizer(
    model: "EngramModel", base_learning_rate: float = 4e-4
) -> MixedOptimizer:
    """
    Creates the optimizer for Engram PEFT according to paper specifications.
    Uses SparseAdam for Engram embeddings and Adam for other optimized parameters.

    Args:
        model: The EngramModel to optimize.
        base_learning_rate: The base learning rate (default: 4e-4).

    Returns:
        MixedOptimizer: A wrapper containing SparseAdam and Adam optimizers.
    """
    embedding_params = []
    other_params = []

    # Filter parameters from engram_layers (base model is frozen)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Identify the multi-head embedding table which uses sparse gradients
        if "multi_head_embedding.embedding.weight" in name:
            embedding_params.append(param)
        else:
            other_params.append(param)

    # Safe retrieval of Engram-specific config attributes
    lr_multiplier = getattr(model.config, "learning_rate_multiplier", 1.0)
    config_weight_decay = getattr(model.config, "weight_decay", 0.0)

    # Engram embedding parameters: scaled LR, no weight decay
    embedding_group = {
        "params": embedding_params,
        "lr": base_learning_rate * lr_multiplier,
        "weight_decay": 0.0,
    }

    # Other Engram parameters (convolution, gating): base LR, config-defined weight decay
    other_group = {
        "params": other_params,
        "lr": base_learning_rate,
        "weight_decay": config_weight_decay,
    }

    optimizers: List[Optimizer] = []
    if embedding_params:
        optimizers.append(SparseAdam([embedding_group]))
    if other_params:
        optimizers.append(Adam([other_group]))

    return MixedOptimizer(optimizers)


def get_scheduler(
    optimizer: Optimizer, num_training_steps: int, warmup_steps: int = 0
) -> LambdaLR:
    """
    Returns the step decay learning rate scheduler used in the Engram paper.

    Schedule:
    - Linear warmup for initial steps.
    - Decay to 31.6% (10^-0.5) of max LR at 80% training progress.
    - Decay to 10% (10^-1.0) of max LR at 90% training progress.

    Args:
        optimizer: The optimizer to schedule.
        num_training_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.

    Returns:
        LambdaLR: The learning rate scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step) / float(max(1, num_training_steps))
        if progress > 0.9:
            return 0.1
        if progress > 0.8:
            return 0.316
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def get_warmup_hold_cosine_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_steps: int = 0,
    hold_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Returns a learning rate scheduler that has a linear warmup, followed by a
    period of constant peak learning rate (hold), followed by a cosine decay
    to a minimum learning rate.

    Args:
        optimizer: The optimizer to schedule.
        num_training_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.
        hold_steps: Number of steps to hold the peak learning rate.
        min_lr_ratio: The ratio of the minimum learning rate to the peak learning rate.

    Returns:
        LambdaLR: The learning rate scheduler.
    """
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        if current_step < warmup_steps + hold_steps:
            return 1.0

        # Cosine decay
        progress = float(current_step - warmup_steps - hold_steps) / float(
            max(1, num_training_steps - warmup_steps - hold_steps)
        )
        if progress > 1.0:
            return min_lr_ratio

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale cosine decay to reach min_lr_ratio instead of 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
