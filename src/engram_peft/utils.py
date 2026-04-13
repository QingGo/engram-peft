from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, overload

import torch
from torch.optim import Adam, Optimizer, SparseAdam
from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from engram_peft.model import EngramModel


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
        for opt in optimizers:
            param_groups.extend(opt.param_groups)

        # We don't call super().__init__ because we manage param_groups manually
        # to avoid double-referencing parameters in the base class.
        self.param_groups = param_groups
        self.defaults = {}
        for opt in optimizers:
            self.defaults.update(opt.defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for opt in self.optimizers:
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

    # Engram embedding parameters: scaled LR, no weight decay
    embedding_group = {
        "params": embedding_params,
        "lr": base_learning_rate * model.config.learning_rate_multiplier,
        "weight_decay": 0.0,
    }

    # Other Engram parameters (convolution, gating): base LR, config-defined weight decay
    other_group = {
        "params": other_params,
        "lr": base_learning_rate,
        "weight_decay": model.config.weight_decay,
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
