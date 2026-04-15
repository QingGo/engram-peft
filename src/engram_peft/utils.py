from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

import torch
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.optim.sparse_adam import SparseAdam

if TYPE_CHECKING:
    from engram_peft import EngramModel

OptimizerFactory = Callable[[List[Dict[str, Any]]], Optimizer]
OptimizerSpec = Union[str, Type[Optimizer], OptimizerFactory]


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


_OPTIMIZER_REGISTRY: Mapping[str, OptimizerFactory] = {
    "adam": lambda param_groups: Adam(param_groups),
    "adamw": lambda param_groups: AdamW(param_groups),
    "sgd": lambda param_groups: SGD(param_groups),
    "sparse_adam": lambda param_groups: SparseAdam(param_groups),
}


def _build_optimizer(
    spec: OptimizerSpec,
    param_groups: List[Dict[str, Any]],
) -> Optimizer:
    if isinstance(spec, str):
        if spec not in _OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unknown optimizer '{spec}'. Available built-ins: "
                f"{', '.join(sorted(_OPTIMIZER_REGISTRY))}"
            )
        return _OPTIMIZER_REGISTRY[spec](param_groups)

    if isinstance(spec, type) and issubclass(spec, Optimizer):
        optimizer_cls = cast(OptimizerFactory, spec)
        return optimizer_cls(param_groups)

    optimizer_factory = cast(OptimizerFactory, spec)
    return optimizer_factory(param_groups)


def get_trainable_param_groups(
    model: "EngramModel",
) -> Dict[str, List[torch.nn.Parameter]]:
    """
    Splits trainable parameters into backbone, Engram dense, and Engram sparse groups.
    """
    backbone_params = [
        param for _, param in model.base_model.named_parameters() if param.requires_grad
    ]
    engram_sparse_params = []
    engram_dense_params = []
    for name, param in model.engram_layers.named_parameters():
        if not param.requires_grad:
            continue
        if "multi_head_embedding.embedding.weight" in name:
            engram_sparse_params.append(param)
        else:
            engram_dense_params.append(param)

    return {
        "backbone": backbone_params,
        "engram_dense": engram_dense_params,
        "engram_sparse": engram_sparse_params,
    }


def get_optimizer(
    model: "EngramModel",
    base_learning_rate: float = 4e-4,
    backbone_learning_rate: Optional[float] = None,
    engram_dense_learning_rate: Optional[float] = None,
    engram_sparse_learning_rate: Optional[float] = None,
    backbone_weight_decay: Optional[float] = None,
    engram_dense_weight_decay: Optional[float] = None,
    engram_sparse_weight_decay: float = 0.0,
    backbone_optimizer: Optional[OptimizerSpec] = None,
    engram_dense_optimizer: OptimizerSpec = "adam",
    engram_sparse_optimizer: OptimizerSpec = "sparse_adam",
) -> MixedOptimizer:
    """
    Creates the optimizer for Engram PEFT according to paper specifications.
    Uses separate optimizer groups for backbone, Engram dense params, and
    Engram sparse embeddings.

    Args:
        model: The EngramModel to optimize.
        base_learning_rate: The base learning rate (default: 4e-4).
        backbone_learning_rate: Learning rate for backbone parameters.
        engram_dense_learning_rate: Learning rate for Engram dense params.
        engram_sparse_learning_rate: Learning rate for Engram sparse embeddings.
        backbone_weight_decay: Weight decay for backbone params.
        engram_dense_weight_decay: Weight decay for Engram dense params.
        engram_sparse_weight_decay: Weight decay for Engram sparse embeddings.
        backbone_optimizer: Optimizer spec for backbone params.
        engram_dense_optimizer: Optimizer spec for Engram dense params.
        engram_sparse_optimizer: Optimizer spec for Engram sparse params.

    Returns:
        MixedOptimizer: A wrapper containing SparseAdam and Adam optimizers.
    """
    # Safe retrieval of Engram-specific config attributes
    lr_multiplier = getattr(model.config, "learning_rate_multiplier", 1.0)
    config_weight_decay = getattr(model.config, "weight_decay", 0.0)
    groups = get_trainable_param_groups(model)

    if backbone_optimizer is None:
        backbone_optimizer = "adam"
    if backbone_learning_rate is None:
        backbone_learning_rate = base_learning_rate
    if engram_dense_learning_rate is None:
        engram_dense_learning_rate = base_learning_rate
    if engram_sparse_learning_rate is None:
        engram_sparse_learning_rate = base_learning_rate * lr_multiplier
    if backbone_weight_decay is None:
        backbone_weight_decay = config_weight_decay
    if engram_dense_weight_decay is None:
        engram_dense_weight_decay = config_weight_decay

    optimizers: List[Optimizer] = []
    if groups["engram_sparse"]:
        optimizers.append(
            _build_optimizer(
                engram_sparse_optimizer,
                [
                    {
                        "params": groups["engram_sparse"],
                        "lr": engram_sparse_learning_rate,
                        "weight_decay": engram_sparse_weight_decay,
                    }
                ],
            )
        )
    if groups["engram_dense"]:
        optimizers.append(
            _build_optimizer(
                engram_dense_optimizer,
                [
                    {
                        "params": groups["engram_dense"],
                        "lr": engram_dense_learning_rate,
                        "weight_decay": engram_dense_weight_decay,
                    }
                ],
            )
        )
    if groups["backbone"]:
        optimizers.append(
            _build_optimizer(
                backbone_optimizer,
                [
                    {
                        "params": groups["backbone"],
                        "lr": backbone_learning_rate,
                        "weight_decay": backbone_weight_decay,
                    }
                ],
            )
        )

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


def get_submodule_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Returns the submodule at the given dot-separated path.
    Example: get_submodule_by_path(model, "model.layers")
    """
    if not path:
        return model
    segments = path.split(".")
    curr = model
    for seg in segments:
        if not hasattr(curr, seg):
            raise AttributeError(f"Module {type(curr).__name__} has no attribute {seg}")
        curr = getattr(curr, seg)
    return curr


def find_largest_module_list(model: torch.nn.Module) -> Optional[str]:
    """
    Heuristically finds the largest nn.ModuleList in the model tree.
    Returns the dot-separated path to it.
    """
    candidates: List[Tuple[str, int]] = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            # We prefer ModuleLists that contain other Modules (standard for layers)
            if all(isinstance(m, torch.nn.Module) for m in module):
                candidates.append((name, len(module)))

    if not candidates:
        return None

    # Sort by length descending, then by name length (prefer shallower paths)
    candidates.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)
    return candidates[0][0]


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
