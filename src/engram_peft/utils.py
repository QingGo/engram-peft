import math
from collections.abc import Callable, Iterable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
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
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

if TYPE_CHECKING:
    from engram_peft import EngramModel

OptimizerFactory = Callable[[list[dict[str, Any]]], Optimizer]
OptimizerSpec = str | type[Optimizer] | OptimizerFactory


class MixedOptimizer(Optimizer):
    """
    Utility optimizer that wraps multiple optimizers (e.g., SparseAdam and Adam).
    This is necessary because PyTorch does not support mixing sparse and dense
    parameters in a single Adam optimizer instance.
    """

    def __init__(self, optimizers: list[Optimizer]):
        self.optimizers = optimizers
        # Combine parameter groups for transparency and compatibility
        param_groups = []
        for _i, opt in enumerate(optimizers):
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

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
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

        for _i, opt in enumerate(self.optimizers):
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

    def state_dict(self) -> dict[str, Any]:
        """Returns the state of the optimizer as a dict."""
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the optimizer state."""
        for opt, sd in zip(self.optimizers, state_dict["optimizers"], strict=False):
            opt.load_state_dict(sd)

    def unscale_(self, scaler: torch.cuda.amp.GradScaler) -> None:
        """Proxies unscale_ to all sub-optimizers for FP16 compatibility."""
        for opt in self.optimizers:
            if hasattr(scaler, "unscale_"):
                scaler.unscale_(opt)
            else:
                # Fallback for older versions or different scaler implementations
                pass


_OPTIMIZER_REGISTRY: Mapping[str, OptimizerFactory] = {
    "adam": lambda param_groups: Adam(param_groups),
    "adamw": lambda param_groups: AdamW(param_groups),
    "sgd": lambda param_groups: SGD(param_groups),
    "sparse_adam": lambda param_groups: SparseAdam(param_groups),
}


def _build_optimizer(
    spec: OptimizerSpec,
    param_groups: list[dict[str, Any]],
) -> Optimizer:
    if isinstance(spec, str):
        if spec not in _OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unknown optimizer '{spec}'. Available built-ins: "
                f"{', '.join(sorted(_OPTIMIZER_REGISTRY))}"
            )
        return _OPTIMIZER_REGISTRY[spec](param_groups)

    if isinstance(spec, type) and issubclass(spec, Optimizer):
        optimizer_cls = cast("OptimizerFactory", spec)
        return optimizer_cls(param_groups)

    optimizer_factory = cast("OptimizerFactory", spec)
    return optimizer_factory(param_groups)


def get_trainable_param_groups(
    model: "EngramModel",
) -> dict[str, list[torch.nn.Parameter]]:
    """
    Splits trainable parameters into backbone, Engram dense, and Engram sparse groups.
    """
    # Include backbone parameters if they are trainable OR if they will be unfrozen later
    freeze_steps = getattr(model.config, "backbone_freeze_steps", 0)
    backbone_params = [
        param
        for _, param in model.base_model.named_parameters()
        if param.requires_grad or freeze_steps > 0
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
    backbone_learning_rate: float | None = None,
    engram_dense_learning_rate: float | None = None,
    engram_sparse_learning_rate: float | None = None,
    backbone_weight_decay: float | None = None,
    engram_dense_weight_decay: float | None = None,
    engram_sparse_weight_decay: float = 0.0,
    backbone_optimizer: OptimizerSpec | None = None,
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
    # Use multipliers from config if specific LRs are not provided
    if engram_dense_learning_rate is None:
        engram_dense_learning_rate = base_learning_rate

    if engram_sparse_learning_rate is None:
        # We now trust the base multiplier for sparse as well, or we could add a sparse-specific multiplier to config
        engram_sparse_learning_rate = base_learning_rate * lr_multiplier

    if backbone_weight_decay is None:
        backbone_weight_decay = config_weight_decay
    if engram_dense_weight_decay is None:
        engram_dense_weight_decay = config_weight_decay

    optimizers: list[Optimizer] = []
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


# Migrated find_largest_module_list to discovery.py


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


def evaluate_model_loss(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Any,
    batch_size: int = 8,
    max_length: int = 128,
    output_dir: str = "outputs/eval",
) -> float:
    """
    Standardizes the calculation of Zero-shot loss for language models.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        dataset: The evaluation dataset.
        batch_size: Evaluation batch size.
        max_length: Maximum sequence length.
        output_dir: Directory for temporary evaluation outputs.

    Returns:
        float: The calculated eval_loss.
    """
    eval_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        and torch.cuda.is_available(),
        dataloader_num_workers=0,  # Avoid overhead for quick eval
        dataloader_pin_memory=True,
        remove_unused_columns=True,
    )

    # Use DataCollatorForLanguageModeling to ensure correct label shifting
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset,
        data_collator=data_collator,
    )

    metrics = trainer.evaluate()
    return cast("float", metrics.get("eval_loss", 0.0))


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor | None:
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
    # Ensure float32 for norm stability
    norms = [torch.norm(g.detach().float(), 2).to(device) for g in grads]
    return cast("torch.Tensor", torch.norm(torch.stack(norms), 2))


def apply_group_wise_clipping(
    groups: dict[str, list[torch.nn.Parameter]],
    max_norm: float,
) -> None:
    """Applies gradient clipping independently to each parameter group."""
    for _group_name, params in groups.items():
        if not params:
            continue

        group_norm = compute_grad_norm(params)
        if group_norm is None:
            continue

        clip_coef = max_norm / (group_norm + 1e-6)
        if clip_coef < 1.0:
            for p in params:
                if p.grad is not None:
                    g = p.grad
                    if g.is_sparse:
                        g.coalesce().values().detach().mul_(clip_coef.to(g.device))
                    else:
                        g.detach().mul_(clip_coef.to(g.device))


def compute_telemetry_stats(
    groups: dict[str, list[torch.nn.Parameter]],
    initial_weights: dict[int, torch.Tensor] = None,
    model_telemetry_stats: dict[str, float] = None,
    last_ce_loss: float = 0.0,
    last_entropy_loss: float = 0.0,
) -> dict[str, float]:
    """Computes parameter/gradient statistics and weight drift across groups."""
    stats: dict[str, float] = {}
    backbone_grad_norm = 0.0
    engram_grads_list = []

    for group_name, params in groups.items():
        if not params:
            continue

        device = next(iter(params)).device
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

        # 2. Gradient Stats
        sum_sq_g = torch.tensor(0.0, device=device)
        has_g = False
        for p in params:
            if p.grad is not None:
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
                engram_grads_list.append(g_norm)

    # 3. Weight Drift - Engram only
    if initial_weights:
        for group_name, params in groups.items():
            if group_name != "backbone":
                total_drift = 0.0
                drift_count = 0
                for p in params:
                    pid = id(p)
                    if pid in initial_weights:
                        init_p = initial_weights[pid]
                        total_drift += torch.norm(
                            p.detach().cpu().float() - init_p.float(), 2
                        ).item()
                        drift_count += 1
                if drift_count > 0:
                    stats[f"telemetry/{group_name}/weight_drift"] = (
                        total_drift / drift_count
                    )

    # 4. Diagnostics
    stats["telemetry/diagnostics/heartbeat"] = 1.0
    if engram_grads_list:
        total_e_norm = (sum(ng**2 for ng in engram_grads_list)) ** 0.5
        if backbone_grad_norm > 0:
            stats["telemetry/diagnostics/grad_norm_ratio"] = total_e_norm / (
                backbone_grad_norm + 1e-8
            )

    # 5. Model Activation Stats
    if model_telemetry_stats:
        for k, v in model_telemetry_stats.items():
            stats[f"telemetry/{k}"] = v

    # 6. Loss Stats
    stats["telemetry/loss/ce"] = last_ce_loss
    stats["telemetry/loss/entropy_penalty"] = last_entropy_loss

    return stats
