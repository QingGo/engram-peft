from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from examples.benchmarks.persistence import BenchmarkResult


def get_differential_params(results: list[BenchmarkResult]) -> dict[str, list[Any]]:
    """Identifies parameters that differ between the given results."""
    if not results:
        return {}

    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.params.keys())

    diff_params = {}
    for key in all_keys:
        values = []
        for r in results:
            val = r.params.get(key)
            # Handle unhashable types for set()
            val_repr: Any
            if isinstance(val, dict | list):
                val_repr = str(val)
            else:
                val_repr = val
            values.append(val_repr)

        # If there's more than one unique value, it's a differential param
        if len(set(values)) > 1:
            diff_params[key] = [r.params.get(key) for r in results]

    return diff_params


def plot_benchmark_comparison(
    results: list[BenchmarkResult],
    output_path: str = "outputs/benchmarks/comparison.png",
) -> None:
    """
    Plots training loss curves with adaptive legends and differential footnotes.
    """
    if not results:
        print("No results to plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(14, 8))

    # 1. Determine Adaptive Legend Labels
    method_counts: dict[str, int] = {}
    for r in results:
        method_counts[r.method] = method_counts.get(r.method, 0) + 1

    current_method_idx: dict[str, int] = {}
    legend_labels = []
    for r in results:
        if method_counts[r.method] > 1:
            idx = current_method_idx.get(r.method, 0) + 1
            current_method_idx[r.method] = idx
            label = f"{r.method.capitalize()}_{idx}"
        else:
            label = r.method.capitalize()
        legend_labels.append(label)

    # 2. Extract and Plot Log Histories
    # Separate base results for horizontal line
    base_results = [r for r in results if r.method == "base"]
    other_results = [r for r in results if r.method != "base"]
    other_labels = [
        label
        for r, label in zip(results, legend_labels, strict=False)
        if r.method != "base"
    ]

    # Plot baseline first so it's in the background
    for r in base_results:
        if "eval_loss" in r.metrics:
            plt.axhline(
                y=r.metrics["eval_loss"],
                color="crimson",
                linestyle="--",
                linewidth=2,
                label="Base Model (Zero-shot)",
                alpha=0.8,
                zorder=1,
            )

    for i, r in enumerate(other_results):
        history = r.metrics.get("log_history", [])
        if not history:
            continue

        df = pd.DataFrame(history)
        label = other_labels[i]

        # Plot training loss
        color = None
        if "loss" in df.columns:
            line = sns.lineplot(
                data=df,
                x="step",
                y="loss",
                label=label,
                alpha=0.8,
                linewidth=2,
            )
            color = line.get_lines()[-1].get_color()

        # Plot eval loss points if they exist
        if "eval_loss" in df.columns:
            eval_df = df.dropna(subset=["eval_loss"])
            if not eval_df.empty:
                # Use training line color if available, else let scatterplot pick
                sns.scatterplot(
                    data=eval_df,
                    x="step",
                    y="eval_loss",
                    color=color,
                    s=100,
                    marker="o",
                    edgecolor="white",
                    zorder=5,
                    legend=False,
                )

    # 3. Create Differential Footnote
    # Filter out CLI-only parameters that shouldn't appear in the footnote
    exclude_keys = {
        "methods",
        "files",
        "plot_only",
        "list",
        "wandb",
        "wandb_offline",
        "wandb_project",
        "wandb_entity",
    }

    def get_relevant_params(r: BenchmarkResult) -> dict[str, Any]:
        params = {k: v for k, v in r.params.items() if k not in exclude_keys}
        # Filter out None values to keep it clean
        params = {k: v for k, v in params.items() if v is not None}
        return params

    relevant_configs = [get_relevant_params(r) for r in results]

    # Identify "Standard" or "Common" parameters via Mode (most frequent value)
    all_keys: set[str] = set()
    for cfg in relevant_configs:
        all_keys.update(cfg.keys())

    common_params = {}
    for key in sorted(all_keys):
        values = []
        for cfg in relevant_configs:
            if key in cfg:
                val = cfg[key]
                values.append(str(val) if isinstance(val, list | dict) else val)

        if not values:
            continue

        # Find the most frequent value
        counts = Counter(values)
        most_common_val, freq = counts.most_common(1)[0]

        # If the most common value is used by nearly all methods that have the key
        # or it's used by the majority, we treat it as common.
        if freq > 1 or len(results) == 1:
            # We need to find the original non-string version of the value
            orig_val = None
            for cfg in relevant_configs:
                if key in cfg:
                    v = cfg[key]
                    v_repr = str(v) if isinstance(v, list | dict) else v
                    if v_repr == most_common_val:
                        orig_val = v
                        break
            common_params[key] = orig_val

    footnote_lines = []
    for i, label in enumerate(legend_labels):
        cfg = relevant_configs[i]
        # Only show parameters that deviate from the most common "Common" set
        diffs = {}
        for k, v in cfg.items():
            if k not in common_params:
                diffs[k] = v
            else:
                # Compare value with the common/mode value
                v_repr = str(v) if isinstance(v, list | dict) else v
                mode_repr = (
                    str(common_params[k])
                    if isinstance(common_params[k], list | dict)
                    else common_params[k]
                )
                if v_repr != mode_repr:
                    diffs[k] = v

        # Only add a footnote line if there are actual deviations
        if diffs:
            # Sort for deterministic output
            diff_str = ", ".join([f"{k}={v}" for k, v in sorted(diffs.items())])
            footnote_lines.append(f"{label}: {diff_str}")
        elif results[i].method == "base" and i == 0:
            # Always show at least one line for Base if it's the first result
            # but only if not everything is already in common.
            # Actually, if everything is common, just model name is enough.
            if "model" in cfg and "model" not in common_params:
                footnote_lines.append(f"{label}: model={cfg['model']}")

    footer_text = ""
    if footnote_lines:
        footer_text += "\n".join(footnote_lines) + "\n"

    if common_params:
        # Show all common params
        common_str = ", ".join([f"{k}={v}" for k, v in sorted(common_params.items())])
        footer_text += f"\nCommon: {common_str}"

    plt.title("Benchmarking Convergence Comparison", pad=20, fontweight="bold")
    plt.xlabel("Steps", labelpad=10)
    plt.ylabel("Loss (Training: Line, Eval: Dots)", labelpad=10)

    # Place legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # Adjust layout and bottom margin to make space for footnote
    plt.tight_layout()
    # Reserve significant bottom margin if footnote is present
    if footer_text:
        # Dynamic margin based on number of lines
        num_lines = footer_text.count("\n") + 1
        margin = min(0.1 + 0.03 * num_lines, 0.4)
        plt.subplots_adjust(bottom=margin)

        plt.figtext(
            0.05,
            0.02,
            footer_text,
            wrap=True,
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize=9,
            color="#333333",
            bbox=dict(
                facecolor="#F9F9F9",
                edgecolor="#E0E0E0",
                boxstyle="round,pad=0.8",
                alpha=0.95,
            ),
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {output_path}")
