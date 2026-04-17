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
    diffs = get_differential_params(results)
    footnote_lines = []

    # Identify unique differences per experiment
    for i, label in enumerate(legend_labels):
        experiment_diffs = []
        for key in sorted(diffs.keys()):
            val = results[i].params.get(key)
            experiment_diffs.append(f"{key}={val}")

        if experiment_diffs:
            footnote_lines.append(f"{label}: " + ", ".join(experiment_diffs))

    # Identify common parameters
    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.params.keys())
    common_params = []
    for key in sorted(all_keys):
        if key not in diffs:
            common_params.append(f"{key}={results[0].params.get(key)}")

    footer_text = ""
    if footnote_lines:
        footer_text += "\n".join(footnote_lines) + "\n"
    if common_params:
        footer_text += (
            "Common: "
            + ", ".join(common_params[:5])
            + ("..." if len(common_params) > 5 else "")
        )

    plt.title("Benchmarking Convergence Comparison", pad=20, fontweight="bold")
    plt.xlabel("Steps", labelpad=10)
    plt.ylabel("Loss (Training: Line, Eval: Dots)", labelpad=10)

    # Place legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # Add footnote at the bottom
    if footer_text:
        plt.figtext(
            0.1,
            -0.05,
            footer_text,
            wrap=True,
            horizontalalignment="left",
            fontsize=10,
            bbox=dict(facecolor="none", edgecolor="gray", boxstyle="round,pad=0.5"),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {output_path}")
