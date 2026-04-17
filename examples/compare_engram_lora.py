"""
Engram-PEFT Modular Benchmark Wrapper.

Usage:
    # Run new experiments
    uv run python examples/compare_engram_lora.py --methods engram --max_steps 50

    # Just replot latest results
    uv run python examples/compare_engram_lora.py --plot_only

    # Compare specific historical runs
    uv run python examples/compare_engram_lora.py --plot_only --files file1.json file2.json
"""

import argparse
import json
import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from examples.benchmarks.engine import BenchmarkEngine
from examples.benchmarks.persistence import BenchmarkResult, ResultManager
from examples.benchmarks.plotting import plot_benchmark_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Engram Benchmarking Suite")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lora", "engram"],
        help="Methods to run. Can include overrides like 'engram:clip_grad_per_layer=True'",
    )

    parser.add_argument(
        "--plot_only", action="store_true", help="Don't run, just aggregate and plot"
    )
    parser.add_argument("--files", nargs="+", help="Explicit JSON files to plot")
    parser.add_argument(
        "--list", action="store_true", help="List all historical results"
    )

    # WandB Configuration
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases tracking"
    )
    parser.add_argument(
        "--wandb_offline", action="store_true", help="Run wandb in offline mode"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="engram-peft-benchmarks",
        help="WandB project name",
    )
    parser.add_argument("--wandb_entity", type=str, help="WandB entity/username")

    args = parser.parse_args()
    manager = ResultManager()

    if args.list:
        results = manager.load_all()
        print(f"\n{'Method':<15} | {'Timestamp':<20} | {'Steps':<6} | {'Eval Loss'}")
        print("-" * 60)
        for r in results:
            steps = r.params.get("max_steps", "N/A")
            loss = r.metrics.get("eval_loss", 0.0)
            print(f"{r.method:<15} | {r.timestamp:<20} | {steps:<6} | {loss:.4f}")
        return

    if args.plot_only:
        if args.files:
            # Load specific files
            results_to_plot = []
            for f in args.files:
                path = os.path.join(manager.base_dir, f) if not os.path.isabs(f) else f
                with open(path) as j:
                    results_to_plot.append(BenchmarkResult.from_dict(json.load(j)))
        else:
            # Load latest for each method
            latest_dict = manager.get_latest_by_method()
            results_to_plot = list(latest_dict.values())

        plot_benchmark_comparison(results_to_plot)
        return

    # Normal Run Mode
    # Use deterministic model name for now
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    engine = BenchmarkEngine(model_name, args)

    engine.run_all(args.methods)

    # Auto-plot after run
    latest_dict = manager.get_latest_by_method()
    plot_benchmark_comparison(list(latest_dict.values()))


if __name__ == "__main__":
    main()
