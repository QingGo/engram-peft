import json
import os
from datetime import datetime
from typing import Any


class BenchmarkResult:
    """Container for benchmark metadata and metrics."""

    def __init__(
        self,
        method: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        timestamp: str | None = None,
    ):
        self.method = method
        self.params = params
        self.metrics = metrics
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        return cls(
            method=str(data["method"]),
            params=dict(data["params"]),
            metrics=dict(data["metrics"]),
            timestamp=str(data.get("timestamp", "N/A")),
        )

    def save(self, base_dir: str = "outputs/benchmarks") -> str:
        """Convenience method to save the result."""
        os.makedirs(base_dir, exist_ok=True)
        filename = f"{self.method}_{self.timestamp}.json"
        path = os.path.join(base_dir, filename)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


class ResultManager:
    def __init__(self, base_dir: str = "outputs/benchmarks"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, result: BenchmarkResult) -> str:
        filename = f"{result.method}_{result.timestamp}.json"
        path = os.path.join(self.base_dir, filename)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        return path

    def load_all(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        if not os.path.exists(self.base_dir):
            return results

        for filename in sorted(os.listdir(self.base_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(self.base_dir, filename)) as f:
                    try:
                        results.append(BenchmarkResult.from_dict(json.load(f)))
                    except Exception as e:
                        print(f"Warning: Failed to load {filename}: {e}")
        return results

    def get_latest_by_method(self) -> dict[str, BenchmarkResult]:
        all_results = self.load_all()
        latest: dict[str, BenchmarkResult] = {}
        for r in all_results:
            latest[r.method] = r
        return latest
