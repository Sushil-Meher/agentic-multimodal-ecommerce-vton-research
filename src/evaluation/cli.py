import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..agent.graph import build_graph
from .dataset import load_benchmark
from .metrics import compute_metrics
from .runner import run_benchmark

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = REPO_ROOT / "experiments" / "benchmarks" / "saree_shopping_v1.json"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def main():
    parser = argparse.ArgumentParser(description="Run the agent benchmark and save results")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    os.environ["AGENT_INCLUDE_EVAL_FIXTURES"] = "1"  # lets the agent's tools see EVAL-* fixtures

    cases = load_benchmark(args.benchmark)
    graph = build_graph()  # uses LLM_PROVIDER / LLM_MODEL from env, same as the agent
    results = run_benchmark(graph, cases)
    metrics = compute_metrics(results)

    provider = os.getenv("LLM_PROVIDER", "unknown")
    model = os.getenv("LLM_MODEL", "unknown")
    run_name = args.run_name or f"{provider}_{model}".replace(":", "-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    payload = {
        "run_name": run_name,
        "timestamp": timestamp,
        "provider": provider,
        "model": model,
        "benchmark_file": str(args.benchmark),
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_name}_{timestamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"wrote {out_path}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
