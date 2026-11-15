import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..agent.graph import build_graph
from .checkpoint import load_partial, save_partial
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
    parser.add_argument(
        "--real-catalog",
        action="store_true",
        help="test against the imported (DRAFT) saree catalog instead of the EVAL-* fixtures",
    )
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run of the same name")
    args = parser.parse_args()

    if args.real_catalog:
        os.environ["AGENT_INCLUDE_DRAFT"] = "1"  # unpublished sarees become searchable for this run
    else:
        os.environ["AGENT_INCLUDE_EVAL_FIXTURES"] = "1"  # lets the agent's tools see EVAL-* fixtures

    cases = load_benchmark(args.benchmark)
    graph = build_graph()  # uses LLM_PROVIDER / LLM_MODEL from env, same as the agent
    provider = os.getenv("LLM_PROVIDER", "unknown")
    model = os.getenv("LLM_MODEL", "unknown")
    run_name = args.run_name or f"{provider}_{model}".replace(":", "-")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = RESULTS_DIR / f"{run_name}_partial.json"
    done = load_partial(partial_path, str(args.benchmark)) if args.resume else {}
    if done:
        print(f"resuming: {len(done)} of {len(cases)} cases already finished", flush=True)

    def progress(results):
        save_partial(partial_path, run_name, str(args.benchmark), results)
        print(f"saved progress: {len(results)}/{len(cases)} cases", flush=True)

    results = run_benchmark(graph, cases, on_result=progress, done=done)
    metrics = compute_metrics(results)
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

    out_path = RESULTS_DIR / f"{run_name}_{timestamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    partial_path.unlink(missing_ok=True)  # the full result file now holds everything

    print(f"wrote {out_path}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
