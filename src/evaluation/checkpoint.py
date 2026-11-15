"""Save benchmark progress after every case so an interrupted run can resume."""
import json
import os
from dataclasses import asdict
from pathlib import Path

from .runner import CaseResult


def save_partial(path: Path, run_name: str, benchmark_file: str, results: list[CaseResult]) -> None:
    payload = {"run_name": run_name, "benchmark_file": benchmark_file, "completed": [asdict(r) for r in results]}
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)  # never leaves a half-written file behind


def load_partial(path: Path, benchmark_file: str) -> dict[str, CaseResult]:
    """Finished cases by id. A partial file from a different benchmark is ignored."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if payload.get("benchmark_file") != benchmark_file:
        return {}
    return {item["case_id"]: CaseResult(**item) for item in payload.get("completed", [])}
