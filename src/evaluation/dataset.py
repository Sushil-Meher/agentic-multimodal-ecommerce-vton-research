import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkCase:
    id: str
    category: str
    turns: list[str]
    expects_tool_call: bool
    expected_tool: str | list[str] | None = None
    expected_constraints: dict | None = None
    relevant_skus: list[str] | None = None  # reserved for future Precision/Recall/NDCG metrics
    capability_gap: bool = False  # tests a requirement no current tool can satisfy
    notes: str = ""


def load_benchmark(path: str | Path) -> list[BenchmarkCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for item in data["cases"]:
        turns = item.get("turns") or [item["query"]]
        cases.append(
            BenchmarkCase(
                id=item["id"],
                category=item["category"],
                turns=turns,
                expects_tool_call=item["expects_tool_call"],
                expected_tool=item.get("expected_tool"),
                expected_constraints=item.get("expected_constraints"),
                relevant_skus=item.get("relevant_skus"),
                capability_gap=item.get("capability_gap", False),
                notes=item.get("notes", ""),
            )
        )
    return cases
