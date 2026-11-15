from langchain_core.messages import AIMessage

from src.agent.graph import build_graph
from src.evaluation.dataset import BenchmarkCase, load_benchmark
from src.evaluation.metrics import compute_metrics
from src.evaluation.runner import CaseResult, run_case


class FakeLLM:
    """Same test double pattern as tests/test_agent.py: canned responses in
    order, no real model call needed."""

    def __init__(self, responses):
        self._responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._responses.pop(0)


class CrashingLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise RuntimeError("model backend unreachable")


def _tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


BENCHMARK_PATH = "experiments/benchmarks/saree_shopping_v1.json"


def test_load_benchmark_case_count_and_categories():
    cases = load_benchmark(BENCHMARK_PATH)
    assert 30 <= len(cases) <= 50

    categories = {c.category for c in cases}
    assert "budget" in categories
    assert "conversational_followup" in categories


def test_load_benchmark_normalizes_query_to_single_turn():
    cases = load_benchmark(BENCHMARK_PATH)
    single_turn = next(c for c in cases if c.id == "color_001")
    assert single_turn.turns == ["Do you have maroon sarees?"]


def test_load_benchmark_marks_known_capability_gap_cases():
    cases = {c.id: c for c in load_benchmark(BENCHMARK_PATH)}
    gap_ids = {"bandha_003", "thread_003", "budget_003", "budget_004", "detail_003"}

    for case_id in gap_ids:
        assert cases[case_id].capability_gap is True

    non_gap_case = cases["color_001"]
    assert non_gap_case.capability_gap is False


def test_load_benchmark_keeps_multi_turn_cases():
    cases = load_benchmark(BENCHMARK_PATH)
    followup = next(c for c in cases if c.id == "followup_001")
    assert len(followup.turns) == 2


def test_run_case_correct_tool_selection_and_success():
    case = BenchmarkCase(
        id="t1",
        category="color",
        turns=["show me maroon sarees"],
        expects_tool_call=True,
        expected_tool="search_products",
        expected_constraints={"color": "Maroon"},
    )
    llm = FakeLLM(
        [
            _tool_call("search_products", {"color": "Maroon"}),
            AIMessage(content="Here you go."),
        ]
    )
    graph = build_graph(llm=llm)

    result = run_case(graph, case)

    assert result.tool_selection_correct is True
    assert result.argument_match_ratio == 1.0
    assert result.tool_call_valid is True
    assert result.task_success is True
    assert result.error is None
    assert result.tool_calls[0]["args"] == {"color": "Maroon"}
    assert result.latency_seconds >= 0


def test_run_case_detects_wrong_tool():
    case = BenchmarkCase(
        id="t2",
        category="budget",
        turns=["budget 5000"],
        expects_tool_call=True,
        expected_tool="recommend_products",
    )
    llm = FakeLLM(
        [
            _tool_call("search_products", {}),
            AIMessage(content="Here you go."),
        ]
    )
    graph = build_graph(llm=llm)

    result = run_case(graph, case)

    assert result.tool_selection_correct is False


def test_run_case_detects_missing_expected_tool_call():
    case = BenchmarkCase(
        id="t3",
        category="occasion",
        turns=["wedding sarees please"],
        expects_tool_call=True,
        expected_tool="search_products",
    )
    llm = FakeLLM([AIMessage(content="Sure, what colors do you like?")])
    graph = build_graph(llm=llm)

    result = run_case(graph, case)

    assert result.tool_selection_correct is False


def test_run_case_no_tool_call_expected_and_none_made():
    case = BenchmarkCase(id="t4", category="ambiguous", turns=["hi"], expects_tool_call=False)
    llm = FakeLLM([AIMessage(content="Hello! How can I help?")])
    graph = build_graph(llm=llm)

    result = run_case(graph, case)

    assert result.tool_selection_correct is True
    assert result.task_success is True


def test_run_case_partial_argument_match():
    case = BenchmarkCase(
        id="t5",
        category="multi_constraint",
        turns=["cotton wedding saree"],
        expects_tool_call=True,
        expected_tool="search_products",
        expected_constraints={"fabric": "Cotton", "occasion": "Wedding"},
    )
    llm = FakeLLM(
        [
            _tool_call("search_products", {"fabric": "Cotton", "occasion": "Festival"}),
            AIMessage(content="Here you go."),
        ]
    )
    graph = build_graph(llm=llm)

    result = run_case(graph, case)

    assert result.argument_match_ratio == 0.5


def test_run_case_flags_invalid_tool_result():
    case = BenchmarkCase(
        id="t6",
        category="thread_count",
        turns=["thread count -5 please"],
        expects_tool_call=True,
        expected_tool="search_products",
    )
    llm = FakeLLM(
        [
            _tool_call("search_products", {"thread_count": -5}),
            AIMessage(content="Sorry, that's not valid."),
        ]
    )
    graph = build_graph(llm=llm)

    result = run_case(graph, case)

    assert result.tool_call_valid is False
    assert result.task_success is False


def test_run_case_records_execution_error():
    case = BenchmarkCase(id="t7", category="ambiguous", turns=["hi"], expects_tool_call=False)
    graph = build_graph(llm=CrashingLLM())

    result = run_case(graph, case)

    assert result.error == "model backend unreachable"
    assert result.task_success is False


def _case_result(**overrides) -> CaseResult:
    defaults = dict(
        case_id="x", category="color", turns=["x"], expects_tool_call=True,
        expected_tool="search_products", expected_constraints=None, capability_gap=False,
        tool_calls=[], final_response="ok", tool_selection_correct=True,
        argument_match_ratio=None, tool_call_valid=None, task_success=True,
        latency_seconds=1.0, error=None,
    )
    defaults.update(overrides)
    return CaseResult(**defaults)


def test_compute_metrics_aggregates_across_results():
    results = [
        _case_result(
            case_id="a", category="color", expected_constraints={"color": "Maroon"},
            tool_calls=[{"name": "search_products", "args": {"color": "Maroon"}}],
            tool_selection_correct=True, argument_match_ratio=1.0, tool_call_valid=True,
            task_success=True, latency_seconds=0.5,
        ),
        _case_result(
            case_id="b", category="budget", expected_tool="recommend_products",
            expected_constraints={"max_price": 5000},
            tool_calls=[{"name": "search_products", "args": {}}],
            tool_selection_correct=False, argument_match_ratio=None, tool_call_valid=True,
            task_success=False, latency_seconds=1.5,
        ),
        _case_result(
            case_id="c", category="ambiguous", expects_tool_call=False, expected_tool=None,
            final_response=None, tool_selection_correct=False, tool_call_valid=None,
            task_success=False, latency_seconds=1.0, error="boom",
        ),
    ]

    metrics = compute_metrics(results)

    assert metrics["total_cases"] == 3
    assert metrics["graded_cases"] == 3
    assert metrics["capability_gap_cases"] == 0
    assert round(metrics["tool_selection_accuracy"], 4) == round(1 / 3, 4)
    assert metrics["mean_argument_match_ratio"] == 1.0
    assert round(metrics["execution_failure_rate"], 4) == round(1 / 3, 4)
    assert metrics["latency_seconds"]["mean"] == 1.0
    assert metrics["by_category"]["color"]["task_success_rate"] == 1.0
    assert metrics["by_category"]["budget"]["task_success_rate"] == 0.0
    assert "capability_gap_summary" not in metrics


def test_compute_metrics_separates_capability_gap_cases():
    results = [
        _case_result(case_id="a", tool_selection_correct=True, task_success=True),
        _case_result(
            case_id="b", capability_gap=True, tool_selection_correct=False, task_success=False,
        ),
    ]

    metrics = compute_metrics(results)

    assert metrics["graded_cases"] == 1
    assert metrics["capability_gap_cases"] == 1
    # the capability-gap case must not drag down the graded accuracy numbers
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["task_success_rate"] == 1.0
    assert metrics["capability_gap_summary"]["case_ids"] == ["b"]
    assert metrics["capability_gap_summary"]["task_success_rate"] == 0.0


def test_compute_metrics_handles_empty_results():
    assert compute_metrics([]) == {"total_cases": 0}


def _retrieval_case(relevant, **extra):
    return BenchmarkCase(
        id="r1", category="retrieval", turns=["show maroon"], expects_tool_call=True,
        expected_tool="search_products", relevant_skus=relevant, **extra,
    )


def test_retrieval_scoring_exact_match_and_partial(product_factory):
    maroon = product_factory(status="ACTIVE", price=1000, inventory_status="AVAILABLE", primary_color="Maroon")
    blue = product_factory(status="ACTIVE", price=1000, inventory_status="AVAILABLE", primary_color="Blue")

    def run(args):
        llm = FakeLLM([_tool_call("search_products", args), AIMessage(content="done")])
        return run_case(build_graph(llm=llm), _retrieval_case([maroon]))

    exact = run({"colors": "maroon"})
    assert exact.returned_skus == [maroon] and exact.retrieval_exact is True
    assert exact.retrieval_precision == 1.0 and exact.retrieval_recall == 1.0

    too_much = run({})  # no filter: returns every in-stock saree
    assert maroon in too_much.returned_skus and blue in too_much.returned_skus
    assert too_much.retrieval_recall == 1.0 and too_much.retrieval_precision < 1.0
    assert too_much.retrieval_exact is False


def test_retrieval_expecting_nothing_requires_a_search_that_finds_nothing():
    case = _retrieval_case([])
    found_nothing = FakeLLM([_tool_call("search_products", {"design": "no such design anywhere"}), AIMessage(content="none")])
    r = run_case(build_graph(llm=found_nothing), case)
    assert r.returned_skus == [] and r.retrieval_exact is True

    never_searched = FakeLLM([AIMessage(content="I have nothing like that.")])
    r = run_case(build_graph(llm=never_searched), case)
    assert r.retrieval_exact is False  # an answer without searching must not count as correct


def test_sku_prefix_ignores_unrelated_products(product_factory):
    mine = product_factory(sku="PFX-ONE", status="ACTIVE", price=1000, inventory_status="AVAILABLE")
    other = product_factory(sku="OTHER-ONE", status="ACTIVE", price=1000, inventory_status="AVAILABLE")
    llm = FakeLLM([_tool_call("search_products", {}), AIMessage(content="ok")])
    r = run_case(build_graph(llm=llm), _retrieval_case([mine], sku_prefix="PFX-"))
    assert other not in r.returned_skus and r.returned_skus == [mine]
    assert r.retrieval_exact is True


def test_argument_aliases_and_retrieval_metrics_summary(product_factory):
    sku = product_factory(status="ACTIVE", price=1000, inventory_status="AVAILABLE", primary_color="Maroon")
    case = BenchmarkCase(
        id="r2", category="retrieval", turns=["maroon"], expects_tool_call=True,
        expected_tool="search_products", expected_constraints={"colors|color": "Maroon"}, relevant_skus=[sku],
    )
    llm = FakeLLM([_tool_call("search_products", {"color": "maroon"}), AIMessage(content="ok")])
    r = run_case(build_graph(llm=llm), case)
    assert r.argument_match_ratio == 1.0
    summary = compute_metrics([r])["retrieval"]
    assert summary["cases"] == 1 and summary["exact_set_match_rate"] == 1.0 and summary["failed_case_ids"] == []


def test_progress_is_saved_after_each_case_and_resumes(tmp_path):
    from src.evaluation.checkpoint import load_partial, save_partial
    from src.evaluation.runner import run_benchmark

    cases = [BenchmarkCase(id=f"c{i}", category="x", turns=["hi"], expects_tool_call=False) for i in range(3)]
    path = tmp_path / "run_partial.json"
    saved = []

    def progress(results):
        save_partial(path, "run", "bench.json", results)
        saved.append(len(results))

    first = run_benchmark(build_graph(llm=FakeLLM([AIMessage(content="a")] * 3)), cases, on_result=progress)
    assert saved == [1, 2, 3] and len(first) == 3

    # a run that died after two cases: only the third is computed again
    save_partial(path, "run", "bench.json", first[:2])
    done = load_partial(path, "bench.json")
    assert set(done) == {"c0", "c1"}
    llm = FakeLLM([AIMessage(content="only one call needed")])
    resumed = run_benchmark(build_graph(llm=llm), cases, done=done)
    assert [r.case_id for r in resumed] == ["c0", "c1", "c2"]
    assert resumed[2].final_response == "only one call needed" and not llm._responses


def test_partial_file_from_another_benchmark_or_corrupt_is_ignored(tmp_path):
    from src.evaluation.checkpoint import load_partial, save_partial

    path = tmp_path / "p.json"
    save_partial(path, "run", "bench_a.json", [])
    assert load_partial(path, "bench_b.json") == {}
    path.write_text("{broken", encoding="utf-8")
    assert load_partial(path, "bench_a.json") == {}
    assert load_partial(tmp_path / "missing.json", "bench_a.json") == {}
