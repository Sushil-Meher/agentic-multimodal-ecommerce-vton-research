import json
import time
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .dataset import BenchmarkCase


@dataclass
class CaseResult:
    case_id: str
    category: str
    turns: list[str]
    expects_tool_call: bool
    expected_tool: object
    expected_constraints: object
    capability_gap: bool
    tool_calls: list
    final_response: str | None
    tool_selection_correct: bool
    argument_match_ratio: float | None
    tool_call_valid: bool | None
    task_success: bool
    latency_seconds: float
    error: str | None


def _values_match(expected, actual) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


def _score(case: BenchmarkCase, tool_calls: list[dict], tool_results: list):
    called_any_tool = len(tool_calls) > 0

    if case.expected_tool is None:
        expected_names = None
    elif isinstance(case.expected_tool, str):
        expected_names = {case.expected_tool}
    else:
        expected_names = set(case.expected_tool)

    if case.expects_tool_call:
        tool_selection_correct = called_any_tool and (
            expected_names is None or tool_calls[0]["name"] in expected_names
        )
    else:
        tool_selection_correct = not called_any_tool

    argument_match_ratio = None
    if called_any_tool and case.expected_constraints:
        args = tool_calls[0]["args"]
        matched = sum(
            1
            for key, value in case.expected_constraints.items()
            if key in args and _values_match(value, args[key])
        )
        argument_match_ratio = matched / len(case.expected_constraints)

    tool_call_valid = None
    if called_any_tool:
        tool_call_valid = all(not (isinstance(r, dict) and "error" in r) for r in tool_results)

    return tool_selection_correct, argument_match_ratio, tool_call_valid


def run_case(graph, case: BenchmarkCase) -> CaseResult:
    messages: list = []
    tool_calls: list[dict] = []
    tool_results: list = []
    final_response = None
    error = None

    start = time.perf_counter()
    try:
        for turn in case.turns:
            messages.append(HumanMessage(content=turn))
            result = graph.invoke({"messages": messages})
            messages = result["messages"]

        for m in messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                tool_calls.extend({"name": c["name"], "args": c["args"]} for c in m.tool_calls)
            elif isinstance(m, ToolMessage):
                try:
                    tool_results.append(json.loads(m.content))
                except (TypeError, ValueError):
                    tool_results.append(m.content)

        final_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None
        )
        final_response = final_ai.content if final_ai else None
    except Exception as exc:  # evaluation must record any agent failure, not just expected ones
        error = str(exc)

    latency = time.perf_counter() - start

    if error is not None:
        return CaseResult(
            case_id=case.id,
            category=case.category,
            turns=case.turns,
            expects_tool_call=case.expects_tool_call,
            expected_tool=case.expected_tool,
            expected_constraints=case.expected_constraints,
            capability_gap=case.capability_gap,
            tool_calls=tool_calls,
            final_response=final_response,
            tool_selection_correct=False,
            argument_match_ratio=None,
            tool_call_valid=None,
            task_success=False,
            latency_seconds=latency,
            error=error,
        )

    tool_selection_correct, argument_match_ratio, tool_call_valid = _score(
        case, tool_calls, tool_results
    )
    task_success = tool_selection_correct and tool_call_valid is not False and bool(final_response)

    return CaseResult(
        case_id=case.id,
        category=case.category,
        turns=case.turns,
        expects_tool_call=case.expects_tool_call,
        expected_tool=case.expected_tool,
        expected_constraints=case.expected_constraints,
        capability_gap=case.capability_gap,
        tool_calls=tool_calls,
        final_response=final_response,
        tool_selection_correct=tool_selection_correct,
        argument_match_ratio=argument_match_ratio,
        tool_call_valid=tool_call_valid,
        task_success=task_success,
        latency_seconds=latency,
        error=None,
    )


def run_benchmark(graph, cases: list[BenchmarkCase]) -> list[CaseResult]:
    return [run_case(graph, case) for case in cases]
