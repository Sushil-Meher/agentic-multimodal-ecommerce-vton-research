from statistics import mean, median

from .runner import CaseResult


def compute_metrics(results: list[CaseResult]) -> dict:
    total = len(results)
    if total == 0:
        return {"total_cases": 0}

    latencies = sorted(r.latency_seconds for r in results)
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]

    metrics = {
        "total_cases": total,
        "execution_failure_rate": mean(r.error is not None for r in results),
        "latency_seconds": {
            "mean": mean(latencies),
            "median": median(latencies),
            "p95": p95,
            "min": min(latencies),
            "max": max(latencies),
        },
    }

    # capability-gap cases test requirements no current tool can satisfy
    # (no motif filter, no min_price, SKU-only lookup, ...) - they must not
    # count as ordinary retrieval failures, so they're excluded from the
    # accuracy metrics below and reported separately instead
    graded = [r for r in results if not r.capability_gap]
    gap = [r for r in results if r.capability_gap]

    metrics["graded_cases"] = len(graded)
    metrics["capability_gap_cases"] = len(gap)

    if graded:
        arg_ratios = [r.argument_match_ratio for r in graded if r.argument_match_ratio is not None]
        validity_flags = [r.tool_call_valid for r in graded if r.tool_call_valid is not None]

        by_category = {}
        for cat in sorted({r.category for r in graded}):
            cat_results = [r for r in graded if r.category == cat]
            by_category[cat] = {
                "count": len(cat_results),
                "tool_selection_accuracy": mean(r.tool_selection_correct for r in cat_results),
                "task_success_rate": mean(r.task_success for r in cat_results),
            }

        metrics.update(
            {
                "tool_selection_accuracy": mean(r.tool_selection_correct for r in graded),
                "mean_argument_match_ratio": mean(arg_ratios) if arg_ratios else None,
                "tool_call_validity_rate": mean(validity_flags) if validity_flags else None,
                "task_success_rate": mean(r.task_success for r in graded),
                "by_category": by_category,
            }
        )

    if gap:
        metrics["capability_gap_summary"] = {
            "note": (
                "These cases test requirements the current tool layer cannot fully satisfy "
                "(e.g. no motif filter, no min_price, SKU-only lookup). Excluded from the "
                "accuracy metrics above so they are never counted as ordinary retrieval failures."
            ),
            "case_ids": [r.case_id for r in gap],
            "task_success_rate": mean(r.task_success for r in gap),
        }

    return metrics
