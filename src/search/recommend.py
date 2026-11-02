from sqlalchemy.engine import Connection

from .products import search_products

# points per matched preference; kept as plain constants so the scoring
# stays easy to read and explain
_WEIGHTS = {
    "occasion": 2,
    "color": 2,
    "bandha_type": 2,
    "fabric": 1,
    "thread_count": 1,
    "budget": 2,
}


def _matches(candidate: dict, field: str, value: str) -> bool:
    candidate_value = candidate.get(field)
    return bool(candidate_value) and candidate_value.lower() == value.lower()


def _score_candidate(
    candidate: dict,
    *,
    max_price: float | None,
    color: str | None,
    bandha_type: str | None,
    thread_count: int | None,
    occasion: str | None,
    fabric: str | None,
) -> tuple[float, str]:
    score = 0.0
    reasons = []

    if occasion and _matches(candidate, "occasion", occasion):
        score += _WEIGHTS["occasion"]
        reasons.append(f"{occasion.lower()} occasion match")
    if color and _matches(candidate, "primary_color", color):
        score += _WEIGHTS["color"]
        reasons.append(f"{color.lower()} color match")
    if bandha_type and _matches(candidate, "bandha_type", bandha_type):
        score += _WEIGHTS["bandha_type"]
        reasons.append(f"{bandha_type} bandha match")
    if fabric and _matches(candidate, "fabric", fabric):
        score += _WEIGHTS["fabric"]
        reasons.append(f"{fabric.lower()} fabric match")
    if thread_count is not None and candidate.get("thread_count") == thread_count:
        score += _WEIGHTS["thread_count"]
        reasons.append(f"{thread_count} thread count match")
    if max_price is not None:
        # candidates are already filtered to max_price by search_products,
        # so this just credits "within budget" as a matched preference
        score += _WEIGHTS["budget"]
        reasons.append("within budget")

    reason = "; ".join(reasons) if reasons else "matches your search"
    return score, reason


def rank_products(candidates: list[dict], **preferences) -> list[dict]:
    """Pure scoring step, no DB access, so ranking logic can be tested on its own."""
    scored = []
    for candidate in candidates:
        score, reason = _score_candidate(candidate, **preferences)
        scored.append({**dict(candidate), "score": score, "reason": reason})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def recommend_products(
    conn: Connection,
    *,
    max_price: float | None = None,
    color: str | None = None,
    bandha_type: str | None = None,
    thread_count: int | None = None,
    occasion: str | None = None,
    fabric: str | None = None,
    limit: int = 5,
    include_price: bool = True,
    include_eval_fixtures: bool = False,
) -> list[dict]:
    """Search for candidates, then rank them. Meant to sit behind an agent
    tool later, same as search_products."""
    if limit <= 0:
        raise ValueError("limit must be > 0")

    # only budget and availability are hard filters here; color/bandha/thread
    # count/occasion/fabric are soft preferences used for ranking below, not
    # for narrowing the candidate pool, so a near-match can still show up
    candidates = search_products(
        conn,
        max_price=max_price,
        in_stock_only=True,
        include_price=True,
        include_eval_fixtures=include_eval_fixtures,
    )

    ranked = rank_products(
        candidates,
        max_price=max_price,
        color=color,
        bandha_type=bandha_type,
        thread_count=thread_count,
        occasion=occasion,
        fabric=fabric,
    )

    if not include_price:
        for item in ranked:
            item.pop("price", None)
            item.pop("currency", None)

    return ranked[:limit]
