"""Helpers for the attribute filters (colours, blouse, design words). Pure functions, no DB access."""
import re

BLOUSE_VALUES = {"plain", "blouse_work", "no_blouse"}
REGIONS = {"body", "border", "pallu"}
_SPLIT = re.compile(r"\s*(?:,|&|\+|/|\band\b)\s*")


def parse_terms(raw: str | None, limit: int = 3) -> list[str]:
    """'red and black' -> ['red', 'black']; also splits on commas, &, + and /."""
    if not raw:
        return []
    terms = []
    for term in _SPLIT.split(raw.lower().strip()):
        term = term.strip()
        if term and term not in terms:
            terms.append(term)
    return terms[:limit]


def escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def normalise_blouse(value: str) -> str:
    cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned not in BLOUSE_VALUES:
        raise ValueError("blouse must be plain, blouse work or no blouse")
    return cleaned


def normalise_region(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned not in REGIONS:
        raise ValueError("region must be body, border or pallu")
    return cleaned


def design_clauses(designs: list[str], region: str | None, params: dict) -> list[str]:
    """One EXISTS per design word; every word must match a label or motif (contains, any case)."""
    clauses = []
    for i, term in enumerate(designs):
        scope = " AND a.region = :region" if region else ""
        clauses.append(
            "EXISTS (SELECT 1 FROM catalog.product_attributes a WHERE a.product_id = p.id "
            f"AND a.attribute IN ('label', 'motif') AND a.value ILIKE :design{i} ESCAPE '\\'{scope})"
        )
        params[f"design{i}"] = f"%{escape_like(term)}%"
    if designs and region:
        params["region"] = region
    return clauses


def _has(value: str | None, word: str) -> bool:
    return bool(value) and word in value.lower()


def colour_match(body: str | None, border: str | None, body_other: list[str], wanted: list[str]):
    """Return (tier, note) or None when the saree does not fit the requested colours.

    One colour: it must be the main body colour.
    Several colours: each must appear in the body (main or minor colours) or the border.
    'best' when one of them is the main body colour, otherwise 'weaker' with an honest note.
    """
    if not wanted:
        return None
    if len(wanted) == 1:
        return ("best", f"{wanted[0]} is the main body colour") if _has(body, wanted[0]) else None
    main = next((w for w in wanted if _has(body, w)), None)
    where = {}
    for word in wanted:
        if _has(body, word):
            where[word] = "body"
        elif _has(border, word):
            where[word] = "border"  # described as the border even if it also appears as a minor body colour
        elif any(_has(c, word) for c in body_other):
            where[word] = "body"
        else:
            return None
    if main:
        rest = ", ".join(f"{w} in the {where[w]}" for w in wanted if w != main)
        return "best", f"{main} is the main body colour; {rest}"
    return "weaker", f"has {' and '.join(wanted)}, but the main body colour is {body or 'different'}"
