"""Helpers for the attribute filters (colours, blouse, design words). Pure functions, no DB access."""
import json
import os
import re
from functools import lru_cache

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
    for i, word in enumerate(designs):
        scope = " AND a.region = :region" if region else ""
        names = []
        for j, term in enumerate(expand_design(word)):
            names.append(f"a.value ILIKE :design{i}_{j} ESCAPE '\\'")
            params[f"design{i}_{j}"] = f"%{escape_like(term)}%"
        clauses.append(
            "EXISTS (SELECT 1 FROM catalog.product_attributes a WHERE a.product_id = p.id "
            f"AND a.attribute IN ('label', 'motif') AND ({' OR '.join(names)}){scope})"
        )
    if designs and region:
        params["region"] = region
    return clauses


@lru_cache(maxsize=1)
def _families() -> list[list[str]]:
    """Groups of colour words that count as one colour family, from the owner's file
    (COLOUR_FAMILIES_FILE). Without the file only plain 'contains' matching applies."""
    path = os.getenv("COLOUR_FAMILIES_FILE", r"D:\saree_dataset\colour_families.json")
    try:
        with open(path, encoding="utf-8") as f:
            groups = json.load(f)["families"]
    except (OSError, ValueError, KeyError):
        return []
    return [[w.lower() for w in words] for words in groups.values()]


def expand_colour(word: str) -> list[str]:
    """'brown' -> ['brown', 'coffee']. A word that contains a family word joins that family
    ('deep brown' is brown). 'red' also finds 'deep red' and 'light red' through plain contains."""
    terms = [word]
    for family in _families():
        if any(t in word for t in family):
            terms += [t for t in family if t not in terms]
    return terms


@lru_cache(maxsize=1)
def _glossary_groups() -> list[list[str]]:
    """Each glossary term with its alternative spellings (AGENT_GLOSSARY_FILE), lower-cased."""
    path = os.getenv("AGENT_GLOSSARY_FILE", r"D:\saree_dataset\glossary.json")
    try:
        with open(path, encoding="utf-8") as f:
            terms = json.load(f)["terms"]
    except (OSError, ValueError, KeyError):
        return []
    return [[t["term"].lower(), *[a.lower() for a in t.get("aliases", [])]] for t in terms]


def expand_design(word: str) -> list[str]:
    """'terracotta' -> ['teracotta', 'terracotta', 'terakota']: a customer's spelling of a store
    term finds the owner's spelling too."""
    for group in _glossary_groups():
        if word in group:
            return group
    return [word]


def _has(value: str | None, word: str) -> bool:
    return bool(value) and any(t in value.lower() for t in expand_colour(word))


def _shade(value: str, word: str) -> str:
    """The saree's own colour word, flagged when it is only a similar shade of what was asked."""
    return word if word in value.lower() else f"{value} (similar to {word})"


def is_literal(body: str | None, border: str | None, body_other: list[str], wanted: list[str]) -> bool:
    """True when every requested colour word literally appears (not just a similar shade)."""
    return all(any(w in (c or "").lower() for c in [body, border, *body_other]) for w in wanted)


def colour_match(body: str | None, border: str | None, body_other: list[str], wanted: list[str]):
    """Return (tier, note) or None when the saree does not fit the requested colours.

    One colour: it must be the main body colour.
    Several colours: each must appear in the body (main or minor colours) or the border.
    'best' when one of them is the main body colour, otherwise 'weaker' with an honest note.
    Similar shades from the colour-family file count and are named as such in the note.
    """
    if not wanted:
        return None
    if len(wanted) == 1:
        word = wanted[0]
        return ("best", f"{_shade(body, word)} is the main body colour") if _has(body, word) else None
    main = next((w for w in wanted if _has(body, w)), None)
    where = {}
    for word in wanted:
        if _has(body, word):
            where[word] = ("body", body)
        elif _has(border, word):
            where[word] = ("border", border)
        else:
            other = next((c for c in body_other if _has(c, word)), None)
            if other is None:
                return None
            where[word] = ("body", other)
    if main:
        rest = ", ".join(f"{_shade(where[w][1], w)} in the {where[w][0]}" for w in wanted if w != main)
        return "best", f"{_shade(body, main)} is the main body colour; {rest}"
    return "weaker", f"has {' and '.join(wanted)}, but the main body colour is {body or 'different'}"
