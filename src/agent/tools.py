import json
import os
from datetime import datetime
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from langchain_core.tools import tool
from pydantic import Field

from ..api.db import engine
from ..retrieval.search import EmbeddingServiceError
from ..retrieval.search import search_by_image as search_by_image_service
from ..search.details import get_product_details as get_product_details_service
from ..search.products import search_products as search_products_service
from ..search.recommend import recommend_products as recommend_products_service


def _json_default(value):
    if isinstance(value, (UUID, datetime)):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"not JSON serializable: {value!r}")


def _to_json(data) -> str:
    return json.dumps(data, default=_json_default)


def _include_eval_fixtures() -> bool:
    # the evaluation harness sets this so its own EVAL-* fixture catalog is
    # reachable through the real tools; real customers never set it
    return os.getenv("AGENT_INCLUDE_EVAL_FIXTURES") == "1"


def _include_draft() -> bool:
    # development only: lets the unpublished (DRAFT) catalog be searched while it is being tested
    return os.getenv("AGENT_INCLUDE_DRAFT") == "1"


# Optional filters use plain types with "" / 0 as "not given" instead of
# `X | None`: the Ollama client drops anyOf/null from tool schemas, so the
# model used to see every optional parameter as a blank `{}`.
Color = Annotated[
    str, Field(description="Primary color, e.g. Maroon. Omit if the customer didn't mention one.")
]
BandhaType = Annotated[
    str,
    Field(
        description="Full bandha style name, e.g. 'Traditional Bandha' or 'Pasapalli Bandha'. "
        "Omit if the customer didn't mention one."
    ),
]
ThreadCount = Annotated[
    int,
    Field(description="Thread count as a positive integer, e.g. 100. Omit if not mentioned."),
]
Occasion = Annotated[
    str, Field(description="Occasion, e.g. Wedding or Festival. Omit if not mentioned.")
]
Fabric = Annotated[str, Field(description="Fabric, e.g. Cotton or Silk. Omit if not mentioned.")]
MaxPrice = Annotated[
    float,
    Field(description="Budget ceiling in rupees. Omit if the customer didn't state a budget."),
]


MinPrice = Annotated[
    float,
    Field(description="Lowest price in rupees when the customer gives a range. Omit otherwise."),
]
Colors = Annotated[
    str,
    Field(
        description="Colour(s) the customer wants, e.g. 'red' or 'red and black'. One colour means the "
        "main body colour; with two colours both may be in the body, or one in the body and one in the "
        "border. Similar shades are included automatically (brown also finds coffee). Omit if no "
        "colour was mentioned."
    ),
]
Blouse = Annotated[
    str,
    Field(description="Blouse type: 'plain', 'blouse work' or 'no blouse'. Omit if not mentioned."),
]
Design = Annotated[
    str,
    Field(
        description="Pattern, motif or design words the customer asked for, such as 'milan', 'peacock', "
        "'teracotta' or 'bandha paper'. Separate several with 'and'. Omit if not mentioned."
    ),
]
Region = Annotated[
    str,
    Field(
        description="Only if the customer says where the design is: 'body', 'border' or 'pallu'. "
        "Omit otherwise."
    ),
]
BorderColor = Annotated[
    str,
    Field(
        description="Colour of the border bands only, e.g. 'red'. Use when the customer asks about the "
        "border colour; similar shades are included. Omit otherwise."
    ),
]
PalluColor = Annotated[
    str,
    Field(description="Colour of the pallu (the decorated end piece) only. Omit if not mentioned."),
]
ImageName = Annotated[
    str,
    Field(description="File name of the photo the customer uploaded, exactly as given to you."),
]


# the model only needs what a customer would be told; ids, slugs and timestamps just invite it to
# paste raw tables into its answer and make every reply slower
_CUSTOMER_FIELDS = (
    "sku", "name", "price", "currency", "primary_color", "secondary_color", "border_type", "blouse_included",
    "fabric", "occasion", "bandha_type", "thread_count", "motif", "colour_match", "colour_note",
    "similarity", "score", "reason",
)


def _result_limit() -> int:
    # an open question ("what do you have?") must not hand the model the whole catalog to write about
    return int(os.getenv("AGENT_RESULT_LIMIT", "8"))


def _capped(rows: list[dict]) -> list[dict]:
    limit = _result_limit()
    if len(rows) <= limit:
        return rows
    note = (f"{len(rows) - limit} more sarees match and are not shown. Show at most 5 and ask the "
            "customer to narrow down by colour, price or design.")
    return rows[:limit] + [{"note": note}]


def _slim(row) -> dict:
    return {k: v for k, v in dict(row).items() if k in _CUSTOMER_FIELDS and v not in (None, "")}


def _unset(value):
    # a model may still send "" or 0 for an argument it should have omitted
    if isinstance(value, str):
        value = value.strip()
    return None if value in ("", 0, None) else value


@tool
def search_products(
    color: Color = "",
    bandha_type: BandhaType = "",
    thread_count: ThreadCount = 0,
    occasion: Occasion = "",
    fabric: Fabric = "",
    min_price: MinPrice = 0,
    max_price: MaxPrice = 0,
    colors: Colors = "",
    blouse: Blouse = "",
    design: Design = "",
    region: Region = "",
    border_color: BorderColor = "",
    pallu_color: PalluColor = "",
) -> str:
    """Search the in-stock saree catalog by price range, colour(s), border
    colour, pallu colour, blouse type, design words (pattern or motif),
    region, bandha type, thread count, occasion, or fabric. Returns matching products as a JSON list, including
    price. Pass only the filters the customer stated; call with no arguments
    to list everything in stock. When a result has colour_match "weaker",
    tell the customer honestly using its colour_note."""
    with engine.connect() as conn:
        try:
            results = search_products_service(
                conn,
                min_price=_unset(min_price),
                max_price=_unset(max_price),
                colors=_unset(colors) or _unset(color),
                bandha_type=_unset(bandha_type),
                thread_count=_unset(thread_count),
                occasion=_unset(occasion),
                fabric=_unset(fabric),
                blouse=_unset(blouse),
                design=_unset(design),
                region=_unset(region),
                border_color=_unset(border_color),
                pallu_color=_unset(pallu_color),
                in_stock_only=True,
                include_price=True,
                include_eval_fixtures=_include_eval_fixtures(),
                include_draft=_include_draft(),
            )
        except ValueError as exc:
            return _to_json({"error": str(exc)})
    return _to_json(_capped([_slim(row) for row in results]))


@tool
def search_by_image(
    image_name: ImageName,
    min_price: MinPrice = 0,
    max_price: MaxPrice = 0,
    blouse: Blouse = "",
) -> str:
    """Find in-stock sarees that look like a photo the customer uploaded,
    most similar first, with a similarity score and price. Use this when the
    customer sends a picture. Optionally keep only a price range or blouse
    type the customer also stated."""
    with engine.connect() as conn:
        try:
            results = search_by_image_service(
                conn,
                image_name,
                limit=5,
                min_price=_unset(min_price),
                max_price=_unset(max_price),
                blouse=_unset(blouse),
                include_eval_fixtures=_include_eval_fixtures(),
                include_draft=_include_draft(),
            )
        except ValueError as exc:
            return _to_json({"error": str(exc)})
        except EmbeddingServiceError as exc:
            return _to_json({"error": str(exc)})
    return _to_json([_slim(row) for row in results])


@tool
def recommend_products(
    max_price: MaxPrice = 0,
    color: Color = "",
    bandha_type: BandhaType = "",
    thread_count: ThreadCount = 0,
    occasion: Occasion = "",
    fabric: Fabric = "",
) -> str:
    """Rank in-stock sarees against customer preferences, including budget,
    and return the best matches with a score and a short reason as JSON.
    Use this when the customer describes what they want and you should
    suggest specific sarees rather than just list matches. Pass only the
    preferences the customer stated."""
    with engine.connect() as conn:
        try:
            results = recommend_products_service(
                conn,
                max_price=_unset(max_price),
                color=_unset(color),
                bandha_type=_unset(bandha_type),
                thread_count=_unset(thread_count),
                occasion=_unset(occasion),
                fabric=_unset(fabric),
                limit=5,
                include_price=True,
                include_eval_fixtures=_include_eval_fixtures(),
                include_draft=_include_draft(),
            )
        except ValueError as exc:
            return _to_json({"error": str(exc)})
    return _to_json([_slim(row) for row in results])


@tool
def get_product_details(
    sku: Annotated[str, Field(description="Exact product SKU, e.g. SAM-2026-0001.")],
) -> str:
    """Get full details, including price, for one saree by its exact SKU.
    Use this when the customer refers to a specific saree already identified
    by SKU, for example after a search or recommendation result."""
    with engine.connect() as conn:
        result = get_product_details_service(
            conn,
            sku,
            include_price=True,
            include_eval_fixtures=_include_eval_fixtures(),
            include_draft=_include_draft(),
        )
    if result is None:
        return _to_json({"error": f"no active product found for sku {sku}"})
    return _to_json(_slim(result))


TOOLS = [search_products, search_by_image, recommend_products, get_product_details]
