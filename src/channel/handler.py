"""One incoming customer message in, replies out.

The agent only ever sees text (and the name of a saved photo): never the phone number.
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from . import media, store
from .formatting import to_whatsapp

log = logging.getLogger(__name__)

MAX_PHOTOS = int(os.environ.get("CHANNEL_MAX_PHOTOS", "3"))
MAX_CUSTOMER_TEXT = 1000
FALLBACK_REPLY = "Sorry, I couldn't answer that just now. Our team will get back to you shortly."
RATE_LIMIT_REPLY = "You're sending messages very quickly. Please try again in a little while."
BAD_PHOTO_REPLY = "Sorry, I can only read JPEG, PNG or WebP photos up to 10 MB. Could you send it again?"
DEFAULT_PHOTO_REQUEST = "Find sarees that look like it."


@dataclass
class IncomingMessage:
    sender: str  # the customer's WhatsApp number; used for the database only
    kind: Literal["text", "image"] = "text"
    text: str = ""  # message text, or the caption of a photo
    image_bytes: bytes | None = None
    message_id: str | None = None  # WhatsApp message id, used to ignore retried deliveries


@dataclass
class OutgoingMessage:
    kind: Literal["text", "image"]
    text: str
    image_path: str | None = None


def describe_turn(kind: str, content: str | None, upload: str | None) -> str:
    """How a turn is shown to the agent (and remembered in the history)."""
    content = (content or "").strip()
    if kind == "IMAGE":
        return f"[The customer sent a photo, saved as {upload}] {content or DEFAULT_PHOTO_REQUEST}"
    return content


def _history_messages(rows: list[dict]) -> list:
    messages = []
    for r in rows:
        if r["sender"] == "CUSTOMER":
            messages.append(HumanMessage(content=describe_turn(r["kind"], r["content"], (r["metadata"] or {}).get("upload"))))
        else:
            messages.append(AIMessage(content=r["content"] or ""))
    return messages


def _final_reply(new_messages: list) -> str | None:
    for m in reversed(new_messages):
        if isinstance(m, AIMessage) and not m.tool_calls and isinstance(m.content, str) and m.content.strip():
            return m.content
    return None


def _mentioned_products(new_messages: list, reply: str, limit: int) -> list[dict]:
    """Sarees the agent actually named in its answer, in the order it named them."""
    found = {}
    for m in new_messages:
        if not isinstance(m, ToolMessage):
            continue
        try:
            data = json.loads(m.content)
        except (TypeError, ValueError):
            continue
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict) and item.get("sku") and item.get("name") and item["name"] in reply:
                found.setdefault(item["sku"], item)
    return sorted(found.values(), key=lambda p: reply.find(p["name"]))[:limit]


def handle_incoming(engine, graph, msg: IncomingMessage, *, max_photos: int | None = None,
                    history_limit: int = 10, idle_hours: int = 24, hourly_limit: int = 30,
                    new_conversation: bool = False) -> list[OutgoingMessage]:
    max_photos = MAX_PHOTOS if max_photos is None else max_photos
    text_in = (msg.text or "")[:MAX_CUSTOMER_TEXT]

    upload = None
    refusal = None
    if msg.kind == "image":
        try:
            upload = media.save_customer_image(msg.image_bytes or b"")
        except ValueError:
            refusal = BAD_PHOTO_REPLY

    with engine.begin() as conn:
        customer = store.get_or_create_customer(conn, msg.sender)
        conversation = (store.start_new_conversation(conn, customer) if new_conversation
                        else store.open_conversation(conn, customer, idle_hours))
        history_rows = store.load_history(conn, conversation, history_limit)
        too_fast = store.recent_customer_message_count(conn, customer) >= hourly_limit
        saved = store.save_message(
            conn, conversation, "CUSTOMER", "IMAGE" if msg.kind == "image" else "TEXT", text_in,
            metadata={"upload": upload} if upload else {}, whatsapp_message_id=msg.message_id)
        if saved is None:
            return []  # the same WhatsApp message arrived twice
        early = refusal or (RATE_LIMIT_REPLY if too_fast else None)
        if early:
            store.save_message(conn, conversation, "SYSTEM", "TEXT", early)
            return [OutgoingMessage("text", early)]

    turn = HumanMessage(content=describe_turn("IMAGE" if upload else "TEXT", text_in, upload))
    sent = _history_messages(history_rows) + [turn]
    new_messages, reply = [], None
    for attempt in (1, 2):  # a local model process can fail once and restart by itself
        try:
            result = graph.invoke({"messages": sent})
            new_messages = result["messages"][len(sent):]
            reply = _final_reply(new_messages)
            if reply:
                break
        except Exception:  # a failed answer must never reach the customer as an error
            log.exception("agent failed (attempt %s)", attempt)
            new_messages, reply = [], None

    if not reply:
        with engine.begin() as conn:
            store.save_message(conn, conversation, "SYSTEM", "TEXT", FALLBACK_REPLY)
        return [OutgoingMessage("text", FALLBACK_REPLY)]

    reply = to_whatsapp(reply)
    outgoing = [OutgoingMessage("text", reply)]
    with engine.begin() as conn:
        store.save_message(conn, conversation, "AGENT", "TEXT", reply)
        for product in _mentioned_products(new_messages, reply, max_photos):
            url = store.primary_photo_url(conn, product["sku"])
            path = media.wa_photo_path(url) if url else None
            if not path:
                continue
            caption = product["name"] + (f" - ₹{float(product['price']):,.0f}" if product.get("price") else "")
            outgoing.append(OutgoingMessage("image", caption, path))
            store.save_message(conn, conversation, "AGENT", "PRODUCT", caption, media_url=path, metadata={"sku": product["sku"]})
    return outgoing
