"""Customers, conversations and messages (existing ai.* tables)."""
import json
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection


def get_or_create_customer(conn: Connection, whatsapp_number: str) -> UUID:
    return conn.execute(
        text("""
            INSERT INTO ai.customers (whatsapp_number, last_seen_at) VALUES (:n, now())
            ON CONFLICT (whatsapp_number) DO UPDATE SET last_seen_at = now(), updated_at = now()
            RETURNING id"""),
        {"n": whatsapp_number},
    ).scalar_one()


def _new_conversation(conn: Connection, customer_id: UUID) -> UUID:
    return conn.execute(
        text("INSERT INTO ai.conversations (customer_id) VALUES (:c) RETURNING id"), {"c": customer_id}
    ).scalar_one()


def open_conversation(conn: Connection, customer_id: UUID, idle_hours: int = 24) -> UUID:
    """Continue the open conversation unless it has been idle longer than idle_hours."""
    found = conn.execute(
        text("""
            SELECT id FROM ai.conversations
            WHERE customer_id = :c AND status = 'OPEN' AND last_message_at > now() - make_interval(hours => :h)
            ORDER BY last_message_at DESC LIMIT 1"""),
        {"c": customer_id, "h": idle_hours},
    ).scalar_one_or_none()
    if found:
        return found
    conn.execute(text("UPDATE ai.conversations SET status = 'CLOSED' WHERE customer_id = :c AND status = 'OPEN'"), {"c": customer_id})
    return _new_conversation(conn, customer_id)


def start_new_conversation(conn: Connection, customer_id: UUID) -> UUID:
    conn.execute(text("UPDATE ai.conversations SET status = 'CLOSED' WHERE customer_id = :c AND status = 'OPEN'"), {"c": customer_id})
    return _new_conversation(conn, customer_id)


def save_message(
    conn: Connection,
    conversation_id: UUID,
    sender_type: str,
    message_type: str = "TEXT",
    content: str | None = None,
    media_url: str | None = None,
    metadata: dict | None = None,
    whatsapp_message_id: str | None = None,
) -> UUID | None:
    """Returns the new message id, or None when this WhatsApp message id was already stored."""
    message_id = conn.execute(
        text("""
            INSERT INTO ai.messages (conversation_id, sender_type, message_type, content, media_url, metadata,
                                     whatsapp_message_id, created_at)
            VALUES (:c, CAST(:s AS ai.message_sender), CAST(:t AS ai.message_type), :content, :media,
                    CAST(:meta AS jsonb), :wid, clock_timestamp())
            ON CONFLICT (whatsapp_message_id) WHERE whatsapp_message_id IS NOT NULL DO NOTHING
            RETURNING id"""),
        {"c": conversation_id, "s": sender_type, "t": message_type, "content": content, "media": media_url,
         "meta": json.dumps(metadata or {}), "wid": whatsapp_message_id},
    ).scalar_one_or_none()
    if message_id:
        conn.execute(text("UPDATE ai.conversations SET last_message_at = now() WHERE id = :c"), {"c": conversation_id})
    return message_id


def load_history(conn: Connection, conversation_id: UUID, limit: int = 10) -> list[dict]:
    """The latest text and photo turns, oldest first, for the agent's memory."""
    rows = conn.execute(
        text("""
            SELECT sender_type::text AS sender, message_type::text AS kind, content, metadata
            FROM ai.messages
            WHERE conversation_id = :c AND message_type IN ('TEXT', 'IMAGE')
              AND sender_type IN ('CUSTOMER', 'AGENT', 'HUMAN_AGENT')
            ORDER BY created_at DESC LIMIT :n"""),
        {"c": conversation_id, "n": limit},
    ).mappings().all()
    return [dict(r) for r in reversed(rows)]


def recent_customer_message_count(conn: Connection, customer_id: UUID, minutes: int = 60) -> int:
    return conn.execute(
        text("""
            SELECT count(*) FROM ai.messages m JOIN ai.conversations c ON c.id = m.conversation_id
            WHERE c.customer_id = :c AND m.sender_type = 'CUSTOMER' AND m.created_at > now() - make_interval(mins => :m)"""),
        {"c": customer_id, "m": minutes},
    ).scalar_one()


def primary_photo_url(conn: Connection, sku: str) -> str | None:
    return conn.execute(
        text("""
            SELECT pi.image_url FROM catalog.product_images pi JOIN catalog.products p ON p.id = pi.product_id
            WHERE p.sku = :sku
            ORDER BY pi.is_primary DESC, (pi.image_type = 'flat_lay') DESC, pi.created_at LIMIT 1"""),
        {"sku": sku},
    ).scalar_one_or_none()
