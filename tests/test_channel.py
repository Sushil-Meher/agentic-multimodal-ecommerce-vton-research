import os
import time
import uuid

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import text

from src.agent.graph import build_graph
from src.api.db import engine
from src.channel import handler, media, store
from src.channel.formatting import to_whatsapp
from src.channel.handler import FALLBACK_REPLY, RATE_LIMIT_REPLY, BAD_PHOTO_REPLY, IncomingMessage, handle_incoming

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 64


class FakeLLM:
    """Canned replies in order; remembers every message list it was given."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.seen = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        return self._responses.pop(0)


class CrashingLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise RuntimeError("model backend unreachable")


def tool_call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "c1", "type": "tool_call"}])


@pytest.fixture
def phone():
    number = f"+9190{uuid.uuid4().int % 10**8:08d}"
    yield number
    with engine.begin() as conn:  # conversations and messages go with the customer
        conn.execute(text("DELETE FROM ai.customers WHERE whatsapp_number = :n"), {"n": number})


def say(phone, graph, content="hello", **kwargs):
    return handle_incoming(engine, graph, IncomingMessage(sender=phone, text=content, **kwargs.pop("msg", {})), **kwargs)


def rows(phone, columns="sender_type::text, message_type::text, content"):
    with engine.connect() as conn:
        return conn.execute(text(f"""
            SELECT {columns} FROM ai.messages m JOIN ai.conversations c ON c.id = m.conversation_id
            JOIN ai.customers u ON u.id = c.customer_id WHERE u.whatsapp_number = :n ORDER BY m.created_at"""), {"n": phone}).all()


# ---- formatting

def test_markdown_becomes_whatsapp_text():
    raw = "# Sarees\n\nHere is **BunkarHouse00001**:\n| id | name |\n|----|------|\n| 1 | x |\n\n\n\nBye"
    assert to_whatsapp(raw) == "Sarees\n\nHere is *BunkarHouse00001*:\n\nBye"


def test_long_replies_are_cut_cleanly():
    long = "This is a sentence. " * 200
    out = to_whatsapp(long, limit=300)
    assert len(out) <= 310 and out.endswith("…") and "sentence." in out


# ---- photos

def test_customer_photo_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "UPLOAD_DIR", str(tmp_path))
    a = media.save_customer_image(JPEG)
    b = media.save_customer_image(b"\x89PNG\r\n\x1a\n" + b"0" * 10)
    c = media.save_customer_image(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
    assert a.endswith(".jpg") and b.endswith(".png") and c.endswith(".webp") and len({a, b, c}) == 3
    for bad in (b"", b"MZ\x90\x00 an executable", b"<html>", JPEG + b"0" * (media.MAX_IMAGE_BYTES + 1)):
        with pytest.raises(ValueError):
            media.save_customer_image(bad)


def test_old_customer_photos_are_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "UPLOAD_DIR", str(tmp_path))
    old, new = media.save_customer_image(JPEG), media.save_customer_image(JPEG)
    stale = time.time() - 40 * 86400
    os.utime(tmp_path / old, (stale, stale))
    assert media.delete_old_uploads(days=30) == 1
    assert not (tmp_path / old).exists() and (tmp_path / new).exists()


# ---- store

def test_customer_and_conversation_lifecycle(phone):
    with engine.begin() as conn:
        c1, c2 = store.get_or_create_customer(conn, phone), store.get_or_create_customer(conn, phone)
        assert c1 == c2
        first = store.open_conversation(conn, c1)
        assert store.open_conversation(conn, c1) == first  # continued within the idle window
        conn.execute(text("UPDATE ai.conversations SET last_message_at = now() - interval '30 hours' WHERE id = :i"), {"i": first})
        second = store.open_conversation(conn, c1)
        assert second != first
        assert conn.execute(text("SELECT status FROM ai.conversations WHERE id = :i"), {"i": first}).scalar_one() == "CLOSED"
        assert store.start_new_conversation(conn, c1) != second


def test_duplicate_whatsapp_message_ids_are_ignored(phone):
    with engine.begin() as conn:
        conv = store.open_conversation(conn, store.get_or_create_customer(conn, phone))
        assert store.save_message(conn, conv, "CUSTOMER", content="a", whatsapp_message_id="wamid.T1")
        assert store.save_message(conn, conv, "CUSTOMER", content="a", whatsapp_message_id="wamid.T1") is None
        assert store.save_message(conn, conv, "CUSTOMER", content="b")  # no id: always stored


def test_history_is_ordered_and_limited(phone):
    with engine.begin() as conn:
        conv = store.open_conversation(conn, store.get_or_create_customer(conn, phone))
        for i in range(6):
            store.save_message(conn, conv, "CUSTOMER" if i % 2 == 0 else "AGENT", content=f"m{i}")
        store.save_message(conn, conv, "AGENT", "PRODUCT", content="card")  # product cards are not conversation
        history = store.load_history(conn, conv, limit=4)
    assert [h["content"] for h in history] == ["m2", "m3", "m4", "m5"]


# ---- handler

def test_text_turn_is_saved_and_formatted(phone):
    out = say(phone, build_graph(llm=FakeLLM([AIMessage(content="**Hello** there")])), "hi")
    assert [(o.kind, o.text) for o in out] == [("text", "*Hello* there")]
    assert rows(phone) == [("CUSTOMER", "TEXT", "hi"), ("AGENT", "TEXT", "*Hello* there")]


def test_the_model_never_sees_the_phone_number_and_remembers_the_chat(phone):
    llm = FakeLLM([AIMessage(content="first answer"), AIMessage(content="second answer")])
    graph = build_graph(llm=llm)
    say(phone, graph, "show me green sarees")
    say(phone, graph, "which is cheapest?")
    everything = " ".join(str(getattr(m, "content", m)) for call in llm.seen for m in call)
    assert phone not in everything and phone[-6:] not in everything
    second_call = [m.content for m in llm.seen[1]]
    assert "show me green sarees" in second_call and "first answer" in second_call and second_call[-1] == "which is cheapest?"


def test_retried_delivery_is_ignored(phone):
    llm = FakeLLM([AIMessage(content="once")])
    graph = build_graph(llm=llm)
    msg = IncomingMessage(sender=phone, text="hi", message_id="wamid.RETRY")
    assert [o.text for o in handle_incoming(engine, graph, msg)] == ["once"]
    assert handle_incoming(engine, graph, msg) == [] and len(llm.seen) == 1


def test_rate_limit_stops_runaway_senders(phone):
    llm = FakeLLM([AIMessage(content="a"), AIMessage(content="b")])
    graph = build_graph(llm=llm)
    say(phone, graph, "1", hourly_limit=2)
    say(phone, graph, "2", hourly_limit=2)
    out = say(phone, graph, "3", hourly_limit=2)
    assert [o.text for o in out] == [RATE_LIMIT_REPLY] and len(llm.seen) == 2


def test_agent_failure_gives_a_polite_fallback_not_an_error(phone):
    out = say(phone, build_graph(llm=CrashingLLM()), "hi")
    assert [o.text for o in out] == [FALLBACK_REPLY]
    assert ("SYSTEM", "TEXT", FALLBACK_REPLY) in rows(phone)


def test_new_conversation_flag_starts_fresh_memory(phone):
    llm = FakeLLM([AIMessage(content="a"), AIMessage(content="b")])
    graph = build_graph(llm=llm)
    say(phone, graph, "old topic")
    say(phone, graph, "new topic", new_conversation=True)
    assert [m.content for m in llm.seen[1][1:]] == ["new topic"]  # [0] is the system prompt


def test_customer_photo_is_saved_and_the_agent_gets_only_its_name(phone, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "UPLOAD_DIR", str(tmp_path))
    llm = FakeLLM([AIMessage(content="Here are similar sarees")])
    out = handle_incoming(engine, build_graph(llm=llm), IncomingMessage(sender=phone, kind="image", image_bytes=JPEG, text="under 9000"))
    told = llm.seen[0][-1].content
    name = told.split("saved as ")[1].split("]")[0]
    assert (tmp_path / name).is_file() and str(tmp_path) not in told and "under 9000" in told
    assert out[0].text == "Here are similar sarees"
    assert rows(phone, "message_type::text, metadata->>'upload'")[0] == ("IMAGE", name)


def test_unreadable_photo_is_refused_without_calling_the_agent(phone, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "UPLOAD_DIR", str(tmp_path))
    llm = FakeLLM([])
    out = handle_incoming(engine, build_graph(llm=llm), IncomingMessage(sender=phone, kind="image", image_bytes=b"not an image"))
    assert [o.text for o in out] == [BAD_PHOTO_REPLY] and llm.seen == [] and list(tmp_path.iterdir()) == []


def make_saree(product_factory, name, tmp_path, price=7000):
    sku = product_factory(status="ACTIVE", price=price, inventory_status="AVAILABLE", name=name)
    url = f"local:test/{sku}.jpg"
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO catalog.product_images (product_id, image_url, image_type, storage_provider, is_primary, is_private)
                             SELECT id, :u, 'flat_lay', 'local', TRUE, TRUE FROM catalog.products WHERE sku = :s"""), {"u": url, "s": sku})
    (tmp_path / media.wa_photo_name(url)).write_bytes(JPEG)
    return sku


def test_photos_of_the_sarees_named_in_the_reply_are_attached(phone, product_factory, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "WA_IMAGE_DIR", str(tmp_path))
    names = [f"Qzx{uuid.uuid4().hex[:6]}" for _ in range(4)]
    for n in names:
        make_saree(product_factory, n, tmp_path)
    reply = "I suggest " + ", ".join(f"**{n}**" for n in names[::-1])  # named in reverse order
    llm = FakeLLM([tool_call("search_products", {"design": ""}), AIMessage(content=reply)])
    out = say(phone, build_graph(llm=llm), "show sarees", max_photos=3)
    images = [o for o in out if o.kind == "image"]
    assert [o.text.split(" - ")[0] for o in images] == names[::-1][:3]  # in the order named, capped at 3
    assert all(o.image_path and o.text.endswith("₹7,000") for o in images)
    assert [r[1] for r in rows(phone)].count("PRODUCT") == 3


def test_saree_not_named_in_the_reply_gets_no_photo(phone, product_factory, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "WA_IMAGE_DIR", str(tmp_path))
    make_saree(product_factory, f"Qzx{uuid.uuid4().hex[:6]}", tmp_path)
    llm = FakeLLM([tool_call("search_products", {}), AIMessage(content="Sorry, nothing matches.")])
    out = say(phone, build_graph(llm=llm), "anything?")
    assert [o.kind for o in out] == ["text"]


def test_default_photo_limit_is_three():
    assert handler.MAX_PHOTOS == 3


# ---- codes, retry, result cap, photo viewer

def test_internal_codes_are_removed_from_replies():
    from src.channel.formatting import strip_internal_codes

    assert strip_internal_codes("*BunkarHouse00022* (SAM-DATA-0022) at ₹6,200") == "*BunkarHouse00022* at ₹6,200"
    assert strip_internal_codes("Green Saree (SKU: SAM-2026-0003) is ₹2,900") == "Green Saree is ₹2,900"
    assert strip_internal_codes("SKU: `SAM-DATA-0001`, price 7500") == ", price 7500"
    assert "SAM-" not in strip_internal_codes("see `SAM-DATA-0001` and EVAL-0002 and SKU SAM-DATA-0003.")
    assert strip_internal_codes("BunkarHouse00001 and the Samba pattern") == "BunkarHouse00001 and the Samba pattern"


def test_codes_never_reach_the_customer(phone):
    out = say(phone, build_graph(llm=FakeLLM([AIMessage(content="Try *BunkarHouse00022* (SAM-DATA-0022), ₹6,200.")])), "hi")
    assert "SAM-" not in out[0].text and "BunkarHouse00022" in out[0].text
    assert all("SAM-" not in (r[2] or "") for r in rows(phone))


class FailsOnceLLM(FakeLLM):
    def invoke(self, messages):
        if not self.seen:
            self.seen.append(messages)
            raise RuntimeError("model process restarted")
        return super().invoke(messages)


def test_one_failure_is_retried_before_giving_up(phone):
    out = say(phone, build_graph(llm=FailsOnceLLM([AIMessage(content="answered on the second try")])), "hi")
    assert [o.text for o in out] == ["answered on the second try"]


def test_two_failures_still_fall_back_politely(phone):
    assert [o.text for o in say(phone, build_graph(llm=CrashingLLM()), "hi")] == [FALLBACK_REPLY]


def test_search_tool_caps_what_the_model_sees(product_factory, monkeypatch):
    import json

    from src.agent.tools import search_products

    for _ in range(4):
        product_factory(status="ACTIVE", price=1000, inventory_status="AVAILABLE")
    monkeypatch.setenv("AGENT_RESULT_LIMIT", "2")
    payload = json.loads(search_products.invoke({}))
    assert len([p for p in payload if "sku" in p]) == 2 and "more sarees match" in payload[-1]["note"]
    monkeypatch.setenv("AGENT_RESULT_LIMIT", "1000")
    assert not any("note" in p for p in json.loads(search_products.invoke({})))


def test_prompt_limits_answers_and_forbids_catalog_summaries():
    from src.agent.graph import SYSTEM_PROMPT

    assert "at most 5 sarees" in SYSTEM_PROMPT and "never mention SKUs" in SYSTEM_PROMPT
    assert "ask what colour, budget or occasion" in SYSTEM_PROMPT


def test_console_adapter_opens_photos_only_when_asked(monkeypatch, capsys):
    from src.channel.adapters import ConsoleAdapter
    from src.channel.handler import OutgoingMessage

    opened = []
    monkeypatch.setattr(os, "startfile", opened.append, raising=False)
    photo = OutgoingMessage("image", "BunkarHouse00001 - ₹7,500", "C:/x/a.jpg")
    ConsoleAdapter().send("n", photo)
    assert opened == []
    ConsoleAdapter(open_photos=True).send("n", photo)
    assert opened == ["C:/x/a.jpg"] and "BunkarHouse00001" in capsys.readouterr().out
