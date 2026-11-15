"""Turn the model's markdown reply into plain WhatsApp text."""
import re

MAX_REPLY_CHARS = 1500

# internal codes: SAM-DATA-0022, SAM-2026-0003, EVAL-0001
_CODE = r"(?:SAM-(?:DATA-|\d{4}-)\d{4}|EVAL-\d{4})"


def strip_internal_codes(text: str) -> str:
    """Customers know sarees by name; a code the model slipped in is removed, with its brackets and label."""
    text = re.sub(rf"\s*[(\[]\s*(?:SKU\s*:?\s*)?`?{_CODE}`?\s*[)\]]", "", text)
    text = re.sub(rf"\bSKU\s*:?\s*`?{_CODE}`?", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"`?{_CODE}`?", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]+([,.;:!?])", r"\1", text)


def to_whatsapp(text: str, limit: int = MAX_REPLY_CHARS) -> str:
    text = strip_internal_codes(text.replace("\r\n", "\n"))
    # raw tables are unreadable on a phone
    text = "\n".join(line for line in text.split("\n") if not line.lstrip().startswith("|"))
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)  # **bold** -> *bold* (WhatsApp style)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return (cut[:boundary + 1] if boundary > limit // 2 else cut).rstrip() + " …"
