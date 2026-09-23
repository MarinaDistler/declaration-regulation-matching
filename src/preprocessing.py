import re
import unicodedata
from typing import Any


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def declaration_to_text(row) -> str:
    return normalize_text(row.get("G31_1", ""))


def regulation_to_text(row) -> str:
    return normalize_text(row.get("npa", ""))