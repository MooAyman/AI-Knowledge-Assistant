import re
import unicodedata
from typing import Any

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
WORD_RE = re.compile(r"[A-Za-z0-9_\u0600-\u06FF]+")

DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
ALEF_VARIANTS = str.maketrans(
    {
        "\u0622": "\u0627",  # آ -> ا
        "\u0623": "\u0627",  # أ -> ا
        "\u0625": "\u0627",  # إ -> ا
        "\u0671": "\u0627",  # ٱ -> ا
    }
)


def is_arabic_text(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def normalize_arabic(text: str) -> str:
    normalized = DIACRITICS_RE.sub("", text)
    normalized = normalized.translate(ALEF_VARIANTS)
    normalized = normalized.replace("\u0629", "\u0647")  # ة -> ه
    normalized = normalized.replace("\u0649", "\u064a")  # ى -> ي
    return unicodedata.normalize("NFC", normalized)


def tokenize_for_analysis(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def prepare_arabic_query(query: str) -> tuple[str, dict[str, Any]]:
    if not is_arabic_text(query):
        return query, {
            "is_arabic": False,
            "original_query": query,
            "retrieval_query": query,
            "normalized": False,
            "original_tokens": tokenize_for_analysis(query),
            "retrieval_tokens": tokenize_for_analysis(query),
        }

    retrieval_query = normalize_arabic(query)
    return retrieval_query, {
        "is_arabic": True,
        "original_query": query,
        "retrieval_query": retrieval_query,
        "normalized": retrieval_query != query,
        "original_tokens": tokenize_for_analysis(query),
        "retrieval_tokens": tokenize_for_analysis(retrieval_query),
    }
