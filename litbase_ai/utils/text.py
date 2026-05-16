from __future__ import annotations

import html
import re
from urllib.parse import urlparse


DOI_PREFIX_PATTERN = re.compile(r"^(https?://(dx\.)?doi\.org/)", re.IGNORECASE)
MULTI_SPACE_PATTERN = re.compile(r"\s+")
PUNCT_PATTERN = re.compile(r"[^\w\s]")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_FILENAME_PATTERN = re.compile(r'[\\/*?:"<>|]+')


def normalize_doi(doi: str | None) -> str | None:
    """Normalize DOI into lowercase plain identifier."""
    if not doi:
        return None
    cleaned = DOI_PREFIX_PATTERN.sub("", doi.strip()).strip().lower()
    return cleaned or None


def normalize_title(title: str | None) -> str:
    """Normalize title for fuzzy matching."""
    if not title:
        return ""
    text = PUNCT_PATTERN.sub(" ", title.lower())
    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


def clean_filename(name: str, max_len: int = 120) -> str:
    """Sanitize string for filesystem usage."""
    text = NON_FILENAME_PATTERN.sub("_", name)
    text = MULTI_SPACE_PATTERN.sub("_", text.strip())
    text = text.strip("._")
    if not text:
        text = "paper"
    return text[:max_len]


def short_title(title: str | None, max_words: int = 6) -> str:
    """Get condensed lowercase title chunk."""
    if not title:
        return "untitled"
    words = normalize_title(title).split()
    if not words:
        return "untitled"
    return "_".join(words[:max_words])


def extract_first_author(authors: list[str]) -> str:
    """Get first author token from author list."""
    if not authors:
        return "unknown"
    first = authors[0].strip()
    if not first:
        return "unknown"
    surname = first.split(",")[0].split()[-1]
    return normalize_title(surname).replace(" ", "_") or "unknown"


def tokenize_topic(topic: str) -> list[str]:
    """Tokenize topic into meaningful lowercase tokens."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+_\.]*", topic.lower())
    return [token for token in tokens if len(token) > 2]


def safe_join(values: list[str | None], sep: str = ", ") -> str:
    """Join non-empty string values safely."""
    return sep.join(value for value in values if value)


def strip_html(text: str | None) -> str:
    """Remove basic HTML tags and entities."""
    if not text:
        return ""
    clean_text = HTML_TAG_PATTERN.sub(" ", text)
    clean_text = html.unescape(clean_text)
    return MULTI_SPACE_PATTERN.sub(" ", clean_text).strip()


def is_http_url(url: str | None) -> bool:
    """Return True if URL has HTTP(S) scheme."""
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"}

