from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from litbase_ai.utils.text import is_http_url

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None


_REJECT_MARKERS = (
    "/data-providers/",
    "/data-provider/",
    "/providers/",
    "/journals/",
    "/subjects/",
)


def is_plausible_pdf_url(url: str) -> bool:
    """Return True when a URL looks like a direct PDF or download endpoint."""
    if not is_http_url(url):
        return False

    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    host = (parsed.hostname or "").lower()
    combined = f"{path}?{query}"

    if any(marker in combined for marker in _REJECT_MARKERS):
        return False
    if path.endswith(".pdf"):
        return True
    if "/pdf" in path or "/download" in path or "/article/file" in path:
        return True
    if "format=pdf" in query or "type=pdf" in query:
        return True
    if "download=1" in query and "pdf" in combined:
        return True
    if ("hal.science" in host or "archives-ouvertes" in host) and path.endswith("/document"):
        return True
    return False


def extract_pdf_urls_from_html(html: str, base_url: str, limit: int = 12) -> list[str]:
    """Extract direct-PDF candidates from landing-page HTML."""
    candidates: list[str] = []

    def add(url: str | None) -> None:
        if not url:
            return
        resolved = urljoin(base_url, str(url).strip())
        if resolved:
            candidates.append(resolved)

    meta_patterns = (
        r'''<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']''',
        r'''<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']''',
    )
    for pattern in meta_patterns:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE):
            add(match.group(1))

    regex_patterns = (
        r'''href=["\']([^"\']*?\.pdf[^"\']*?)["\']''',
        r'''href=["\']([^"\']*?/pdf[^"\']*?)["\']''',
        r'''href=["\']([^"\']*?/article/file[^"\']*?)["\']''',
        r'''href=["\']([^"\']*?(?:type=printable|format=pdf)[^"\']*?)["\']''',
        r'''data-pdf[^=]*=["\']([^"\']+?)["\']''',
        r'''window\.open\(["\']([^"\']*?\.pdf[^"\']*?)["\']''',
        r'''(?:location|href)\s*=\s*["\']([^"\']*?\.pdf[^"\']*?)["\']''',
        r'''<(?:embed|iframe)[^>]*src=["\']([^"\']*?\.pdf[^"\']*?)["\']''',
    )
    for pattern in regex_patterns:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE):
            add(match.group(1))

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(["a", "iframe", "embed", "link", "meta"]):
                if tag.name == "meta":
                    meta_name = (tag.get("name") or "").lower()
                    if meta_name == "citation_pdf_url":
                        add(tag.get("content"))
                    continue
                add(tag.get("href") or tag.get("src"))
                add(tag.get("data-pdf-url"))
                add(tag.get("data-download-url"))
        except Exception:  # pragma: no cover
            pass
    else:
        for match in re.finditer(r'''(?:href|src)=["\']([^"\']+)["\']''', html, flags=re.IGNORECASE):
            add(match.group(1))

    deduped: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        cleaned = url.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        if is_plausible_pdf_url(cleaned):
            deduped.append(cleaned)
        if len(deduped) >= limit:
            break
    return deduped


def dedupe_candidate_entries(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Dedupe candidate dicts by URL while preserving order."""
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in candidates:
        url = str(item.get("url") or "").strip()
        key = url.lower()
        if not url or key in seen:
            continue
        seen.add(key)
        deduped.append({**item, "url": url})
    return deduped
