"""LibGen (Library Genesis) PDF download source with multi-mirror rotation.

Supports:
- Automatic mirror rotation across multiple LibGen mirrors
- DOI-based search and download
- Title-based search fallback
- PDF extraction from LibGen download pages
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from litbase_ai.utils.logging import get_logger

logger = get_logger(__name__)

# ── LibGen mirrors (ordered by reliability) ──────────────────────────────
LIBGEN_MIRRORS = [
    "https://libgen.li",
    "https://libgen.is",
    "https://libgen.st",
    "https://libgen.rs",
    "https://libgen.gs",
]

# Download link patterns
_LIBGEN_DOWNLOAD_PATTERNS = [
    # Direct PDF links
    re.compile(r'''href\s*=\s*["']([^"']*\.pdf[^"']*)["']''', re.IGNORECASE),
    # get.php download links
    re.compile(r'''href\s*=\s*["']([^"']*get\.php[^"']+)["']''', re.IGNORECASE),
    # Download button patterns
    re.compile(r'''<a[^>]*href\s*=\s*["']([^"']*(?:download|get|dl)[^"']*)["'][^>]*>''', re.IGNORECASE),
    # Mirror links
    re.compile(r'''href\s*=\s*["']([^"']*(?:mirror|cloudflare|ipfs)[^"']*)["']''', re.IGNORECASE),
]

# Patterns that indicate no results found
_LIBGEN_NO_RESULTS_PATTERNS = [
    re.compile(r"no\s+(?:results?|documents?|items?)\s+found", re.IGNORECASE),
    re.compile(r"0\s+results?", re.IGNORECASE),
    re.compile(r"not\s+found", re.IGNORECASE),
]


class LibGenClient:
    """Lightweight LibGen client with mirror rotation."""

    def __init__(
        self,
        mirrors: list[str] | None = None,
        proxy: str | None = None,
        connect_timeout: float = 20.0,
        read_timeout: float = 30.0,
        user_agent: str = "LitBase-AI/0.3 (+libgen-discovery)",
    ):
        self.mirrors = mirrors or list(LIBGEN_MIRRORS)
        self.proxy = proxy
        self.user_agent = user_agent
        self.timeout = httpx.Timeout(
            connect=max(1.0, float(connect_timeout)),
            read=max(1.0, float(read_timeout)),
            write=max(1.0, float(read_timeout)),
            pool=max(1.0, float(connect_timeout)),
        )
        # Mirror health tracking
        self._mirror_health: dict[str, dict[str, Any]] = {}
        self._failure_cooldowns: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def try_download(self, doi: str, output_path: Path, title: str = "") -> tuple[bool, dict[str, Any]]:
        """Attempt to download a paper via LibGen.

        Args:
            doi: Paper DOI for lookup.
            output_path: Where to save the PDF.
            title: Paper title as fallback search term.

        Returns:
            (success, info_dict) with trace data.
        """
        if not doi and not title:
            return False, {"source": "libgen", "status": "failed", "reason": "no_identifier"}

        trace_entries: list[dict[str, Any]] = []

        for mirror in self._get_active_mirrors():
            # Try DOI search first
            if doi:
                ok, info = self._try_mirror_by_doi(doi=doi, mirror=mirror, output_path=output_path)
                trace_entries.extend(info)
                if ok:
                    return True, {"source": "libgen", "status": "downloaded", "trace": trace_entries}
                self._mark_mirror_failure(mirror)

            # Fallback to title search
            if title:
                ok, info = self._try_mirror_by_title(title=title, mirror=mirror, output_path=output_path)
                trace_entries.extend(info)
                if ok:
                    return True, {"source": "libgen", "status": "downloaded", "trace": trace_entries}
                self._mark_mirror_failure(mirror)

        return False, {"source": "libgen", "status": "failed", "reason": "all_mirrors_exhausted", "trace": trace_entries}

    def resolve_pdf_url(self, doi: str, title: str = "") -> str | None:
        """Resolve a DOI/title to a LibGen PDF URL without downloading."""
        for mirror in self._get_active_mirrors():
            url = self._try_find_pdf_url(mirror=mirror, doi=doi, title=title)
            if url:
                return url
        return None

    # ── Mirror Management ─────────────────────────────────────────────────

    def _get_active_mirrors(self) -> list[str]:
        """Get mirrors sorted by health, with failed ones at the end."""
        now = time.time()
        active: list[tuple[str, float]] = []
        cooldown: list[tuple[str, float]] = []

        for mirror in self.mirrors:
            health = self._mirror_health.get(mirror, {})
            score = health.get("score", 0.5)

            last_fail = self._failure_cooldowns.get(mirror, 0.0)
            if now - last_fail < 300:  # 5 min cooldown
                cooldown.append((mirror, score))
            else:
                active.append((mirror, score))

        active.sort(key=lambda x: x[1], reverse=True)
        cooldown.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in active] + [d for d, _ in cooldown]

    def _mark_mirror_failure(self, mirror: str) -> None:
        """Mark a mirror as failed for cooldown."""
        self._failure_cooldowns[mirror] = time.time()
        if mirror not in self._mirror_health:
            self._mirror_health[mirror] = {}
        self._mirror_health[mirror]["score"] = max(0.1, self._mirror_health[mirror].get("score", 0.5) - 0.15)

    def _mark_mirror_success(self, mirror: str) -> None:
        """Mark a mirror as successful."""
        if mirror not in self._mirror_health:
            self._mirror_health[mirror] = {}
        self._mirror_health[mirror]["score"] = min(1.0, self._mirror_health[mirror].get("score", 0.5) + 0.1)
        self._failure_cooldowns.pop(mirror, None)

    # ── Download Logic ─────────────────────────────────────────────────────

    def _try_mirror_by_doi(
        self, doi: str, mirror: str, output_path: Path
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Try to find and download a paper by DOI on a LibGen mirror."""
        info: list[dict[str, Any]] = []
        encoded_doi = quote(doi, safe="")
        # LibGen search URL
        search_url = f"{mirror.rstrip('/')}/scimag/ads.php?doi={encoded_doi}"

        info.append({"status": "attempt", "url": search_url, "source": "libgen"})

        try:
            result = self._search_and_download(
                search_url=search_url,
                mirror=mirror,
                output_path=output_path,
            )
            if result:
                ok, extra_info = result
                info.extend(extra_info)
                if ok:
                    self._mark_mirror_success(mirror)
                    return True, info
            info.append({
                "status": "non_pdf",
                "url": search_url,
                "source": "libgen",
                "reason": "no_download_link_found",
            })
            return False, info
        except Exception as exc:
            info.append({
                "status": "request_failed",
                "url": search_url,
                "source": "libgen",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return False, info

    def _try_mirror_by_title(
        self, title: str, mirror: str, output_path: Path
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Try to find and download a paper by title on a LibGen mirror."""
        info: list[dict[str, Any]] = []
        encoded_title = quote(title, safe="")
        search_url = f"{mirror.rstrip('/')}/scimag/?q={encoded_title}"

        info.append({"status": "attempt", "url": search_url, "source": "libgen:title"})

        try:
            result = self._search_and_download(
                search_url=search_url,
                mirror=mirror,
                output_path=output_path,
            )
            if result:
                ok, extra_info = result
                info.extend(extra_info)
                if ok:
                    self._mark_mirror_success(mirror)
                    return True, info
            info.append({
                "status": "non_pdf",
                "url": search_url,
                "source": "libgen",
                "reason": "no_download_link_found",
            })
            return False, info
        except Exception as exc:
            info.append({
                "status": "request_failed",
                "url": search_url,
                "source": "libgen",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return False, info

    def _try_find_pdf_url(self, mirror: str, doi: str, title: str = "") -> str | None:
        """Find PDF URL on a LibGen mirror without downloading."""
        if doi:
            encoded = quote(doi, safe="")
            search_url = f"{mirror.rstrip('/')}/scimag/ads.php?doi={encoded}"
        elif title:
            encoded = quote(title, safe="")
            search_url = f"{mirror.rstrip('/')}/scimag/?q={encoded}"
        else:
            return None

        try:
            return self._extract_download_url(search_url, mirror)
        except Exception:
            return None

    def _search_and_download(
        self, search_url: str, mirror: str, output_path: Path
    ) -> tuple[bool, list[dict[str, Any]]] | None:
        """Search on a LibGen mirror and attempt download.

        Returns None if no results found, otherwise (success, info_list).
        """
        download_url = self._extract_download_url(search_url, mirror)
        if not download_url:
            return None

        return self._download_from_url(download_url, output_path)

    def _extract_download_url(self, search_url: str, mirror: str) -> str | None:
        """Fetch search page and extract the first download URL."""
        client_kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "follow_redirects": True,
            "headers": {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/pdf,*/*",
            },
            "trust_env": not bool(self.proxy),
        }
        if self.proxy:
            client_kwargs["proxy"] = self.proxy

        with httpx.Client(**client_kwargs) as client:
            resp = client.get(search_url)

            if resp.status_code >= 400:
                return None

            html = self._decode_html(resp.content)

            # Check for no results
            for pattern in _LIBGEN_NO_RESULTS_PATTERNS:
                if pattern.search(html):
                    return None

            # Try to extract PDF or download link
            for pattern in _LIBGEN_DOWNLOAD_PATTERNS:
                match = pattern.search(html)
                if match:
                    link = match.group(1)
                    if link.startswith("//"):
                        link = "https:" + link
                    elif not link.startswith("http"):
                        link = urljoin(search_url, link)

                    # Prefer .pdf links
                    if link.lower().endswith(".pdf"):
                        return link

                    # For get.php links, follow redirect to get the actual PDF
                    if "get.php" in link or "download" in link.lower():
                        actual_url = self._resolve_redirect(link, client)
                        if actual_url:
                            return actual_url
                        return link  # Return as-is if can't resolve

                    return link

        return None

    def _resolve_redirect(self, url: str, client: httpx.Client) -> str | None:
        """Follow redirects to find the final PDF URL."""
        try:
            resp = client.get(url, follow_redirects=True)
            if resp.status_code < 400:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "application/pdf" in content_type:
                    return str(resp.url)
                # Check if the final URL ends with .pdf
                if str(resp.url).lower().endswith(".pdf"):
                    return str(resp.url)
        except Exception:
            pass
        return None

    def _download_from_url(
        self, url: str, output_path: Path
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Download PDF from a LibGen download URL."""
        info: list[dict[str, Any]] = []

        client_kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "follow_redirects": True,
            "headers": {
                "User-Agent": self.user_agent,
                "Accept": "application/pdf,application/octet-stream,*/*",
            },
            "trust_env": not bool(self.proxy),
        }
        if self.proxy:
            client_kwargs["proxy"] = self.proxy

        try:
            with httpx.Client(**client_kwargs) as client:
                resp = client.get(url)

                if resp.status_code >= 400:
                    info.append({
                        "status": "http_error",
                        "url": str(resp.url),
                        "source": "libgen",
                        "reason": f"http_{resp.status_code}",
                    })
                    return False, info

                content = resp.content or b""

                if self._looks_like_pdf(resp.headers, content):
                    if self._write_pdf(content, output_path):
                        info.append({
                            "status": "downloaded",
                            "url": str(resp.url),
                            "source": "libgen",
                        })
                        return True, info

                info.append({
                    "status": "non_pdf",
                    "url": str(resp.url),
                    "source": "libgen",
                    "reason": "content_not_pdf",
                })
                return False, info

        except Exception as exc:
            info.append({
                "status": "request_failed",
                "url": url,
                "source": "libgen",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return False, info

    # ── Content Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _looks_like_pdf(headers: httpx.Headers, content: bytes) -> bool:
        """Check if response looks like a PDF."""
        content_type = (headers.get("Content-Type") or "").lower()
        head = content[:2048]
        if b"%PDF" in head:
            return True
        if "application/pdf" in content_type and len(content) > 1024:
            return True
        return False

    @staticmethod
    def _decode_html(content: bytes) -> str:
        """Decode bytes to HTML string."""
        if not content:
            return ""
        for encoding in ("utf-8", "latin-1", "iso-8859-1"):
            try:
                return content.decode(encoding, errors="ignore")
            except Exception:
                continue
        return ""

    @staticmethod
    def _write_pdf(content: bytes, path: Path) -> bool:
        """Write PDF content to file with validation."""
        if not content or len(content) < 1024:
            return False

        tmp_path = path.with_suffix(path.suffix + ".part")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(content)

            if not LibGenClient._is_valid_pdf(tmp_path):
                tmp_path.unlink(missing_ok=True)
                return False

            tmp_path.replace(path)
            logger.info("LibGen downloaded: %s", path.name)
            return True
        except Exception as exc:
            logger.warning("Failed writing LibGen PDF %s: %s", path, exc)
            tmp_path.unlink(missing_ok=True)
            return False

    @staticmethod
    def _is_valid_pdf(path: Path) -> bool:
        """Validate that a file is a proper PDF."""
        try:
            size = path.stat().st_size
            if size < 1000:
                return False
            with path.open("rb") as fh:
                header = fh.read(5)
                if header != b"%PDF-":
                    return False
                fh.seek(max(0, size - 2048))
                tail = fh.read()
            return b"%%EOF" in tail or size > 4096
        except OSError:
            return False
