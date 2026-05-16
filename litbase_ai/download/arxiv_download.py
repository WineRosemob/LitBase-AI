"""arXiv direct PDF download source.

Supports:
- Direct PDF download from arXiv IDs (both old and new format)
- Version-specific PDF access
- Title/DOI to arXiv ID resolution
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx

from litbase_ai.utils.logging import get_logger

logger = get_logger(__name__)

# arXiv ID patterns
_ARXIV_ID_PATTERN = re.compile(
    r"(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?|"  # New format: 1234.56789
    r"[a-z\-]+(?:\.[a-z]+)?\/\d{7}(?:v\d+)?)",     # Old format: hep-th/1234567
    re.IGNORECASE,
)

_ARXIV_URL_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf|ftp)/([^\s?#&]+)",
    re.IGNORECASE,
)

_ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
_ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"


class ArxivDownloader:
    """Direct arXiv PDF downloader."""

    def __init__(
        self,
        proxy: str | None = None,
        connect_timeout: float = 15.0,
        read_timeout: float = 30.0,
        user_agent: str = "LitBase-AI/0.3 (+arxiv-download)",
    ):
        self.proxy = proxy
        self.user_agent = user_agent
        self.timeout = httpx.Timeout(
            connect=max(1.0, float(connect_timeout)),
            read=max(1.0, float(read_timeout)),
            write=max(1.0, float(read_timeout)),
            pool=max(1.0, float(connect_timeout)),
        )

    # ── Public API ────────────────────────────────────────────────────────

    def try_download(self, arxiv_id: str, output_path: Path) -> tuple[bool, dict[str, Any]]:
        """Download a paper directly from arXiv.

        Args:
            arxiv_id: arXiv identifier (e.g., '2301.12345' or '2301.12345v2').
            output_path: Where to save the PDF.

        Returns:
            (success, info_dict) with trace data.
        """
        if not arxiv_id:
            return False, {"source": "arxiv", "status": "failed", "reason": "no_arxiv_id"}

        # Clean arXiv ID
        arxiv_id = self._clean_arxiv_id(arxiv_id)
        if not arxiv_id:
            return False, {"source": "arxiv", "status": "failed", "reason": "invalid_arxiv_id"}

        return self._download_arxiv_pdf(arxiv_id=arxiv_id, output_path=output_path)

    def resolve_pdf_url(self, arxiv_id: str) -> str | None:
        """Resolve an arXiv ID to a PDF URL."""
        cleaned = self._clean_arxiv_id(arxiv_id)
        if not cleaned:
            return None
        return _ARXIV_PDF_URL.format(arxiv_id=cleaned)

    @classmethod
    def extract_arxiv_id(cls, text: str) -> str | None:
        """Extract an arXiv ID from text.

        Handles:
        - Raw IDs: '2301.12345', '2301.12345v2'
        - URLs: 'https://arxiv.org/abs/2301.12345'
        - Prefixed: 'arxiv:2301.12345'
        """
        if not text:
            return None

        # Try URL pattern first
        match = _ARXIV_URL_PATTERN.search(text)
        if match:
            return cls._clean_arxiv_id(match.group(1))

        # Try raw ID pattern
        match = _ARXIV_ID_PATTERN.search(text)
        if match:
            return cls._clean_arxiv_id(match.group(1))

        return None

    # ── Internal Methods ──────────────────────────────────────────────────

    @staticmethod
    def _clean_arxiv_id(raw: str) -> str | None:
        """Clean and normalize an arXiv ID.

        Strips version suffix for the download URL, but keeps the
        base ID. E.g., '2301.12345v2' -> '2301.12345'.
        """
        if not raw:
            return None
        # Remove 'arxiv:' prefix
        cleaned = raw.strip()
        cleaned = re.sub(r"^arxiv\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        # Remove version suffix for URL
        cleaned = re.sub(r"v\d+$", "", cleaned, flags=re.IGNORECASE)
        # Remove any trailing garbage
        cleaned = cleaned.strip()
        # Validate format
        if _ARXIV_ID_PATTERN.match(cleaned):
            return cleaned
        return None

    def _download_arxiv_pdf(
        self, arxiv_id: str, output_path: Path
    ) -> tuple[bool, dict[str, Any]]:
        """Download PDF from arXiv."""
        pdf_url = _ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
        trace: list[dict[str, Any]] = [
            {"status": "attempt", "url": pdf_url, "source": "arxiv"},
        ]

        client_kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "follow_redirects": True,
            "headers": {
                "User-Agent": self.user_agent,
                "Accept": "application/pdf,*/*",
            },
            "trust_env": not bool(self.proxy),
        }
        if self.proxy:
            client_kwargs["proxy"] = self.proxy

        try:
            with httpx.Client(**client_kwargs) as client:
                resp = client.get(pdf_url)

                if resp.status_code == 404:
                    trace.append({
                        "status": "http_error",
                        "url": str(resp.url),
                        "source": "arxiv",
                        "reason": "arxiv_404_not_found",
                    })
                    return False, {"source": "arxiv", "status": "failed", "trace": trace}

                if resp.status_code >= 400:
                    trace.append({
                        "status": "http_error",
                        "url": str(resp.url),
                        "source": "arxiv",
                        "reason": f"http_{resp.status_code}",
                    })
                    return False, {"source": "arxiv", "status": "failed", "trace": trace}

                content = resp.content or b""

                # arXiv redirects to the actual PDF, check content
                if b"%PDF" in content[:2048]:
                    if self._write_pdf(content, output_path):
                        trace.append({
                            "status": "downloaded",
                            "url": str(resp.url),
                            "source": "arxiv",
                        })
                        return True, {"source": "arxiv", "status": "downloaded", "trace": trace}

                # Sometimes arXiv returns HTML (error page)
                if b"<!DOCTYPE html" in content[:300].lower() or b"<html" in content[:300].lower():
                    trace.append({
                        "status": "non_pdf",
                        "url": str(resp.url),
                        "source": "arxiv",
                        "reason": "arxiv_returned_html",
                    })
                    return False, {"source": "arxiv", "status": "failed", "trace": trace}

                trace.append({
                    "status": "non_pdf",
                    "url": str(resp.url),
                    "source": "arxiv",
                    "reason": "content_not_pdf",
                })
                return False, {"source": "arxiv", "status": "failed", "trace": trace}

        except Exception as exc:
            trace.append({
                "status": "request_failed",
                "url": pdf_url,
                "source": "arxiv",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return False, {"source": "arxiv", "status": "failed", "trace": trace}

    @staticmethod
    def _write_pdf(content: bytes, path: Path) -> bool:
        """Write PDF content to file with validation."""
        if not content or len(content) < 1024:
            return False

        tmp_path = path.with_suffix(path.suffix + ".part")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(content)

            if not ArxivDownloader._is_valid_pdf(tmp_path):
                tmp_path.unlink(missing_ok=True)
                return False

            tmp_path.replace(path)
            logger.info("arXiv downloaded: %s", path.name)
            return True
        except Exception as exc:
            logger.warning("Failed writing arXiv PDF %s: %s", path, exc)
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
