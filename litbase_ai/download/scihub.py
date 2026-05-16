"""Sci-Hub PDF download source with multi-mirror rotation.

Supports:
- Automatic domain rotation across multiple Sci-Hub mirrors
- DOI-based download URL construction
- PDF extraction from Sci-Hub landing pages
- CAPTCHA / Cloudflare detection with fallback
- Domain health probing and cooldown
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

# ── Default Sci-Hub mirrors (ordered by reliability) ──────────────────────
DEFAULT_SCIHUB_DOMAINS = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
    "https://sci-hub.ee",
    "https://sci-hub.is",
]

# Extra mirrors that may be region-specific
EXTRA_SCIHUB_DOMAINS = [
    "https://sci-hub.wf",
    "https://sci-hub.se",
]

# Probing TTL (seconds) — don't re-probe too often
_PROBE_TTL_SECONDS = 4 * 3600  # 4 hours

# Cooldown for failed domains (seconds)
_FAILURE_COOLDOWN_SECONDS = 300  # 5 minutes

# PDF content patterns from Sci-Hub pages
_SCIHUB_PDF_PATTERNS = [
    re.compile(r'<iframe\s+[^>]*src\s*=\s*["\']([^"\']+?\.pdf[^"\']*)["\']', re.IGNORECASE),
    re.compile(r'<embed\s+[^>]*src\s*=\s*["\']([^"\']+?\.pdf[^"\']*)["\']', re.IGNORECASE),
    re.compile(r'''href\s*=\s*["']([^"']+?\.pdf[^"']*)["']''', re.IGNORECASE),
    re.compile(r'''location\.href\s*=\s*["']([^"']+?\.pdf[^"']*)["']''', re.IGNORECASE),
    re.compile(r'''window\.location\s*=\s*["']([^"']+?\.pdf[^"']*)["']''', re.IGNORECASE),
    re.compile(r'''<button[^>]*onclick\s*=\s*["']location\s*=\s*['"]([^"']+)['"][^>]*["']''', re.IGNORECASE),
]


class SciHubClient:
    """Lightweight Sci-Hub client with domain rotation and health tracking."""

    def __init__(
        self,
        domains: list[str] | None = None,
        proxy: str | None = None,
        connect_timeout: float = 20.0,
        read_timeout: float = 30.0,
        user_agent: str = "LitBase-AI/0.3 (+sci-hub-discovery)",
    ):
        self.domains = domains or list(DEFAULT_SCIHUB_DOMAINS)
        self.proxy = proxy
        self.user_agent = user_agent
        self.timeout = httpx.Timeout(
            connect=max(1.0, float(connect_timeout)),
            read=max(1.0, float(read_timeout)),
            write=max(1.0, float(read_timeout)),
            pool=max(1.0, float(connect_timeout)),
        )
        # In-memory domain health tracking
        self._domain_health: dict[str, dict[str, Any]] = {}
        self._last_probe_time: float = 0.0
        self._failure_cooldowns: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def try_download(self, doi: str, output_path: Path) -> tuple[bool, dict[str, Any]]:
        """Attempt to download a paper via Sci-Hub.

        Returns:
            (success, info_dict) where info_dict contains trace data.
        """
        if not doi:
            return False, {"source": "scihub", "status": "failed", "reason": "no_doi"}

        # Probe domains if needed
        self._maybe_probe_domains()

        # Try each domain in order
        domains_to_try = self._get_active_domains()
        trace_entries: list[dict[str, Any]] = []

        for domain in domains_to_try:
            url = self._build_scihub_url(doi, domain)
            trace_entries.append({"status": "attempt", "url": url, "source": "scihub"})

            ok, info = self._try_single_domain(doi=doi, domain=domain, output_path=output_path)
            trace_entries.extend(info)

            if ok:
                self._record_domain_result(domain, success=True, latency_ms=0)
                return True, {"source": "scihub", "status": "downloaded", "trace": trace_entries}

            self._record_domain_result(domain, success=False, latency_ms=0)
            # Mark domain for cooldown on failure
            self._failure_cooldowns[domain] = time.time()

        return False, {"source": "scihub", "status": "failed", "reason": "all_mirrors_exhausted", "trace": trace_entries}

    def resolve_pdf_url(self, doi: str) -> str | None:
        """Quickly resolve a DOI to a Sci-Hub PDF URL without downloading.

        Returns the final PDF URL if found, or None.
        """
        if not doi:
            return None

        self._maybe_probe_domains()
        domains_to_try = self._get_active_domains()

        for domain in domains_to_try:
            url = self._build_scihub_url(doi, domain)
            try:
                with httpx.Client(
                    timeout=self.timeout,
                    follow_redirects=True,
                    headers={"User-Agent": self.user_agent},
                    proxy=self.proxy,
                ) as client:
                    resp = client.get(url)
                    if resp.status_code >= 400:
                        continue

                    content = resp.content or b""
                    # Check if direct PDF
                    if self._looks_like_pdf(resp.headers, content):
                        self._record_domain_result(domain, success=True, latency_ms=0)
                        return str(resp.url)

                    # Try to extract PDF URL from HTML
                    pdf_url = self._extract_pdf_url_from_html(content, str(resp.url))
                    if pdf_url:
                        self._record_domain_result(domain, success=True, latency_ms=0)
                        return pdf_url
            except Exception:
                self._record_domain_result(domain, success=False, latency_ms=0)
                self._failure_cooldowns[domain] = time.time()
                continue

        return None

    # ── Domain Management ─────────────────────────────────────────────────

    def _maybe_probe_domains(self) -> None:
        """Probe Sci-Hub domains if TTL has expired."""
        now = time.time()
        if now - self._last_probe_time < _PROBE_TTL_SECONDS:
            return

        logger.debug("Probing Sci-Hub domains...")
        for domain in self.domains:
            self._probe_single_domain(domain)
        self._last_probe_time = now

    def _probe_single_domain(self, domain: str) -> None:
        """Test if a Sci-Hub domain is reachable."""
        try:
            with httpx.Client(
                timeout=httpx.Timeout(connect=5.0, read=8.0, write=5.0, pool=3.0),
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
                proxy=self.proxy,
            ) as client:
                t0 = time.perf_counter()
                resp = client.get(domain)
                latency_ms = (time.perf_counter() - t0) * 1000

                # Any response means the domain is alive
                is_reachable = resp.status_code in (200, 301, 302, 403, 503)
                self._domain_health[domain] = {
                    "reachable": is_reachable,
                    "status_code": resp.status_code,
                    "latency_ms": round(latency_ms, 1),
                    "last_probe": time.time(),
                }
                logger.debug("Sci-Hub probe %s: status=%s latency=%.0fms", domain, resp.status_code, latency_ms)
        except Exception as exc:
            self._domain_health[domain] = {
                "reachable": False,
                "status_code": 0,
                "latency_ms": 99999.0,
                "last_probe": time.time(),
                "error": str(exc),
            }
            logger.debug("Sci-Hub probe %s: unreachable (%s)", domain, type(exc).__name__)

    def _get_active_domains(self) -> list[str]:
        """Get list of domains that are not in cooldown, sorted by health.

        Domains in cooldown are pushed to the end.
        """
        now = time.time()
        active: list[tuple[str, float]] = []
        cooldown: list[tuple[str, float]] = []

        for domain in self.domains:
            health = self._domain_health.get(domain, {})
            if not health:
                # Never probed — treat as active with default score
                active.append((domain, 0.5))
                continue

            latency = health.get("latency_ms", 99999.0)
            score = 1.0 / (1.0 + latency / 1000.0)

            last_fail = self._failure_cooldowns.get(domain, 0.0)
            if now - last_fail < _FAILURE_COOLDOWN_SECONDS:
                cooldown.append((domain, score))
            else:
                active.append((domain, score))

        # Sort by score descending (higher is better)
        active.sort(key=lambda x: x[1], reverse=True)
        cooldown.sort(key=lambda x: x[1], reverse=True)

        return [d for d, _ in active] + [d for d, _ in cooldown]

    def _record_domain_result(self, domain: str, success: bool, latency_ms: float) -> None:
        """Record download attempt result for adaptive scoring."""
        if domain not in self._domain_health:
            self._domain_health[domain] = {}
        entry = self._domain_health[domain]
        # Exponential moving average for latency
        alpha = 0.2
        old_latency = entry.get("latency_ms", 5000.0)
        entry["latency_ms"] = alpha * latency_ms + (1.0 - alpha) * old_latency if latency_ms > 0 else old_latency
        entry["last_attempt"] = time.time()
        entry["last_success"] = success

    # ── Download Logic ─────────────────────────────────────────────────────

    def _build_scihub_url(self, doi: str, domain: str) -> str:
        """Build the Sci-Hub URL for a given DOI."""
        encoded_doi = quote(doi, safe="")
        # Sci-Hub supports both /<doi> and /https://doi.org/<doi> formats
        return f"{domain.rstrip('/')}/{encoded_doi}"

    def _try_single_domain(
        self, doi: str, domain: str, output_path: Path
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Attempt download from a single Sci-Hub domain.

        Returns:
            (success, info_entries)
        """
        url = self._build_scihub_url(doi, domain)
        info: list[dict[str, Any]] = []

        client_kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "follow_redirects": True,
            "headers": {
                "User-Agent": self.user_agent,
                "Accept": "application/pdf,text/html,application/octet-stream,*/*",
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
                        "source": "scihub",
                        "reason": f"http_{resp.status_code}",
                    })
                    # Check for Cloudflare
                    if resp.status_code in (403, 503):
                        if self._is_cloudflare_block(resp):
                            info.append({
                                "status": "blocked",
                                "url": str(resp.url),
                                "source": "scihub",
                                "reason": "cloudflare_detected",
                            })
                    return False, info

                content = resp.content or b""

                # Check if the response is directly a PDF
                if self._looks_like_pdf(resp.headers, content):
                    if self._write_pdf(content, output_path):
                        info.append({
                            "status": "downloaded",
                            "url": str(resp.url),
                            "source": "scihub",
                        })
                        return True, info
                    info.append({
                        "status": "non_pdf",
                        "url": str(resp.url),
                        "source": "scihub",
                        "reason": "invalid_pdf_content",
                    })
                    return False, info

                # Check for CAPTCHA
                if self._is_captcha(content):
                    info.append({
                        "status": "blocked",
                        "url": str(resp.url),
                        "source": "scihub",
                        "reason": "captcha_detected",
                    })
                    return False, info

                # Try to extract PDF URL from the HTML page
                html_text = self._decode_html(content)
                pdf_url = self._extract_pdf_url_from_html(content, str(resp.url))

                if pdf_url:
                    info.append({
                        "status": "resolved_from_landing",
                        "url": pdf_url,
                        "source": "scihub",
                    })
                    # Download the resolved PDF
                    pdf_resp = client.get(pdf_url)
                    if pdf_resp.status_code < 400:
                        pdf_content = pdf_resp.content or b""
                        if self._looks_like_pdf(pdf_resp.headers, pdf_content):
                            if self._write_pdf(pdf_content, output_path):
                                info.append({
                                    "status": "downloaded",
                                    "url": str(pdf_resp.url),
                                    "source": "scihub",
                                })
                                return True, info
                        info.append({
                            "status": "non_pdf",
                            "url": str(pdf_resp.url),
                            "source": "scihub",
                            "reason": "resolved_url_not_pdf",
                        })
                    else:
                        info.append({
                            "status": "http_error",
                            "url": str(pdf_resp.url),
                            "source": "scihub",
                            "reason": f"http_{pdf_resp.status_code}",
                        })
                    return False, info

                info.append({
                    "status": "non_pdf",
                    "url": str(resp.url),
                    "source": "scihub",
                    "reason": "no_pdf_found_on_page",
                })
                return False, info

        except Exception as exc:
            info.append({
                "status": "request_failed",
                "url": url,
                "source": "scihub",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return False, info

    # ── Content Analysis Helpers ──────────────────────────────────────────

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
    def _is_captcha(content: bytes) -> bool:
        """Detect CAPTCHA in response content."""
        text = content[:10000].lower()
        captcha_indicators = [
            b"captcha",
            b"recaptcha",
            b"g-recaptcha",
            b"verify you are a human",
            b"are you a human",
            b"please verify",
        ]
        return any(indicator in text for indicator in captcha_indicators)

    @staticmethod
    def _is_cloudflare_block(resp: httpx.Response) -> bool:
        """Check if response indicates Cloudflare blocking."""
        server = (resp.headers.get("Server") or "").lower()
        if "cloudflare" in server:
            return True
        # Check common Cloudflare response headers
        cf_headers = ["cf-ray", "cf-cache-status"]
        for h in cf_headers:
            if resp.headers.get(h):
                return True
        # Check body for Cloudflare challenge
        content = (resp.content or b"")[:10000].lower()
        return b"cloudflare" in content and (b"challenge" in content or b"ray id" in content)

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

    def _extract_pdf_url_from_html(self, content: bytes, base_url: str) -> str | None:
        """Extract PDF URL from Sci-Hub page HTML."""
        html_text = self._decode_html(content[:200000])

        # Try known Sci-Hub patterns first
        for pattern in _SCIHUB_PDF_PATTERNS:
            match = pattern.search(html_text)
            if match:
                pdf_url = match.group(1)
                if pdf_url.startswith("//"):
                    pdf_url = "https:" + pdf_url
                elif not pdf_url.startswith("http"):
                    pdf_url = urljoin(base_url, pdf_url)
                if pdf_url.lower().endswith(".pdf") or "pdf" in pdf_url.lower():
                    return pdf_url

        # Generic PDF link regex
        generic_patterns = [
            re.compile(r'''href\s*=\s*["']([^"']+?\.pdf(?:\?[^"']*)?)["']''', re.IGNORECASE),
            re.compile(r'''src\s*=\s*["']([^"']+?\.pdf(?:\?[^"']*)?)["']''', re.IGNORECASE),
        ]
        for pattern in generic_patterns:
            for match in pattern.finditer(html_text):
                pdf_url = match.group(1)
                if pdf_url.startswith("//"):
                    pdf_url = "https:" + pdf_url
                elif not pdf_url.startswith("http"):
                    pdf_url = urljoin(base_url, pdf_url)
                return pdf_url

        return None

    @staticmethod
    def _write_pdf(content: bytes, path: Path) -> bool:
        """Write PDF content to file with validation."""
        if not content or len(content) < 1024:
            return False

        tmp_path = path.with_suffix(path.suffix + ".part")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(content)

            # Validate PDF
            if not SciHubClient._is_valid_pdf(tmp_path):
                tmp_path.unlink(missing_ok=True)
                return False

            tmp_path.replace(path)
            logger.info("Sci-Hub downloaded: %s", path.name)
            return True
        except Exception as exc:
            logger.warning("Failed writing Sci-Hub PDF %s: %s", path, exc)
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
