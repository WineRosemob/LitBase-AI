"""EZProxy institutional proxy download source.

EZProxy is a URL-rewriting proxy used by university libraries worldwide.
It rewrites publisher URLs through the library's proxy server, granting
institutional access to subscribed journals.

Usage:
  1. Set EZPROXY_URL_TEMPLATE env var (e.g., "https://login.ezproxy.lib.university.edu/login?url={url}")
  2. Optionally set EZPROXY_COOKIE_FILE for session persistence
  3. Login once via browser to get cookies, then the tool uses them

No browser automation needed — just provide the proxy URL template and cookies.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from litbase_ai.utils.logging import get_logger

logger = get_logger(__name__)


class EZProxyClient:
    """Lightweight EZProxy client with cookie-based session management."""

    def __init__(
        self,
        proxy_template: str,
        cookie_file: str | Path | None = None,
        proxy: str | None = None,
        connect_timeout: float = 20.0,
        read_timeout: float = 30.0,
        user_agent: str = "LitBase-AI/0.4 (+ezproxy)",
    ):
        """
        Args:
            proxy_template: URL template with {url} placeholder.
                Example: 'https://login.ezproxy.lib.university.edu/login?url={url}'
            cookie_file: Path to JSON cookie file (Netscape or simple JSON list).
            proxy: Optional HTTP/SOCKS proxy for the connection itself.
        """
        self.proxy_template = proxy_template.rstrip("/")
        self.cookie_file = Path(cookie_file) if cookie_file else None
        self.proxy = proxy
        self.user_agent = user_agent
        self.timeout = httpx.Timeout(
            connect=max(1.0, float(connect_timeout)),
            read=max(1.0, float(read_timeout)),
            write=max(1.0, float(read_timeout)),
            pool=max(1.0, float(connect_timeout)),
        )
        self._cookies: list[dict[str, str]] = []
        self._load_cookies()

    # ── Public API ────────────────────────────────────────────────────

    def resolve_pdf_url(self, original_url: str) -> str | None:
        """Convert an original publisher URL to an EZProxy URL."""
        if not original_url or not self.proxy_template:
            return None
        # Check if URL is already proxied
        if "ezproxy" in original_url.lower():
            return original_url
        return self._make_proxy_url(original_url)

    def try_download(self, original_url: str, output_path: Path) -> tuple[bool, dict[str, Any]]:
        """Download a paper through EZProxy.

        Args:
            original_url: The original publisher PDF/landing page URL.
            output_path: Where to save the PDF.

        Returns:
            (success, info_dict)
        """
        if not original_url:
            return False, {"source": "ezproxy", "status": "failed", "reason": "no_url"}

        proxy_url = self.resolve_pdf_url(original_url)
        if not proxy_url:
            return False, {"source": "ezproxy", "status": "failed", "reason": "no_proxy_template"}

        return self._download_via_proxy(proxy_url, output_path)

    def test_connection(self) -> tuple[bool, str]:
        """Test if EZProxy is reachable and session is valid."""
        if not self.proxy_template:
            return False, "no template configured"

        # Extract the base domain from template for testing
        try:
            test_url = self.proxy_template.split("{url}")[0].rstrip("?&= ")
            if not test_url.startswith("http"):
                # Try to construct a reasonable test URL
                parsed = urlparse(self.proxy_template)
                test_url = f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return False, "invalid template"

        trace: list[dict] = []
        try:
            with self._client() as client:
                resp = client.get(test_url)
                if resp.status_code < 400:
                    return True, f"reachable (status={resp.status_code})"
                # Check if redirected to login
                if "login" in str(resp.url).lower() or "ezproxy" in str(resp.url).lower():
                    return False, f"redirected to login (status={resp.status_code})"
                return False, f"http_{resp.status_code}"
        except Exception as exc:
            return False, f"{type(exc).__name__}"

    def set_cookies(self, cookies: list[dict[str, str]]) -> None:
        """Set session cookies manually."""
        self._cookies = cookies
        if self.cookie_file:
            self._save_cookies()

    # ── Internal ──────────────────────────────────────────────────────

    def _make_proxy_url(self, original_url: str) -> str:
        """Construct the EZProxy URL from template."""
        # EZProxy expects the URL to be encoded
        encoded = quote(original_url, safe="")
        return self.proxy_template.replace("{url}", encoded)

    def _client(self) -> httpx.Client:
        """Create an httpx.Client with cookies and proxy."""
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

        client = httpx.Client(**client_kwargs)

        # Set cookies
        for cookie in self._cookies:
            client.cookies.set(
                name=cookie.get("name", ""),
                value=cookie.get("value", ""),
                domain=cookie.get("domain"),
                path=cookie.get("path", "/"),
            )

        return client

    def _download_via_proxy(self, proxy_url: str, output_path: Path) -> tuple[bool, dict[str, Any]]:
        """Attempt download through EZProxy."""
        trace: list[dict[str, Any]] = []

        trace.append({"status": "attempt", "url": proxy_url, "source": "ezproxy"})

        try:
            with self._client() as client:
                resp = client.get(proxy_url)

                if resp.status_code >= 400:
                    # Detect if we got redirected to login page
                    if resp.status_code in (302, 301, 303):
                        redirect_url = resp.headers.get("Location", "")
                        if "login" in redirect_url.lower() or "sso" in redirect_url.lower():
                            trace.append({
                                "status": "http_error",
                                "url": str(resp.url),
                                "source": "ezproxy",
                                "reason": "session_expired_need_login",
                            })
                            return False, {"source": "ezproxy", "status": "failed", "trace": trace}

                    trace.append({
                        "status": "http_error",
                        "url": str(resp.url),
                        "source": "ezproxy",
                        "reason": f"http_{resp.status_code}",
                    })
                    return False, {"source": "ezproxy", "status": "failed", "trace": trace}

                content = resp.content or b""

                if self._looks_like_pdf(resp.headers, content):
                    if self._write_pdf(content, output_path):
                        trace.append({
                            "status": "downloaded",
                            "url": str(resp.url),
                            "source": "ezproxy",
                        })
                        return True, {"source": "ezproxy", "status": "downloaded", "trace": trace}
                    trace.append({
                        "status": "non_pdf",
                        "url": str(resp.url),
                        "source": "ezproxy",
                        "reason": "invalid_pdf_content",
                    })
                    return False, {"source": "ezproxy", "status": "failed", "trace": trace}

                # Not a PDF — could be login page or HTML
                trace.append({
                    "status": "non_pdf",
                    "url": str(resp.url),
                    "source": "ezproxy",
                    "reason": "content_not_pdf",
                })
                return False, {"source": "ezproxy", "status": "failed", "trace": trace}

        except Exception as exc:
            trace.append({
                "status": "request_failed",
                "url": proxy_url,
                "source": "ezproxy",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return False, {"source": "ezproxy", "status": "failed", "trace": trace}

    def _load_cookies(self) -> None:
        """Load cookies from file if available."""
        if not self.cookie_file or not self.cookie_file.exists():
            return
        try:
            data = json.loads(self.cookie_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._cookies = [
                    {"name": c.get("name", ""), "value": c.get("value", ""),
                     "domain": c.get("domain", ""), "path": c.get("path", "/")}
                    for c in data if isinstance(c, dict)
                ]
                logger.info("Loaded %d EZProxy cookies from %s", len(self._cookies), self.cookie_file)
        except Exception as exc:
            logger.debug("Failed to load EZProxy cookies: %s", exc)

    def _save_cookies(self) -> None:
        """Save cookies to file."""
        if not self.cookie_file:
            return
        try:
            self.cookie_file.parent.mkdir(parents=True, exist_ok=True)
            self.cookie_file.write_text(json.dumps(self._cookies, ensure_ascii=False, indent=2))
            logger.debug("Saved %d EZProxy cookies to %s", len(self._cookies), self.cookie_file)
        except Exception as exc:
            logger.debug("Failed to save EZProxy cookies: %s", exc)

    # ── Content Helpers ───────────────────────────────────────────────

    @staticmethod
    def _looks_like_pdf(headers: httpx.Headers, content: bytes) -> bool:
        content_type = (headers.get("Content-Type") or "").lower()
        head = content[:2048]
        if b"%PDF" in head:
            return True
        if "application/pdf" in content_type and len(content) > 1024:
            return True
        return False

    @staticmethod
    def _write_pdf(content: bytes, path: Path) -> bool:
        if not content or len(content) < 1024:
            return False
        tmp_path = path.with_suffix(path.suffix + ".part")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(content)
            if not EZProxyClient._is_valid_pdf(tmp_path):
                tmp_path.unlink(missing_ok=True)
                return False
            tmp_path.replace(path)
            logger.info("EZProxy downloaded: %s", path.name)
            return True
        except Exception as exc:
            logger.warning("Failed writing EZProxy PDF %s: %s", path, exc)
            tmp_path.unlink(missing_ok=True)
            return False

    @staticmethod
    def _is_valid_pdf(path: Path) -> bool:
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
