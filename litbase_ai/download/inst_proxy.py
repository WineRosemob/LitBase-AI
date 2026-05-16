"""Institutional proxy download source with per-school WebVPN URL conversion.

Supports:
1. HTTP_PROXY mode — route ALL traffic through a proxy server
2. URL_REWRITE mode — AES-CFB encrypted WebVPN URL (100+ CN universities)
3. EZPROXY mode — URL template rewriting

School database at litbase_ai/data/webvpn.json provides crypto keys
for correct AES-CFB encryption per university.

Config via env:
  INST_PROXY_MODE=http_proxy|url_rewrite|ezproxy
  INST_PROXY_URL=http://proxy.university.edu:8080  (http_proxy mode)
  INST_PROXY_SCHOOL=<your_school_name_in_webvpn_json>  (url_rewrite mode, looks up host+keys from DB)
  INST_PROXY_COOKIE_FILE=/path/to/cookies.json
"""

from __future__ import annotations
import binascii
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from litbase_ai.download.candidate_utils import extract_pdf_urls_from_html
from litbase_ai.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover
    PlaywrightTimeoutError = Exception
    sync_playwright = None

_BROWSER_FALLBACK_UA = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) "
    "Gecko/20100101 Firefox/126.0"
)


def _get_aes():
    try:
        from Crypto.Cipher import AES
        return AES
    except ImportError:
        try:
            from Cryptodome.Cipher import AES
            return AES
        except ImportError:
            raise ImportError("pycryptodome required for WebVPN URL encryption: pip install pycryptodome")


class InstProxyClient:
    """Institutional proxy client with school-aware URL rewriting."""

    def __init__(
        self,
        mode: str = "http_proxy",
        proxy_url: str | None = None,
        school_name: str | None = None,
        cookie_file: str | Path | None = None,
        disabled_host_keywords: list[str] | None = None,
        connect_timeout: float = 20.0,
        read_timeout: float = 30.0,
        enable_browser_fallback: bool = True,
        browser_headless: bool = True,
        browser_timeout_ms: int = 45000,
        user_agent: str = "LitBase-AI/0.4 (+institutional-proxy)",
    ):
        self.mode = mode
        self.proxy_url = (proxy_url or "").strip()
        self.school_name = school_name
        self.cookie_file = Path(cookie_file) if cookie_file else None
        self.user_agent = user_agent
        self.timeout = httpx.Timeout(
            connect=max(1.0, float(connect_timeout)),
            read=max(1.0, float(read_timeout)),
            write=max(1.0, float(read_timeout)),
            pool=max(1.0, float(connect_timeout)),
        )
        self.enable_browser_fallback = bool(enable_browser_fallback)
        self.browser_headless = bool(browser_headless)
        self.browser_timeout_ms = max(5000, int(browser_timeout_ms))
        self.disabled_host_keywords = [
            str(item).strip().lower() for item in (disabled_host_keywords or []) if str(item).strip()
        ]

        # Resolve school host and keys
        self._school_host: str = ""
        self._school_key: bytes = b"wrdvpnisthebest!"
        self._school_iv: bytes = b"wrdvpnisthebest!"

        if school_name:
            self._resolve_school()

        # If proxy_url is set but school_name isn't, use proxy_url as the base
        if not self._school_host and self.proxy_url:
            self._school_host = self.proxy_url.rstrip("/")

        self._cookies: list[dict[str, str]] = []
        self._load_cookies()

    # ── Public ─────────────────────────────────────────────────────────

    def try_download(self, original_url: str, output_path: Path) -> tuple[bool, dict[str, Any]]:
        if not original_url:
            return False, {"source": "inst_proxy", "status": "failed", "reason": "no_url"}
        if self._is_disabled_by_host_policy(original_url):
            return (
                False,
                {
                    "source": "inst_proxy",
                    "status": "failed",
                    "reason": "publisher_disabled_by_policy",
                    "trace": [
                        {
                            "status": "blocked",
                            "url": original_url,
                            "source": "inst_proxy",
                            "reason": "publisher_disabled_by_policy",
                        }
                    ],
                },
            )

        if self.mode == "url_rewrite":
            rewritten = self._rewrite_url(original_url)
            return self._download_direct(rewritten, output_path)
        elif self.mode == "ezproxy":
            rewritten = self._rewrite_ezproxy(original_url)
            return self._download_direct(rewritten, output_path)
        else:
            return self._download_via_proxy(original_url, output_path)

    def test_connection(self) -> tuple[bool, str]:
        if self.mode == "url_rewrite" and self._school_host:
            try:
                with self._client(use_proxy=False) as c:
                    r = c.get(self._school_host)
                    return r.status_code < 500, f"status={r.status_code}"
            except Exception as e:
                return False, str(e)
        return False, "not configured"

    def set_cookies(self, cookies: list[dict[str, str]]):
        self._cookies = cookies
        if self.cookie_file:
            self._save_cookies()

    # ── School Resolution ────────────────────────────────────────────

    def _resolve_school(self):
        """Look up school host and crypto keys from the database."""
        try:
            from litbase_ai.download.school_db import get_host_for, get_keys_for
            host = get_host_for(self.school_name)
            if host:
                self._school_host = host
                self._school_key, self._school_iv = get_keys_for(self.school_name)
                logger.info(
                    "[InstProxy] School=%s host=%s",
                    self.school_name, self._school_host,
                )
            else:
                logger.warning("[InstProxy] School not found in DB: %s", self.school_name)
        except Exception as e:
            logger.warning("[InstProxy] School lookup failed: %s", e)

    # ── URL Rewriting ────────────────────────────────────────────────

    def _rewrite_url(self, original_url: str) -> str:
        """Convert a publisher URL to a WebVPN-proxied URL using AES-CFB."""
        if "webvpn" in original_url.lower() or "wvpn" in original_url.lower():
            return original_url

        base = self._school_host or self.proxy_url or ""
        base = base.rstrip("/")
        if not base:
            return original_url

        parsed = urlparse(original_url)
        hostname = parsed.hostname
        if not hostname:
            return original_url

        scheme = parsed.scheme.lower()
        port = parsed.port
        path = parsed.path or "/"
        query = parsed.query

        try:
            AES = _get_aes()
            cipher = AES.new(self._school_key, AES.MODE_CFB, self._school_iv, segment_size=128)
            encrypted = cipher.encrypt(hostname.encode("utf-8"))
            encrypted_hex = binascii.hexlify(self._school_iv).decode() + binascii.hexlify(encrypted).decode()
        except Exception:
            encrypted_hex = hostname

        scheme_part = f"{scheme}-{port}" if port else scheme
        result = f"{base}/{scheme_part}/{encrypted_hex}{path}"
        if query:
            result += f"?{query}"
        return result

    def _rewrite_ezproxy(self, original_url: str) -> str:
        """EZProxy URL template rewriting."""
        from urllib.parse import quote
        return (self.proxy_url or "").replace("{url}", quote(original_url, safe=""))

    # ── HTTP Client ──────────────────────────────────────────────────

    def _client(self, use_proxy: bool = True) -> httpx.Client:
        client = httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/pdf,text/html,application/octet-stream,*/*",
            },
            trust_env=False,
        )
        if use_proxy and self.mode == "http_proxy" and self.proxy_url:
            client.proxies = {"http://": self.proxy_url, "https://": self.proxy_url}

        for c in self._cookies:
            client.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))
        return client

    # ── Download ─────────────────────────────────────────────────────

    def _download_direct(self, url: str, path: Path) -> tuple[bool, dict[str, Any]]:
        trace = [{"status": "attempt", "url": url, "source": "inst_proxy"}]
        try:
            with self._client(use_proxy=False) as c:
                r = c.get(url)
                if r.status_code >= 400:
                    trace.append({"status": "http_error", "url": str(r.url), "source": "inst_proxy",
                                  "reason": f"http_{r.status_code}"})
                    return False, {"source": "inst_proxy", "status": "failed", "trace": trace}

                content = r.content or b""
                if self._looks_like_pdf(r.headers, content) and self._write_pdf(content, path):
                    trace.append({"status": "downloaded", "url": str(r.url), "source": "inst_proxy"})
                    return True, {"source": "inst_proxy", "status": "downloaded", "trace": trace}

                # Some publishers return an HTML gateway page first (e.g. /pdfft).
                # Try extracting candidate PDF links and download again.
                html = self._decode_text(content)
                resolved_pdf_urls = extract_pdf_urls_from_html(html, str(r.url), limit=8) if html else []
                if resolved_pdf_urls:
                    for candidate in resolved_pdf_urls:
                        proxied_candidate = self._proxy_candidate_url(candidate)
                        trace.append(
                            {
                                "status": "resolved_from_landing",
                                "url": proxied_candidate,
                                "source": "inst_proxy",
                            }
                        )
                        rr = c.get(proxied_candidate)
                        if rr.status_code >= 400:
                            trace.append(
                                {
                                    "status": "http_error",
                                    "url": str(rr.url),
                                    "source": "inst_proxy",
                                    "reason": f"http_{rr.status_code}",
                                }
                            )
                            continue
                        content2 = rr.content or b""
                        if self._looks_like_pdf(rr.headers, content2) and self._write_pdf(content2, path):
                            trace.append({"status": "downloaded", "url": str(rr.url), "source": "inst_proxy"})
                            return True, {"source": "inst_proxy", "status": "downloaded", "trace": trace}
                        trace.append(
                            {
                                "status": "non_pdf",
                                "url": str(rr.url),
                                "source": "inst_proxy",
                                "reason": "resolved_link_not_pdf",
                            }
                        )

                if self.enable_browser_fallback:
                    browser_ok, browser_trace = self._download_with_playwright(url=url, path=path)
                    trace.extend(browser_trace)
                    if browser_ok:
                        return True, {"source": "inst_proxy", "status": "downloaded", "trace": trace}

                trace.append(
                    {
                        "status": "non_pdf",
                        "url": str(r.url),
                        "source": "inst_proxy",
                        "reason": "content_not_pdf",
                    }
                )
                return False, {"source": "inst_proxy", "status": "failed", "trace": trace}
        except Exception as e:
            trace.append({"status": "request_failed", "url": url, "source": "inst_proxy",
                          "reason": f"{type(e).__name__}: {e}"})
            return False, {"source": "inst_proxy", "status": "failed", "trace": trace}

    def _download_via_proxy(self, url: str, path: Path) -> tuple[bool, dict[str, Any]]:
        return self._download_direct(url, path)  # Same logic, proxy handled by client

    # ── Cookie I/O ────────────────────────────────────────────────────

    def _load_cookies(self):
        cookies: list[dict[str, str]] = []
        if self.cookie_file and self.cookie_file.exists():
            try:
                loaded = json.loads(self.cookie_file.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    cookies.extend([c for c in loaded if isinstance(c, dict)])
            except Exception:
                pass

            # Optional publisher SSO cookie bundle from webvpn_login.py
            # This helps with sites that require an additional publisher login.
            publisher_cookie_file = self.cookie_file.parent / "publisher_cookies.json"
            if publisher_cookie_file.exists():
                try:
                    loaded_pub = json.loads(publisher_cookie_file.read_text(encoding="utf-8"))
                    if isinstance(loaded_pub, list):
                        cookies.extend([c for c in loaded_pub if isinstance(c, dict)])
                except Exception:
                    pass

        # Deduplicate by (name, domain, path) while preserving first-seen order.
        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for c in cookies:
            name = str(c.get("name") or "")
            domain = str(c.get("domain") or "")
            path = str(c.get("path") or "/")
            if not name:
                continue
            key = (name, domain, path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        self._cookies = deduped
        if self._cookies:
            logger.info("[InstProxy] Loaded %d cookies", len(self._cookies))

    def _save_cookies(self):
        if not self.cookie_file:
            return
        try:
            self.cookie_file.parent.mkdir(parents=True, exist_ok=True)
            self.cookie_file.write_text(json.dumps(self._cookies, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _looks_like_pdf(headers, content):
        if b"%PDF" in content[:2048]:
            return True
        if "application/pdf" in (headers.get("Content-Type") or "").lower() and len(content) > 1024:
            return True
        return False

    @staticmethod
    def _write_pdf(content, path):
        if not content or len(content) < 1024:
            return False
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(content)
            if not InstProxyClient._is_valid_pdf(tmp):
                tmp.unlink(missing_ok=True)
                return False
            tmp.replace(path)
            logger.info("InstProxy downloaded: %s", path.name)
            return True
        except Exception as e:
            logger.warning("InstProxy write failed: %s", e)
            tmp.unlink(missing_ok=True)
            return False

    @staticmethod
    def _is_valid_pdf(path):
        try:
            size = path.stat().st_size
            if size < 1000:
                return False
            with path.open("rb") as f:
                if f.read(5) != b"%PDF-":
                    return False
                f.seek(max(0, size - 2048))
                return b"%%EOF" in f.read() or size > 4096
        except OSError:
            return False

    def _proxy_candidate_url(self, url: str) -> str:
        if self.mode == "url_rewrite":
            return self._rewrite_url(url)
        if self.mode == "ezproxy":
            return self._rewrite_ezproxy(url)
        return url

    def _is_disabled_by_host_policy(self, url: str) -> bool:
        if not self.disabled_host_keywords:
            return False
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        host = (parsed.netloc or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return False
        for keyword in self.disabled_host_keywords:
            if keyword and (keyword in host or host.endswith(keyword)):
                return True
        return False

    @staticmethod
    def _decode_text(content: bytes) -> str:
        if not content:
            return ""
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1", errors="ignore")

    def _download_with_playwright(self, url: str, path: Path) -> tuple[bool, list[dict[str, Any]]]:
        trace: list[dict[str, Any]] = [
            {"status": "browser_fallback_attempt", "url": url, "source": "inst_proxy"}
        ]
        if sync_playwright is None:
            trace.append(
                {
                    "status": "browser_fallback_unavailable",
                    "url": url,
                    "source": "inst_proxy",
                    "reason": "playwright_not_installed",
                }
            )
            return False, trace

        captured: list[tuple[str, bytes]] = []
        attempted_request_urls: set[str] = set()

        def on_response(resp) -> None:
            if captured:
                return
            try:
                ctype = (resp.headers.get("content-type") or "").lower()
            except Exception:
                ctype = ""
            resp_url = str(resp.url)
            if "application/pdf" not in ctype and not resp_url.lower().endswith(".pdf"):
                return
            try:
                body = resp.body()
            except Exception:
                return
            if body and len(body) > 1024:
                captured.append((resp_url, body))

        def try_request_download(context, candidate_url: str) -> bool:
            normalized = str(candidate_url or "").strip()
            if not normalized or normalized in attempted_request_urls:
                return False
            attempted_request_urls.add(normalized)
            trace.append({"status": "browser_request_attempt", "url": normalized, "source": "inst_proxy"})
            try:
                api_resp = context.request.get(
                    normalized,
                    timeout=self.browser_timeout_ms,
                    headers={"Accept": "application/pdf,application/octet-stream,*/*"},
                )
                if int(api_resp.status) >= 400:
                    trace.append(
                        {
                            "status": "http_error",
                            "url": normalized,
                            "source": "inst_proxy",
                            "reason": f"http_{api_resp.status}",
                        }
                    )
                    return False
                body = api_resp.body() or b""
                headers = api_resp.headers or {}
                if self._looks_like_pdf(headers, body) and self._write_pdf(body, path):
                    trace.append({"status": "downloaded", "url": normalized, "source": "inst_proxy:browser_request"})
                    return True
                trace.append(
                    {
                        "status": "non_pdf",
                        "url": normalized,
                        "source": "inst_proxy",
                        "reason": "browser_request_not_pdf",
                    }
                )
                return False
            except Exception as exc:
                trace.append(
                    {
                        "status": "request_failed",
                        "url": normalized,
                        "source": "inst_proxy",
                        "reason": f"browser_request_failed: {type(exc).__name__}: {exc}",
                    }
                )
                return False

        try:
            with sync_playwright() as playwright:
                browser = playwright.firefox.launch(headless=self.browser_headless)
                context = browser.new_context(
                    user_agent=_BROWSER_FALLBACK_UA,
                    accept_downloads=True,
                )

                cookies = self._to_playwright_cookies(base_url=url)
                if cookies:
                    context.add_cookies(cookies)

                # Fast path: direct API request with shared browser cookies.
                if try_request_download(context, url):
                    context.close()
                    browser.close()
                    return True, trace

                page = context.new_page()
                page.on("response", on_response)
                page.goto(url, wait_until="networkidle", timeout=self.browser_timeout_ms)

                if not captured:
                    html = page.content()
                    resolved = extract_pdf_urls_from_html(html, page.url, limit=8)
                    for candidate in resolved:
                        proxied_candidate = self._proxy_candidate_url(candidate)
                        trace.append(
                            {
                                "status": "browser_resolved_from_landing",
                                "url": proxied_candidate,
                                "source": "inst_proxy",
                            }
                        )
                        if try_request_download(context, proxied_candidate):
                            break
                        page.goto(proxied_candidate, wait_until="networkidle", timeout=self.browser_timeout_ms)
                        if captured:
                            break

                context.close()
                browser.close()
        except PlaywrightTimeoutError as exc:
            trace.append(
                {
                    "status": "browser_fallback_failed",
                    "url": url,
                    "source": "inst_proxy",
                    "reason": f"playwright_timeout: {exc}",
                }
            )
            return False, trace
        except Exception as exc:
            trace.append(
                {
                    "status": "browser_fallback_failed",
                    "url": url,
                    "source": "inst_proxy",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            return False, trace

        if captured:
            resolved_url, body = captured[0]
            if self._write_pdf(body, path):
                trace.append({"status": "downloaded", "url": resolved_url, "source": "inst_proxy:browser"})
                return True, trace
            trace.append(
                {
                    "status": "browser_fallback_failed",
                    "url": resolved_url,
                    "source": "inst_proxy",
                    "reason": "invalid_pdf_content",
                }
            )
            return False, trace

        trace.append(
            {
                "status": "browser_fallback_failed",
                "url": url,
                "source": "inst_proxy",
                "reason": "no_pdf_response_captured",
            }
        )
        return False, trace

    def _to_playwright_cookies(self, base_url: str) -> list[dict[str, Any]]:
        cookies: list[dict[str, Any]] = []
        for c in self._cookies:
            name = str(c.get("name") or "").strip()
            value = str(c.get("value") or "")
            if not name:
                continue
            entry: dict[str, Any] = {"name": name, "value": value, "path": str(c.get("path") or "/")}
            domain = c.get("domain")
            if isinstance(domain, str) and domain.strip():
                entry["domain"] = domain.strip()
            else:
                entry["url"] = base_url
            if "secure" in c:
                entry["secure"] = bool(c.get("secure"))
            if "httpOnly" in c:
                entry["httpOnly"] = bool(c.get("httpOnly"))
            cookies.append(entry)
        return cookies
