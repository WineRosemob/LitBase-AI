from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from litbase_ai.download.inst_proxy import InstProxyClient
from litbase_ai.download.school_db import get_host_for, get_keys_for
from litbase_ai.download.webvpn_login import WebVPNLoginClient


class SchoolDBTest(unittest.TestCase):
    def test_get_host_for_school_and_url_passthrough(self) -> None:
        host = get_host_for("云南大学")
        self.assertIsNotNone(host)
        assert host is not None
        self.assertTrue(host.startswith("https://"))

        self.assertEqual(
            get_host_for("https://webvpn.example.edu.cn/"),
            "https://webvpn.example.edu.cn",
        )

    def test_get_keys_for_unknown_school_returns_default(self) -> None:
        key, iv = get_keys_for("不存在的学校")
        self.assertEqual(key, b"wrdvpnisthebest!")
        self.assertEqual(iv, b"wrdvpnisthebest!")


class InstProxyRewriteTest(unittest.TestCase):
    def test_rewrite_url_falls_back_when_crypto_not_available(self) -> None:
        client = InstProxyClient(
            mode="url_rewrite",
            proxy_url="https://webvpn.example.edu.cn",
        )
        with patch("litbase_ai.download.inst_proxy._get_aes", side_effect=ImportError("no crypto")):
            rewritten = client._rewrite_url("https://example.org/paper.pdf?download=1")

        self.assertTrue(rewritten.startswith("https://webvpn.example.edu.cn/https/"))
        self.assertIn("/example.org/paper.pdf", rewritten)
        self.assertTrue(rewritten.endswith("?download=1"))

    def test_rewrite_url_skips_existing_webvpn_url(self) -> None:
        client = InstProxyClient(mode="url_rewrite", proxy_url="https://webvpn.example.edu.cn")
        url = "https://webvpn.example.edu.cn/https/example.org/paper.pdf"
        self.assertEqual(client._rewrite_url(url), url)

    def test_try_download_resolves_pdf_from_html_gateway(self) -> None:
        client = InstProxyClient(
            mode="url_rewrite",
            proxy_url="https://webvpn.example.edu.cn",
        )

        class _FakeResponse:
            def __init__(self, url: str, status_code: int, headers: dict[str, str], content: bytes):
                self.url = url
                self.status_code = status_code
                self.headers = headers
                self.content = content

        class _FakeHTTPClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, url: str):
                if url.endswith("/pdfft"):
                    html = b'<html><meta name="citation_pdf_url" content="/paper.pdf"></html>'
                    return _FakeResponse(
                        url=url,
                        status_code=200,
                        headers={"Content-Type": "text/html"},
                        content=html,
                    )
                pdf_content = b"%PDF-1.4\\n" + (b"A" * 1200) + b"\\n%%EOF"
                return _FakeResponse(
                    url=url,
                    status_code=200,
                    headers={"Content-Type": "application/pdf"},
                    content=pdf_content,
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "inst_proxy_gateway_test.pdf"
            with patch.object(client, "_client", return_value=_FakeHTTPClient()):
                ok, info = client.try_download(
                    original_url="https://www.sciencedirect.com/science/article/pii/S2666546822000428/pdfft",
                    output_path=out,
                )

            self.assertTrue(ok)
            self.assertTrue(out.exists())
            trace = info.get("trace") or []
            statuses = [item.get("status") for item in trace if isinstance(item, dict)]
            self.assertIn("resolved_from_landing", statuses)
            self.assertIn("downloaded", statuses)

    def test_try_download_uses_browser_fallback_after_non_pdf(self) -> None:
        client = InstProxyClient(
            mode="url_rewrite",
            proxy_url="https://webvpn.example.edu.cn",
        )

        class _FakeResponse:
            def __init__(self, url: str, status_code: int, headers: dict[str, str], content: bytes):
                self.url = url
                self.status_code = status_code
                self.headers = headers
                self.content = content

        class _FakeHTTPClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, url: str):
                return _FakeResponse(
                    url=url,
                    status_code=200,
                    headers={"Content-Type": "text/html"},
                    content=b"<html><body>gateway shell</body></html>",
                )

        browser_trace = [{"status": "downloaded", "url": "https://example.org/final.pdf", "source": "inst_proxy:browser"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "inst_proxy_browser_fallback.pdf"
            with patch.object(client, "_client", return_value=_FakeHTTPClient()), patch.object(
                client,
                "_download_with_playwright",
                return_value=(True, browser_trace),
            ) as fallback_mock:
                ok, info = client.try_download(
                    original_url="https://example.org/pdfft",
                    output_path=out,
                )

            self.assertTrue(ok)
            self.assertEqual(fallback_mock.call_count, 1)
            trace = info.get("trace") or []
            statuses = [item.get("status") for item in trace if isinstance(item, dict)]
            self.assertIn("downloaded", statuses)


class WebVPNLoginSimulationTest(unittest.TestCase):
    def test_parse_cookies_supports_json_and_netscape(self) -> None:
        cookies_json = json.dumps([{"name": "sid", "value": "abc", "domain": ".example.org", "path": "/"}])
        parsed_json = WebVPNLoginClient._parse_cookies(cookies_json)
        self.assertEqual(parsed_json[0]["name"], "sid")
        self.assertEqual(parsed_json[0]["value"], "abc")

        cookies_obj = json.dumps({"cookies": [{"name": "token", "value": "xyz", "domain": "example.org"}]})
        parsed_obj = WebVPNLoginClient._parse_cookies(cookies_obj)
        self.assertEqual(parsed_obj[0]["name"], "token")
        self.assertEqual(parsed_obj[0]["value"], "xyz")

        netscape = (
            "# Netscape HTTP Cookie File\n"
            ".example.org\tTRUE\t/\tFALSE\t0\tsession\tvalue\n"
        )
        parsed_netscape = WebVPNLoginClient._parse_cookies(netscape)
        self.assertEqual(parsed_netscape[0]["name"], "session")
        self.assertEqual(parsed_netscape[0]["value"], "value")

    def test_test_cookies_uses_saved_cookie_and_accepts_non_login_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            client = WebVPNLoginClient(
                vpn_url="https://webvpn.example.edu.cn",
                cookie_dir=tmp_dir,
            )
            client.cookie_file.write_text(
                json.dumps([{"name": "sid", "value": "abc", "domain": ".example.edu.cn", "path": "/"}]),
                encoding="utf-8",
            )

            class _FakeCookieJar:
                def __init__(self) -> None:
                    self.items: list[tuple[str, str]] = []

                def set(self, name: str, value: str, domain: str = "", path: str = "/") -> None:
                    self.items.append((name, value))

            class _FakeResponse:
                status_code = 200
                url = "https://webvpn.example.edu.cn/home"

            class _FakeClient:
                def __init__(self, *args, **kwargs) -> None:
                    self.cookies = _FakeCookieJar()

                def get(self, *args, **kwargs):
                    return _FakeResponse()

            with patch("litbase_ai.download.webvpn_login.httpx.Client", _FakeClient):
                ok, msg = client.test_cookies()

            self.assertTrue(ok)
            self.assertIn("status=200", msg)

    def test_import_export_cookie_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            client = WebVPNLoginClient(
                vpn_url="https://webvpn.example.edu.cn",
                cookie_dir=tmp_dir,
            )
            src = Path(tmp_dir) / "cookies.json"
            src.write_text(
                json.dumps([{"name": "sid", "value": "abc", "domain": ".example.edu.cn", "path": "/"}]),
                encoding="utf-8",
            )

            imported = client.import_cookies_file(src)
            self.assertTrue(imported)

            netscape = client.export_cookies_netscape()
            assert netscape is not None
            self.assertIn("sid", netscape)
            self.assertIn("abc", netscape)


if __name__ == "__main__":
    unittest.main()
