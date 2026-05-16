from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from litbase_ai.config import AppConfig
from litbase_ai.download.cookie_bootstrap import CookieBootstrapPolicy
from litbase_ai.pipeline import LitBasePipeline


class _FakeWebVPNClient:
    def __init__(self, vpn_url: str):
        self.vpn_url = vpn_url
        self.cookie_file = Path("/tmp/litbase_fake_webvpn_cookie.json")

    def test_cookies(self) -> tuple[bool, str]:
        return True, "status=200"


class PipelineWebVPNAutoLoginTest(unittest.TestCase):
    def test_init_auto_enables_inst_proxy_when_webvpn_cookie_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env.test"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENALEX_MAILTO=test@example.com",
                        "UNPAYWALL_EMAIL=test@example.com",
                        "LLM_API_KEY=test-llm-key",
                        "WEBVPN_URL=https://webvpn.example.edu",
                        "WEBVPN_AUTO_LOGIN=true",
                        "ENABLE_INST_PROXY=false",
                    ]
                ),
                encoding="utf-8",
            )
            config = AppConfig.load(env_file=env_path)
            output_dir = Path(tmp_dir) / "out"

            with patch("litbase_ai.pipeline.PDFDownloader") as downloader_cls, patch(
                "litbase_ai.download.webvpn_login.WebVPNLoginClient",
                _FakeWebVPNClient,
            ), patch(
                "litbase_ai.pipeline.load_cookie_bootstrap_policy",
                return_value=CookieBootstrapPolicy(),
            ):
                LitBasePipeline(
                    config=config,
                    output_dir=output_dir,
                    topic="climate",
                    enable_llm=False,
                )

            kwargs = downloader_cls.call_args.kwargs
            self.assertTrue(kwargs["enable_inst_proxy"])
            self.assertEqual(kwargs["inst_proxy_mode"], "url_rewrite")
            self.assertEqual(kwargs["inst_proxy_url"], "https://webvpn.example.edu")
            self.assertEqual(kwargs["inst_proxy_cookie_file"], "/tmp/litbase_fake_webvpn_cookie.json")


if __name__ == "__main__":
    unittest.main()
