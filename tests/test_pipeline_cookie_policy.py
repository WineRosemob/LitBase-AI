from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from litbase_ai.config import AppConfig
from litbase_ai.download.cookie_bootstrap import CookieBootstrapPolicy
from litbase_ai.pipeline import LitBasePipeline


class PipelineCookiePolicyTest(unittest.TestCase):
    def test_policy_disables_proxy_flows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env.test"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENALEX_MAILTO=test@example.com",
                        "UNPAYWALL_EMAIL=test@example.com",
                        "LLM_API_KEY=test-key",
                        "WEBVPN_URL=https://webvpn.example.edu",
                        "WEBVPN_AUTO_LOGIN=true",
                        "ENABLE_INST_PROXY=true",
                        "ENABLE_EZPROXY=true",
                        "EZPROXY_TEMPLATE=https://ezproxy.example/login?url={url}",
                        f"EZPROXY_COOKIE_FILE={Path(tmp_dir) / 'ezproxy_cookie.json'}",
                    ]
                ),
                encoding="utf-8",
            )
            config = AppConfig.load(env_file=env_path)
            output_dir = Path(tmp_dir) / "out"

            policy = CookieBootstrapPolicy(
                disable_inst_proxy=True,
                disable_ezproxy=True,
                disabled_publishers=["elsevier"],
            )

            with patch("litbase_ai.pipeline.PDFDownloader") as downloader_cls, patch(
                "litbase_ai.pipeline.load_cookie_bootstrap_policy",
                return_value=policy,
            ), patch(
                "litbase_ai.pipeline.publisher_host_keywords",
                return_value=["sciencedirect.com"],
            ):
                LitBasePipeline(
                    config=config,
                    output_dir=output_dir,
                    topic="climate",
                    enable_llm=False,
                )

            kwargs = downloader_cls.call_args.kwargs

        self.assertFalse(kwargs["enable_inst_proxy"])
        self.assertFalse(kwargs["enable_ezproxy"])
        self.assertEqual(kwargs["inst_proxy_disabled_host_keywords"], ["sciencedirect.com"])


if __name__ == "__main__":
    unittest.main()
