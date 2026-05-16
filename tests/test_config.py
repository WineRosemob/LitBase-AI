from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from litbase_ai.config import AppConfig


class AppConfigTest(unittest.TestCase):
    def test_defaults_disable_legacy_download_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env.test"
            env_path.write_text("", encoding="utf-8")

            config = AppConfig.load(env_file=env_path)

        self.assertFalse(config.enable_scihub)
        self.assertFalse(config.enable_libgen)
        self.assertTrue(config.enable_arxiv_download)
        self.assertFalse(config.enable_ezproxy)
        self.assertFalse(config.enable_inst_proxy)

    def test_load_reads_download_network_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env.test"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENALEX_MAILTO=test@example.com",
                        "UNPAYWALL_EMAIL=test@example.com",
                        "CORE_API_KEY=core-secret",
                        "LLM_CONNECT_TIMEOUT=11",
                        "LLM_READ_TIMEOUT=52",
                        "LLM_MAX_RETRIES=4",
                        "LLM_RETRY_BACKOFF_SECONDS=1.5",
                        "DOWNLOAD_PROXY=http://127.0.0.1:7890",
                        "DOWNLOAD_CONNECT_TIMEOUT=12",
                        "DOWNLOAD_READ_TIMEOUT=34",
                        "DOWNLOAD_REQUEST_DELAY_MIN=0.1",
                        "DOWNLOAD_REQUEST_DELAY_MAX=0.4",
                        "SEARCH_SOURCE_WORKERS=6",
                    ]
                ),
                encoding="utf-8",
            )

            config = AppConfig.load(env_file=env_path)

        self.assertEqual(config.core_api_key, "core-secret")
        self.assertEqual(config.llm_connect_timeout, 11.0)
        self.assertEqual(config.llm_read_timeout, 52.0)
        self.assertEqual(config.llm_max_retries, 4)
        self.assertEqual(config.llm_retry_backoff_seconds, 1.5)
        self.assertEqual(config.download_proxy, "http://127.0.0.1:7890")
        self.assertEqual(config.download_connect_timeout, 12.0)
        self.assertEqual(config.download_read_timeout, 34.0)
        self.assertEqual(config.download_request_delay_min, 0.1)
        self.assertEqual(config.download_request_delay_max, 0.4)
        self.assertEqual(config.search_source_workers, 6)

    def test_public_placeholders_are_treated_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env.test"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENALEX_MAILTO=your_email@example.com",
                        "UNPAYWALL_EMAIL=your_email@example.com",
                        "LLM_API_KEY=your_llm_api_key_here",
                        "WEBVPN_URL=https://your-institution-proxy.example.com",
                    ]
                ),
                encoding="utf-8",
            )

            config = AppConfig.load(env_file=env_path)

        self.assertIsNone(config.openalex_mailto)
        self.assertIsNone(config.unpaywall_email)
        self.assertIsNone(config.llm_api_key)
        self.assertIsNone(config.webvpn_url)


if __name__ == "__main__":
    unittest.main()
