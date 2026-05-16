from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from litbase_ai.scoring.llm_score import LLMScorer


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": '{"llm_score": 88, "reason": "ok"}'}}]}


class _FakeClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, json: dict) -> _FakeResponse:
        type(self).calls += 1
        if type(self).calls == 1:
            raise httpx.TimeoutException("timeout")
        return _FakeResponse()


class LLMScorerRetryTest(unittest.TestCase):
    def test_call_api_retries_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompt.txt"
            prompt_path.write_text("Topic: {topic}\nTitle: {title}\nAbstract: {abstract}", encoding="utf-8")

            scorer = LLMScorer(
                api_key="test-key",
                base_url="https://example.com",
                model="test-model",
                prompt_template_path=prompt_path,
                max_retries=2,
                retry_backoff_seconds=0,
            )

            _FakeClient.calls = 0
            with patch("litbase_ai.scoring.llm_score.httpx.Client", _FakeClient):
                result = scorer._call_api("hello")  # noqa: SLF001

        self.assertIsNotNone(result)
        self.assertEqual(_FakeClient.calls, 2)


if __name__ == "__main__":
    unittest.main()
