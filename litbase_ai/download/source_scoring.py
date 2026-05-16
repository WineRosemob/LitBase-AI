from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_ALPHA = 0.12
_DEFAULT_SUCCESS = 0.5
_DEFAULT_LATENCY_MS = 4000.0
_DEFAULT_DATA_DIR = Path.home() / ".litbase-ai"
_DEFAULT_SCORES_PATH = _DEFAULT_DATA_DIR / "download_source_scores.json"


def normalize_source_label(source: str) -> str:
    """Normalize detailed trace labels into stable source buckets."""
    text = (source or "unknown").strip().lower()
    if not text:
        return "unknown"
    text = text.split(":", 1)[0]
    text = text.split("[", 1)[0]
    if text.startswith("metadata."):
        return text
    if text.startswith("raw."):
        return text
    return text.split(".", 1)[0] or "unknown"


class DownloadSourceScorer:
    """Persist lightweight EMA source scores across runs."""

    def __init__(self, scores_path: Path | None = None):
        self.scores_path = scores_path or _DEFAULT_SCORES_PATH

    def record(self, source: str, success: bool, latency_ms: float = 0.0, reason: str = "") -> None:
        key = normalize_source_label(source)
        scores = self._load_scores()
        entry = scores.get(
            key,
            {
                "success_ema": _DEFAULT_SUCCESS,
                "latency_ema": _DEFAULT_LATENCY_MS,
                "attempts": 0,
                "last_error": "",
                "last_update": 0,
            },
        )
        entry["success_ema"] = _ALPHA * (1.0 if success else 0.0) + (1.0 - _ALPHA) * float(entry.get("success_ema", _DEFAULT_SUCCESS))
        if success and latency_ms > 0:
            entry["latency_ema"] = _ALPHA * latency_ms + (1.0 - _ALPHA) * float(entry.get("latency_ema", _DEFAULT_LATENCY_MS))
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["last_error"] = "" if success else reason
        entry["last_update"] = int(time.time())
        scores[key] = entry
        self._save_scores(scores)

    def success_score(self, source: str) -> float:
        key = normalize_source_label(source)
        entry = self._load_scores().get(key) or {}
        return float(entry.get("success_ema", _DEFAULT_SUCCESS))

    def latency_ms(self, source: str) -> float:
        key = normalize_source_label(source)
        entry = self._load_scores().get(key) or {}
        return float(entry.get("latency_ema", _DEFAULT_LATENCY_MS))

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return self._load_scores()

    def _load_scores(self) -> dict[str, dict[str, Any]]:
        if not self.scores_path.exists():
            return {}
        try:
            with self.scores_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return {str(key): value for key, value in data.items() if isinstance(value, dict)}
        except Exception:
            return {}
        return {}

    def _save_scores(self, scores: dict[str, dict[str, Any]]) -> None:
        self.scores_path.parent.mkdir(parents=True, exist_ok=True)
        with self.scores_path.open("w", encoding="utf-8") as fh:
            json.dump(scores, fh, ensure_ascii=False, indent=2)
