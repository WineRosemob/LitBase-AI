from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class CacheManager:
    """Simple JSON file cache for expensive scoring operations."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _digest(self, raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _cache_path(self, namespace: str, raw_key: str) -> Path:
        ns_dir = self.cache_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{self._digest(raw_key)}.json"

    def get(self, namespace: str, raw_key: str) -> dict[str, Any] | None:
        path = self._cache_path(namespace, raw_key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, namespace: str, raw_key: str, value: dict[str, Any]) -> None:
        path = self._cache_path(namespace, raw_key)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
