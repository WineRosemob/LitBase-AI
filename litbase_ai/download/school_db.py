"""School database for multi-university WebVPN support.

Loads from litbase_ai/data/webvpn.json which contains 100+ Chinese
university WebVPN configurations. Each school entry has:
- host: WebVPN base URL
- crypto_key / crypto_iv: AES-CFB encryption keys (optional, default: wrdvpnisthebest!)
- type: "webvpn" (default), "easyconnect", "atrust", or "ezproxy"
- gateway: additional gateway domain for EasyConnect/aTrust
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_FILE = _DATA_DIR / "webvpn.json"

_DEFAULT_KEY = b"wrdvpnisthebest!"

_cache: dict[str, "SchoolEntry"] | None = None


@dataclass
class SchoolEntry:
    name: str
    province: str
    host: str
    key: bytes = _DEFAULT_KEY
    iv: bytes = _DEFAULT_KEY
    school_type: str = "webvpn"
    gateway: str = ""

    @property
    def base_url(self) -> str:
        """Full HTTPS base URL."""
        if self.host.startswith("http"):
            return self.host.rstrip("/")
        return f"https://{self.host}"


def _load_all() -> dict[str, "SchoolEntry"]:
    global _cache
    if _cache is not None:
        return _cache

    _cache = {}
    if not _DATA_FILE.exists():
        return _cache

    try:
        db = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return _cache

    for province, schools in db.items():
        if not isinstance(schools, dict):
            continue
        for name, info in schools.items():
            if not isinstance(info, dict):
                continue
            host = info.get("host", "").strip()
            if not host:
                continue

            key = _DEFAULT_KEY
            iv = _DEFAULT_KEY
            if "crypto_key" in info and info["crypto_key"]:
                key = info["crypto_key"].encode("utf-8")
            if "crypto_iv" in info and info["crypto_iv"]:
                iv = info["crypto_iv"].encode("utf-8")

            entry = SchoolEntry(
                name=name,
                province=province,
                host=host,
                key=key,
                iv=iv,
                school_type=info.get("type", "webvpn"),
                gateway=info.get("gateway", ""),
            )
            _cache[name] = entry

    return _cache


def search(name: str) -> SchoolEntry | None:
    """Find a school by exact name or fuzzy match."""
    db = _load_all()

    # Exact match
    if name in db:
        return db[name]

    # Case-insensitive exact match
    for n, entry in db.items():
        if n.lower() == name.lower():
            return entry

    # Contains match
    name_lower = name.lower()
    matches = [
        (n, e) for n, e in db.items()
        if name_lower in n.lower() or name_lower in e.province.lower()
    ]
    if matches:
        return matches[0][1]

    return None


def search_multi(query: str, limit: int = 10) -> list[SchoolEntry]:
    """Search schools by keyword, returning multiple matches."""
    db = _load_all()
    q = query.lower()
    results = []
    for name, entry in db.items():
        if q in name.lower() or q in entry.province.lower() or q in entry.host.lower():
            results.append(entry)
            if len(results) >= limit:
                break
    return results


def list_all() -> list[SchoolEntry]:
    """Return all school entries."""
    return sorted(_load_all().values(), key=lambda e: f"{e.province}/{e.name}")


def get_host_for(name: str) -> str | None:
    """Get WebVPN base URL for a school name, with auto-detection.

    If the name looks like a URL (contains '://'), return it directly.
    Otherwise search the database.
    """
    if "://" in name:
        return name.rstrip("/")

    entry = search(name)
    if entry:
        return entry.base_url
    return None


def get_keys_for(name: str) -> tuple[bytes, bytes]:
    """Get AES-CFB key and IV for a school."""
    entry = search(name)
    if entry:
        return entry.key, entry.iv
    return _DEFAULT_KEY, _DEFAULT_KEY
