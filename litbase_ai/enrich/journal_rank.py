from __future__ import annotations

import csv
import re
from pathlib import Path

from litbase_ai.utils.logging import get_logger


logger = get_logger(__name__)


class JournalRankLookup:
    """Lookup helper for journal quartiles."""

    def __init__(self, csv_path: Path | None = None):
        package_dir = Path(__file__).resolve().parent.parent
        self.csv_path = csv_path or (package_dir / "data" / "journal_quartile_sample.csv")
        self._mapping: dict[str, str] = {}
        self._loaded = False

    def load(self) -> None:
        """Load journal quartile table from CSV."""
        self._mapping = {}
        if not self.csv_path.exists():
            logger.warning("Journal rank CSV not found: %s", self.csv_path)
            self._loaded = True
            return
        try:
            with self.csv_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    name = row.get("journal_name")
                    quartile = (row.get("quartile") or "unknown").strip()
                    if not name:
                        continue
                    normalized = self._normalize_journal_name(name)
                    self._mapping[normalized] = quartile
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load journal rank CSV: %s", exc)
        self._loaded = True
        logger.info("JournalRankLookup loaded %s journal entries.", len(self._mapping))

    def get_quartile(self, journal_name: str | None) -> str:
        """Return journal quartile by normalized name."""
        if not self._loaded:
            self.load()
        if not journal_name:
            return "unknown"
        normalized = self._normalize_journal_name(journal_name)
        return self._mapping.get(normalized, "unknown")

    def _normalize_journal_name(self, name: str) -> str:
        text = name.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

