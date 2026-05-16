from __future__ import annotations

import csv
from difflib import SequenceMatcher
from pathlib import Path

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None

from litbase_ai.models import ScoredPaper
from litbase_ai.utils.logging import get_logger
from litbase_ai.utils.text import normalize_doi, normalize_title


logger = get_logger(__name__)


class HumanFeedbackManager:
    """Apply optional human labels/scores from feedback.csv."""

    LABEL_SCORE = {
        "core": 95,
        "relevant": 80,
        "background": 70,
        "method": 76,
        "data": 74,
        "irrelevant": 20,
        "unsure": 50,
    }

    def __init__(self, feedback_file: Path | None = None):
        self.feedback_file = feedback_file
        self.feedback_rows: list[dict[str, str]] = []
        self.last_stats: dict[str, int | bool] = {
            "human_feedback_enabled": bool(feedback_file),
            "feedback_applied_count": 0,
        }

    def load_feedback(self) -> dict[str, list[dict[str, str]]]:
        if not self.feedback_file or not self.feedback_file.exists():
            self.feedback_rows = []
            return {"rows": []}
        rows: list[dict[str, str]] = []
        with self.feedback_file.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
        self.feedback_rows = rows
        return {"rows": rows}

    def apply_feedback(self, papers: list[ScoredPaper]) -> list[ScoredPaper]:
        self.last_stats = {
            "human_feedback_enabled": bool(self.feedback_file and self.feedback_file.exists()),
            "feedback_applied_count": 0,
        }
        if not self.feedback_file or not self.feedback_file.exists():
            return papers
        if not self.feedback_rows:
            self.load_feedback()
        if not self.feedback_rows:
            return papers

        doi_index: dict[str, dict[str, str]] = {}
        title_rows: list[dict[str, str]] = []
        for row in self.feedback_rows:
            doi = normalize_doi(row.get("doi"))
            if doi:
                doi_index[doi] = row
            title_rows.append(row)

        for paper in papers:
            matched_row = None
            paper_doi = normalize_doi(paper.metadata.doi)
            if paper_doi and paper_doi in doi_index:
                matched_row = doi_index[paper_doi]
            elif title_rows:
                matched_row = self._match_by_title(paper.metadata.title or "", title_rows)

            if not matched_row:
                continue
            label = (matched_row.get("human_label") or "").strip().lower() or None
            feedback_score = self._parse_feedback_score(matched_row.get("feedback_score"), label)
            if label:
                paper.score.human_label = label
            if feedback_score is not None:
                paper.score.human_feedback_score = feedback_score
            self.last_stats["feedback_applied_count"] = int(self.last_stats.get("feedback_applied_count", 0)) + 1
        return papers

    def _parse_feedback_score(self, raw: str | None, label: str | None) -> float | None:
        if raw:
            try:
                score = float(raw)
                return max(0.0, min(100.0, score))
            except Exception:
                pass
        if label and label in self.LABEL_SCORE:
            return float(self.LABEL_SCORE[label])
        return None

    def _match_by_title(self, title: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
        target = normalize_title(title)
        if not target:
            return None
        best_row = None
        best_score = 0.0
        for row in rows:
            candidate = normalize_title(row.get("title"))
            if not candidate:
                continue
            if fuzz is not None:
                score = float(fuzz.ratio(target, candidate))
            else:
                score = SequenceMatcher(None, target, candidate).ratio() * 100
            if score > best_score:
                best_score = score
                best_row = row
        if best_score >= 92:
            return best_row
        return None
