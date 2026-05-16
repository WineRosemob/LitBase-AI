from __future__ import annotations

from typing import Iterable

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None

from litbase_ai.models import PaperMetadata
from litbase_ai.utils.logging import get_logger
from litbase_ai.utils.text import normalize_doi, normalize_title


logger = get_logger(__name__)


class PaperMerger:
    """Merge and deduplicate papers from multiple data sources."""

    def __init__(self, title_similarity_threshold: int = 92):
        self.title_similarity_threshold = title_similarity_threshold

    def merge(self, paper_lists: list[list[PaperMetadata]]) -> list[PaperMetadata]:
        papers = self._flatten(paper_lists)
        logger.info("Merging papers from multi-sources. input=%s", len(papers))
        papers = self._deduplicate_by_doi(papers)
        papers = self._deduplicate_by_title(papers)
        logger.info("Merged papers total=%s", len(papers))
        return papers

    def _flatten(self, paper_lists: Iterable[list[PaperMetadata]]) -> list[PaperMetadata]:
        result: list[PaperMetadata] = []
        for paper_list in paper_lists:
            result.extend(paper_list)
        return result

    def _deduplicate_by_doi(self, papers: list[PaperMetadata]) -> list[PaperMetadata]:
        doi_map: dict[str, PaperMetadata] = {}
        no_doi: list[PaperMetadata] = []
        for paper in papers:
            doi = normalize_doi(paper.doi)
            if not doi:
                no_doi.append(paper)
                continue
            if doi not in doi_map:
                paper.doi = doi
                doi_map[doi] = paper
            else:
                doi_map[doi] = self._merge_two_papers(doi_map[doi], paper)
        return list(doi_map.values()) + no_doi

    def _deduplicate_by_title(self, papers: list[PaperMetadata]) -> list[PaperMetadata]:
        unique_papers: list[PaperMetadata] = []
        for paper in papers:
            title_norm = normalize_title(paper.title)
            if not title_norm:
                unique_papers.append(paper)
                continue
            merged = False
            for idx, existing in enumerate(unique_papers):
                score = self._title_similarity(title_norm, normalize_title(existing.title))
                if score >= self.title_similarity_threshold:
                    unique_papers[idx] = self._merge_two_papers(existing, paper)
                    merged = True
                    break
            if not merged:
                unique_papers.append(paper)
        return unique_papers

    def _title_similarity(self, left: str, right: str) -> float:
        if fuzz is not None:
            return float(fuzz.token_set_ratio(left, right))
        # Fallback when rapidfuzz is unavailable.
        from difflib import SequenceMatcher

        return SequenceMatcher(None, left, right).ratio() * 100

    def _merge_two_papers(self, base: PaperMetadata, incoming: PaperMetadata) -> PaperMetadata:
        merged_sources = set(base.raw.get("merged_sources", []))
        merged_sources.update(incoming.raw.get("merged_sources", []))
        merged_sources.add(base.source_database)
        merged_sources.add(incoming.source_database)

        merged = base.model_copy(deep=True)
        merged.title = self._pick_better_text(base.title, incoming.title)
        merged.abstract = self._pick_better_text(base.abstract, incoming.abstract)
        merged.keywords = self._merge_list(base.keywords, incoming.keywords)
        merged.authors = self._merge_list(base.authors, incoming.authors)
        merged.year = self._pick_not_none(base.year, incoming.year)
        merged.doi = self._pick_not_none(base.doi, incoming.doi)
        merged.journal = self._pick_not_none(base.journal, incoming.journal)
        merged.publisher = self._pick_not_none(base.publisher, incoming.publisher)
        merged.citation_count = self._pick_higher_int(base.citation_count, incoming.citation_count)
        merged.open_access_status = self._pick_not_none(base.open_access_status, incoming.open_access_status)
        merged.pdf_url = self._pick_not_none(base.pdf_url, incoming.pdf_url)
        merged.landing_page_url = self._pick_not_none(base.landing_page_url, incoming.landing_page_url)
        merged.paper_type = self._pick_not_none(base.paper_type, incoming.paper_type)
        merged.source_database = "+".join(sorted(merged_sources))
        merged_matched_queries = self._merge_list(
            [str(x) for x in (base.raw.get("matched_queries") or [])],
            [str(x) for x in (incoming.raw.get("matched_queries") or [])],
        )
        merged_concepts = self._merge_list(
            [str(x) for x in (base.raw.get("concepts") or [])],
            [str(x) for x in (incoming.raw.get("concepts") or [])],
        )
        merged_topics = self._merge_list(
            [str(x) for x in (base.raw.get("topics") or [])],
            [str(x) for x in (incoming.raw.get("topics") or [])],
        )
        merged_subjects = self._merge_list(
            [str(x) for x in (base.raw.get("subjects") or [])],
            [str(x) for x in (incoming.raw.get("subjects") or [])],
        )
        primary_topic = base.raw.get("primary_topic") or incoming.raw.get("primary_topic")
        merged.raw = {
            **base.raw,
            **incoming.raw,
            "merged_sources": sorted(merged_sources),
            "matched_queries": merged_matched_queries,
            "concepts": merged_concepts,
            "topics": merged_topics,
            "subjects": merged_subjects,
            "primary_topic": primary_topic,
            "source_count": len(merged_sources),
        }
        if merged_matched_queries:
            merged.raw["matched_queries"] = merged_matched_queries
        return merged

    def _pick_not_none(self, left: str | int | None, right: str | int | None):
        return left if left not in (None, "") else right

    def _pick_better_text(self, left: str | None, right: str | None) -> str | None:
        left_len = len(left or "")
        right_len = len(right or "")
        return left if left_len >= right_len else right

    def _pick_higher_int(self, left: int | None, right: int | None) -> int | None:
        if left is None:
            return right
        if right is None:
            return left
        return max(left, right)

    def _merge_list(self, left: list[str], right: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for item in left + right:
            if not item:
                continue
            key = item.strip().lower()
            if key not in seen:
                seen.add(key)
                merged.append(item)
        return merged
