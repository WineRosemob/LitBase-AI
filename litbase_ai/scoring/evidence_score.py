from __future__ import annotations

import hashlib
import re
from typing import Any

from litbase_ai.models import EvidenceItem, ExpandedQuery, PaperMetadata, ScoredPaper
from litbase_ai.utils.cache import CacheManager
from litbase_ai.utils.text import tokenize_topic


class EvidenceBasedScorer:
    """Extract metadata-grounded evidence snippets for explainable relevance."""

    def __init__(self, cache_manager: CacheManager | None = None, progress=None):
        self.cache_manager = cache_manager
        self.progress = progress
        self.last_stats: dict[str, int] = {
            "evidence_extracted": 0,
            "evidence_missing_count": 0,
            "cache_hit": 0,
            "cache_miss": 0,
        }

    def extract_evidence_from_metadata(
        self,
        paper: PaperMetadata,
        topic: str,
        expanded_query: ExpandedQuery | None = None,
    ) -> list[EvidenceItem]:
        cache_key = self._build_cache_key(paper, topic, expanded_query)
        if self.cache_manager:
            cached = self.cache_manager.get("evidence", cache_key)
            if cached and isinstance(cached.get("items"), list):
                self.last_stats["cache_hit"] += 1
                items = []
                for item in cached.get("items", []):
                    try:
                        items.append(EvidenceItem(**item))
                    except Exception:
                        continue
                if items:
                    return items
        self.last_stats["cache_miss"] += 1

        terms = self._collect_terms(topic, expanded_query)
        items: list[EvidenceItem] = []

        title = paper.title or ""
        if title and self._match_any(title, terms):
            items.append(EvidenceItem(source="title", text=title[:240], relevance="matched_topic", confidence=0.9))

        for sentence in self._split_sentences(paper.abstract or ""):
            if self._match_any(sentence, terms):
                items.append(EvidenceItem(source="abstract", text=sentence[:320], relevance="matched_topic", confidence=0.75))
            if len(items) >= 3:
                break

        if len(items) < 3 and paper.keywords:
            matched_keywords = [kw for kw in paper.keywords if self._match_any(kw, terms)]
            if matched_keywords:
                text = "; ".join(matched_keywords[:6])
                items.append(EvidenceItem(source="keywords", text=text[:320], relevance="matched_topic", confidence=0.7))

        if not items:
            items = [EvidenceItem(source="metadata", text="No strong evidence extracted from metadata.", relevance="weak", confidence=0.3)]

        if self.cache_manager:
            self.cache_manager.set("evidence", cache_key, {"items": [x.model_dump() for x in items]})
        return items

    def score_batch(
        self,
        papers: list[ScoredPaper],
        topic: str,
        expanded_query: ExpandedQuery | None = None,
    ) -> list[ScoredPaper]:
        self.last_stats = {
            "evidence_extracted": 0,
            "evidence_missing_count": 0,
            "cache_hit": 0,
            "cache_miss": 0,
        }
        task_id = self.progress.task("Evidence extraction", total=len(papers)) if self.progress else None
        for idx, paper in enumerate(papers):
            items = self.extract_evidence_from_metadata(paper.metadata, topic, expanded_query=expanded_query)
            paper.score.evidence_items = items
            self.last_stats["evidence_extracted"] += len(items)
            if len(items) == 1 and items[0].source == "metadata":
                self.last_stats["evidence_missing_count"] += 1
            if task_id is not None:
                self.progress.update(task_id, advance=1, description=f"Evidence {idx + 1}/{len(papers)}")
        return papers

    def _collect_terms(self, topic: str, expanded_query: ExpandedQuery | None) -> list[str]:
        terms: list[str] = [topic]
        terms.extend(tokenize_topic(topic))
        if expanded_query:
            terms.extend(expanded_query.english_keywords)
            terms.extend(expanded_query.chinese_keywords)
            terms.extend(expanded_query.synonyms)
            terms.extend(expanded_query.related_terms)
            if expanded_query.english_topic:
                terms.append(expanded_query.english_topic)
            if expanded_query.chinese_topic:
                terms.append(expanded_query.chinese_topic)
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            norm = " ".join(str(term).split())
            key = norm.lower()
            if norm and key not in seen:
                seen.add(key)
                deduped.append(norm)
        return deduped[:80]

    def _match_any(self, text: str, terms: list[str]) -> bool:
        text_lower = text.lower()
        for term in terms:
            if not term:
                continue
            if re.search(r"[\u4e00-\u9fff]", term):
                if term in text:
                    return True
            else:
                if term.lower() in text_lower:
                    return True
        return False

    def _split_sentences(self, text: str) -> list[str]:
        if not text:
            return []
        candidates = re.split(r"[。！？!?;\n]|(?<=\.)\s+", text)
        cleaned = [" ".join(x.split()).strip() for x in candidates if x and x.strip()]
        return cleaned

    def _build_cache_key(self, paper: PaperMetadata, topic: str, expanded_query: ExpandedQuery | None) -> str:
        abstract_hash = hashlib.md5((paper.abstract or "").encode("utf-8")).hexdigest()  # noqa: S324
        eq = expanded_query.model_dump() if expanded_query else {}
        payload: dict[str, Any] = {
            "topic": topic,
            "paper_id": paper.id,
            "title": paper.title,
            "abstract_hash": abstract_hash,
            "expanded_query": eq,
            "version": "evidence_v1",
        }
        return str(payload)
