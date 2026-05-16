from __future__ import annotations

import math
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None

from litbase_ai.enrich.journal_rank import JournalRankLookup
from litbase_ai.models import ExpandedQuery, PaperMetadata, PaperScore
from litbase_ai.scoring.base import BaseScorer
from litbase_ai.utils.text import tokenize_topic


class RuleBasedScorer(BaseScorer):
    """Rule-based scorer with multilingual relevance features."""

    def __init__(self, scoring_config: dict[str, Any], journal_rank_lookup: JournalRankLookup | None = None):
        self.scoring_config = scoring_config
        self.journal_rank_lookup = journal_rank_lookup
        self.current_year = datetime.now().year

    def score(
        self,
        paper: PaperMetadata,
        topic: str,
        expanded_query: ExpandedQuery | None = None,
    ) -> PaperScore:
        relevance, detail = self._score_relevance(paper, topic, expanded_query)
        year_score = self._score_year(paper)
        citation_score = self._score_citations(paper)
        journal_score = self._score_journal(paper)
        access_score = self._score_access(paper)
        type_score = self._score_type(paper)

        scores = {
            "relevance": relevance,
            "year": year_score,
            "citation": citation_score,
            "journal": journal_score,
            "access": access_score,
            "type": type_score,
        }
        rule_score = self._weighted_total(scores)
        labels = self._build_labels(paper, relevance, detail["query_coverage_score"])

        return PaperScore(
            paper_id=paper.id,
            relevance_score=relevance,
            year_score=year_score,
            citation_score=citation_score,
            journal_score=journal_score,
            access_score=access_score,
            type_score=type_score,
            rule_score=rule_score,
            llm_score=None,
            final_score=rule_score,
            llm_reason=None,
            labels=labels,
            title_match_score=detail["title_match_score"],
            abstract_match_score=detail["abstract_match_score"],
            keyword_match_score=detail["keyword_match_score"],
            phrase_match_score=detail["phrase_match_score"],
            fuzzy_match_score=detail["fuzzy_match_score"],
        )

    def _score_relevance(
        self,
        paper: PaperMetadata,
        topic: str,
        expanded_query: ExpandedQuery | None = None,
    ) -> tuple[float, dict[str, float]]:
        terms = self._collect_terms(topic, expanded_query)
        phrases = self._collect_phrases(topic, expanded_query)
        loose_queries = self._collect_loose_queries(topic, expanded_query)

        title_text = (paper.title or "").lower()
        abstract_text = (paper.abstract or "").lower()
        keyword_text = " ".join(self._collect_paper_keywords(paper)).lower()
        combined_text = f"{title_text} {abstract_text} {keyword_text}"

        title_match_score = self._field_match_score(title_text, terms)
        abstract_match_score = self._field_match_score(abstract_text, terms)
        keyword_match_score = self._field_match_score(keyword_text, terms)
        phrase_match_score = self._phrase_match_score(combined_text, phrases)
        fuzzy_match_score = self._fuzzy_match_score(title_text, abstract_text, loose_queries, topic)
        query_coverage_score = self._query_coverage_score(combined_text, terms)
        keyword_match_score = min(100.0, 0.7 * keyword_match_score + 0.3 * query_coverage_score)

        relevance_score = (
            0.30 * title_match_score
            + 0.25 * abstract_match_score
            + 0.20 * keyword_match_score
            + 0.15 * phrase_match_score
            + 0.10 * fuzzy_match_score
        )

        detail = {
            "title_match_score": round(title_match_score, 2),
            "abstract_match_score": round(abstract_match_score, 2),
            "keyword_match_score": round(keyword_match_score, 2),
            "phrase_match_score": round(phrase_match_score, 2),
            "fuzzy_match_score": round(fuzzy_match_score, 2),
            "query_coverage_score": round(query_coverage_score, 2),
        }
        return round(min(100.0, relevance_score), 2), detail

    def _field_match_score(self, field_text: str, terms: list[str]) -> float:
        if not field_text or not terms:
            return 0.0
        weights = []
        hits = []
        for term in terms:
            term = term.strip()
            if not term:
                continue
            weight = 1.2 if re.search(r"[\u4e00-\u9fff]", term) else 1.0
            weights.append(weight)
            hit = 1.0 if self._term_in_text(term, field_text) else 0.0
            hits.append(hit * weight)
        if not weights:
            return 0.0
        return 100.0 * (sum(hits) / sum(weights))

    def _phrase_match_score(self, text: str, phrases: list[str]) -> float:
        if not text or not phrases:
            return 0.0
        hits = sum(1 for phrase in phrases if phrase and self._term_in_text(phrase, text))
        return min(100.0, 100.0 * hits / max(1, len(phrases)))

    def _fuzzy_match_score(
        self,
        title_text: str,
        abstract_text: str,
        loose_queries: list[str],
        topic: str,
    ) -> float:
        abstract_short = abstract_text[:1200]
        if not loose_queries:
            loose_queries = [topic]
        candidates = loose_queries[:10]

        if fuzz is not None:
            title_topic = float(fuzz.token_set_ratio(title_text, topic.lower()))
            abstract_scores = [float(fuzz.token_set_ratio(abstract_short, q.lower())) for q in candidates]
            abstract_best = max(abstract_scores) if abstract_scores else 0.0
        else:
            title_topic = SequenceMatcher(None, title_text, topic.lower()).ratio() * 100
            abstract_scores = [SequenceMatcher(None, abstract_short, q.lower()).ratio() * 100 for q in candidates]
            abstract_best = max(abstract_scores) if abstract_scores else 0.0
        return min(100.0, 0.6 * title_topic + 0.4 * abstract_best)

    def _query_coverage_score(self, text: str, terms: list[str]) -> float:
        if not terms:
            return 0.0
        hits = sum(1 for term in terms if self._term_in_text(term, text))
        return 100.0 * hits / max(1, len(terms))

    def _collect_terms(self, topic: str, expanded_query: ExpandedQuery | None) -> list[str]:
        terms = []
        if expanded_query:
            terms.extend(expanded_query.english_keywords)
            terms.extend(expanded_query.chinese_keywords)
            terms.extend(expanded_query.synonyms)
            terms.extend(expanded_query.related_terms)
            if expanded_query.english_topic:
                terms.append(expanded_query.english_topic)
            if expanded_query.chinese_topic:
                terms.append(expanded_query.chinese_topic)
        else:
            terms.extend(tokenize_topic(topic))
            terms.append(topic)
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            norm = " ".join(str(term).split())
            key = norm.lower()
            if norm and key not in seen:
                seen.add(key)
                deduped.append(norm)
        return deduped[:60]

    def _collect_phrases(self, topic: str, expanded_query: ExpandedQuery | None) -> list[str]:
        phrases = [topic]
        if expanded_query:
            phrases.extend(expanded_query.phrase_queries)
            if expanded_query.english_topic:
                phrases.append(expanded_query.english_topic)
            if expanded_query.chinese_topic:
                phrases.append(expanded_query.chinese_topic)
        return self._dedupe(phrases)[:20]

    def _collect_loose_queries(self, topic: str, expanded_query: ExpandedQuery | None) -> list[str]:
        queries = [topic]
        if expanded_query:
            queries.extend(expanded_query.loose_queries)
            queries.extend(expanded_query.boolean_queries)
        return self._dedupe(queries)[:20]

    def _collect_paper_keywords(self, paper: PaperMetadata) -> list[str]:
        keywords = list(paper.keywords)
        for key in ("concepts", "topics", "subjects"):
            values = paper.raw.get(key) or []
            if isinstance(values, list):
                keywords.extend([str(v) for v in values if str(v).strip()])
        primary_topic = paper.raw.get("primary_topic")
        if primary_topic:
            keywords.append(str(primary_topic))
        return self._dedupe(keywords)

    def _term_in_text(self, term: str, text: str) -> bool:
        term_lower = term.lower()
        if re.search(r"[\u4e00-\u9fff]", term):
            return term in text
        return term_lower in text

    def _build_labels(self, paper: PaperMetadata, relevance_score: float, query_coverage_score: float) -> list[str]:
        labels: list[str] = []
        if relevance_score >= 75:
            labels.append("high_relevance")
        if query_coverage_score >= 60:
            labels.append("good_query_coverage")
        if paper.year and paper.year >= self.current_year - 3:
            labels.append("recent")
        if paper.citation_count and paper.citation_count >= 100:
            labels.append("highly_cited")
        if paper.open_access_status:
            labels.append("open_access")
        return labels

    def _score_year(self, paper: PaperMetadata) -> float:
        year = paper.year
        year_score_cfg = self.scoring_config.get("year_score", {})
        if not year:
            base_score = float(year_score_cfg.get("older", 35))
        else:
            age = self.current_year - year
            if age <= 3:
                base_score = float(year_score_cfg.get("recent_3_years", 100))
            elif age <= 5:
                base_score = float(year_score_cfg.get("recent_5_years", 85))
            elif age <= 10:
                base_score = float(year_score_cfg.get("recent_10_years", 65))
            else:
                base_score = float(year_score_cfg.get("older", 35))

        citation = paper.citation_count or 0
        bonus = 0
        if citation >= 500:
            bonus = 10
        elif citation >= 100:
            bonus = 5
        return min(100.0, base_score + bonus)

    def _score_citations(self, paper: PaperMetadata) -> float:
        citations = float(paper.citation_count or 0)
        if citations <= 0:
            return 0.0
        if paper.year:
            years = max(1, self.current_year - paper.year + 1)
            citation_per_year = citations / years
        else:
            citation_per_year = citations
        score = 100 * math.log1p(citation_per_year) / math.log1p(100)
        return round(min(100.0, score), 2)

    def _score_journal(self, paper: PaperMetadata) -> float:
        config = self.scoring_config.get("journal_score", {})
        quartile = "unknown"
        if self.journal_rank_lookup:
            quartile = self.journal_rank_lookup.get_quartile(paper.journal)
        quartile_key = quartile if quartile in config else quartile.upper()
        return float(config.get(quartile_key, config.get("unknown", 50)))

    def _score_access(self, paper: PaperMetadata) -> float:
        config = self.scoring_config.get("access_score", {})
        if paper.pdf_url:
            return float(config.get("has_pdf", 100))
        if paper.landing_page_url and paper.open_access_status:
            return float(config.get("has_oa_landing_page", 70))
        if paper.doi:
            return float(config.get("has_doi_only", 40))
        return float(config.get("no_access", 0))

    def _score_type(self, paper: PaperMetadata) -> float:
        config = self.scoring_config.get("type_score", {})
        paper_type = (paper.paper_type or "").lower()
        if "journal" in paper_type:
            return float(config.get("journal_article", 100))
        if "proceeding" in paper_type or "conference" in paper_type:
            return float(config.get("proceedings_article", 80))
        if "preprint" in paper_type:
            return float(config.get("preprint", 75))
        if "book" in paper_type or "chapter" in paper_type:
            return float(config.get("book_chapter", 60))
        return float(config.get("unknown", 50))

    def _weighted_total(self, scores: dict[str, float]) -> float:
        weights = self.scoring_config.get("weights", {})
        total = 0.0
        for metric, value in scores.items():
            total += float(weights.get(metric, 0)) * value
        return round(min(100.0, total), 2)

    def _dedupe(self, values: list[str]) -> list[str]:
        seen = set()
        result = []
        for value in values:
            norm = " ".join(str(value).split())
            key = norm.lower()
            if norm and key not in seen:
                seen.add(key)
                result.append(norm)
        return result

