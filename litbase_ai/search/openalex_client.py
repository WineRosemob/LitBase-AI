from __future__ import annotations

import math
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from litbase_ai.models import ExpandedQuery, PaperMetadata
from litbase_ai.query.expander import QueryExpander
from litbase_ai.search.base import BaseSearchClient
from litbase_ai.utils.logging import get_logger
from litbase_ai.utils.text import normalize_doi


logger = get_logger(__name__)


class OpenAlexClient(BaseSearchClient):
    """OpenAlex Works API client with multi-query retrieval strategies."""

    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, mailto: str | None = None, per_page: int = 100):
        self.mailto = mailto
        self.per_page = min(max(20, per_page), 200)
        self.user_agent = "LitBase-AI/0.2 (+legal-oa-only)"
        self.timeout = 15.0
        self.last_status = "ok"
        self.last_reason = ""
        self.last_search_stats: dict[str, float | int] = {
            "queries": 0,
            "returned": 0,
            "failed_queries": 0,
            "elapsed_seconds": 0.0,
        }

    def search_works(self, topic: str, limit: int = 500, year_from: int | None = None) -> list[PaperMetadata]:
        expanded = QueryExpander(llm_scorer=None).expand(topic)
        return self.search_with_expanded_query(expanded_query=expanded, limit=limit, year_from=year_from)

    def search_with_expanded_query(
        self,
        expanded_query: ExpandedQuery,
        limit: int = 500,
        year_from: int | None = None,
        progress=None,
    ) -> list[PaperMetadata]:
        start = time.perf_counter()
        self.last_status = "ok"
        self.last_reason = ""
        self.last_search_stats = {
            "queries": 0,
            "returned": 0,
            "failed_queries": 0,
            "elapsed_seconds": 0.0,
        }
        query_pool = self._build_query_pool(expanded_query)
        if not query_pool:
            query_pool = [expanded_query.original_topic]
        max_queries = 8 if limit <= 80 else (14 if limit <= 250 else 24)
        query_pool = query_pool[:max_queries]

        logger.info(
            "OpenAlex expanded search started | queries=%s | limit=%s",
            len(query_pool),
            limit,
        )
        self.last_search_stats["queries"] = len(query_pool)
        papers: list[PaperMetadata] = []
        per_query_base = max(8, min(80, math.ceil(limit / max(1, len(query_pool))) + 8))
        mode_sequence = ["default", "title_abstract", "title"]
        task_id = progress.task("Searching OpenAlex queries", total=len(query_pool)) if progress else None
        consecutive_failures = 0

        for idx, query in enumerate(query_pool):
            if len(papers) >= limit:
                break
            mode = mode_sequence[idx % len(mode_sequence)]
            query_limit = max(6, per_query_base)
            failed_before = int(self.last_search_stats.get("failed_queries", 0))
            result = self._search_single_query(
                query=query,
                limit=query_limit,
                year_from=year_from,
                search_mode=mode,
            )
            papers.extend(result)
            failed_after = int(self.last_search_stats.get("failed_queries", 0))
            if failed_after > failed_before:
                consecutive_failures += 1
            else:
                consecutive_failures = 0 if result else consecutive_failures
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    advance=1,
                    description=f"OpenAlex {idx + 1}/{len(query_pool)}: {query[:60]} ({len(result)})",
                )
            if consecutive_failures >= 3:
                logger.warning("OpenAlex had %s consecutive failed queries. Stop early.", consecutive_failures)
                break
            if idx % 4 == 0 and len(papers) < limit:
                supplemental = self._search_single_query(
                    query=query,
                    limit=max(4, query_limit // 2),
                    year_from=year_from,
                    search_mode="default",
                )
                papers.extend(supplemental)
            if len(papers) >= limit:
                break
            if idx % 3 == 0:
                time.sleep(0.15)

        if consecutive_failures < 3:
            try:
                concept_papers = self._search_by_concepts_or_topics(
                    expanded_query=expanded_query,
                    limit=max(6, limit // 5),
                    year_from=year_from,
                )
                papers.extend(concept_papers)
            except Exception as exc:  # pragma: no cover
                logger.warning("OpenAlex concept/topic search failed: %s", exc)
                self.last_search_stats["failed_queries"] = int(self.last_search_stats.get("failed_queries", 0)) + 1

        logger.info("OpenAlex expanded search finished with %s candidates.", len(papers))
        final = papers[:limit]
        self.last_search_stats["returned"] = len(final)
        self.last_search_stats["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        if len(final) == 0 and int(self.last_search_stats.get("failed_queries", 0)) > 0:
            self.last_status = "failed"
            self.last_reason = "all queries failed"
        return final

    def _search_single_query(
        self,
        query: str,
        limit: int,
        year_from: int | None,
        search_mode: str = "default",
    ) -> list[PaperMetadata]:
        try:
            if search_mode == "title":
                papers = self._search_by_title(query=query, limit=limit, year_from=year_from)
            elif search_mode == "title_abstract":
                papers = self._search_by_abstract_and_title(query=query, limit=limit, year_from=year_from)
            else:
                params = self._build_params(query=query, year_from=year_from, search_mode="default")
                papers = self._request_and_parse(params=params, limit=limit)

            for paper in papers:
                self._attach_match_trace(paper, query=query, search_mode=search_mode)
            return papers
        except Exception as exc:  # pragma: no cover
            logger.warning("OpenAlex query failed | mode=%s | query=%s | err=%s", search_mode, query, exc)
            self.last_search_stats["failed_queries"] = int(self.last_search_stats.get("failed_queries", 0)) + 1
            return []

    def _search_by_title(self, query: str, limit: int, year_from: int | None) -> list[PaperMetadata]:
        params = self._build_params(query=query, year_from=year_from, search_mode="title")
        return self._request_and_parse(params=params, limit=limit)

    def _search_by_abstract_and_title(self, query: str, limit: int, year_from: int | None) -> list[PaperMetadata]:
        params = self._build_params(query=query, year_from=year_from, search_mode="title_abstract")
        return self._request_and_parse(params=params, limit=limit)

    def _search_by_concepts_or_topics(
        self,
        expanded_query: ExpandedQuery,
        limit: int,
        year_from: int | None,
    ) -> list[PaperMetadata]:
        concept_queries = []
        concept_queries.extend(expanded_query.related_terms[:8])
        concept_queries.extend(expanded_query.synonyms[:6])
        concept_queries.extend(expanded_query.english_keywords[:6])
        concept_queries.extend(expanded_query.chinese_keywords[:6])
        max_terms = 4 if limit <= 8 else (6 if limit <= 20 else 8)
        concept_queries = self._dedupe(concept_queries)[:max_terms]

        results: list[PaperMetadata] = []
        if not concept_queries:
            return results
        each_limit = max(4, math.ceil(limit / len(concept_queries)))
        for concept_query in concept_queries:
            try:
                params = self._build_params(query=concept_query, year_from=year_from, search_mode="default")
                params["sort"] = "publication_year:desc"
                papers = self._request_and_parse(params=params, limit=each_limit)
                for paper in papers:
                    self._attach_match_trace(paper, query=concept_query, search_mode="concept")
                results.extend(papers)
            except Exception as exc:  # pragma: no cover
                logger.warning("OpenAlex concept query failed | query=%s | err=%s", concept_query, exc)
                self.last_search_stats["failed_queries"] = int(self.last_search_stats.get("failed_queries", 0)) + 1
        return results

    def _build_params(self, query: str, year_from: int | None, search_mode: str) -> dict[str, Any]:
        params: dict[str, Any] = {"sort": "relevance_score:desc"}
        filters = []
        clean_query = self._sanitize_filter_text(query)
        if search_mode == "title":
            filters.append(f"title.search:{clean_query}")
        elif search_mode == "title_abstract":
            filters.append(f"title_and_abstract.search:{clean_query}")
        else:
            params["search"] = query
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if filters:
            params["filter"] = ",".join(filters)
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def _request_and_parse(self, params: dict[str, Any], limit: int) -> list[PaperMetadata]:
        items = self._request_cursor_pages(params=params, limit=limit)
        return [self._parse_work(item) for item in items]

    def _request_cursor_pages(self, params: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor = "*"
        while len(results) < limit and cursor:
            request_params = dict(params)
            request_params["cursor"] = cursor
            request_params["per-page"] = min(self.per_page, max(1, limit - len(results)))
            payload = self._request_payload(request_params)
            items = payload.get("results", [])
            if not items:
                break
            results.extend(items)
            meta = payload.get("meta") or {}
            next_cursor = meta.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            time.sleep(0.12)
        return results[:limit]

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _request_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            return response.json()

    def _parse_work(self, item: dict[str, Any]) -> PaperMetadata:
        source = ((item.get("primary_location") or {}).get("source")) or {}
        open_access = item.get("open_access") or {}
        authorships = item.get("authorships") or []
        best_oa_location = item.get("best_oa_location") or {}
        primary_topic = item.get("primary_topic") or {}
        concepts = item.get("concepts") or []
        topics = item.get("topics") or []
        keywords_raw = item.get("keywords") or []

        authors = [
            (authorship.get("author") or {}).get("display_name")
            for authorship in authorships
            if (authorship.get("author") or {}).get("display_name")
        ]
        keywords = [
            keyword.get("display_name")
            for keyword in keywords_raw
            if isinstance(keyword, dict) and keyword.get("display_name")
        ]
        concept_terms = [
            concept.get("display_name")
            for concept in concepts[:12]
            if isinstance(concept, dict) and concept.get("display_name")
        ]
        topic_terms = [
            topic.get("display_name")
            for topic in topics[:8]
            if isinstance(topic, dict) and topic.get("display_name")
        ]
        primary_topic_name = primary_topic.get("display_name")
        if primary_topic_name:
            topic_terms.insert(0, primary_topic_name)
        merged_keywords = self._dedupe(keywords + concept_terms + topic_terms)

        doi = normalize_doi(item.get("doi"))
        return PaperMetadata(
            id=str(item.get("id") or doi or item.get("display_name") or ""),
            title=item.get("display_name") or item.get("title") or "Untitled",
            abstract=self._restore_abstract(item.get("abstract_inverted_index")),
            keywords=merged_keywords,
            authors=authors,
            year=item.get("publication_year"),
            doi=doi,
            journal=source.get("display_name"),
            publisher=source.get("host_organization_name"),
            citation_count=item.get("cited_by_count"),
            source_database="OpenAlex",
            open_access_status=open_access.get("oa_status"),
            pdf_url=self._extract_pdf_url(item),
            landing_page_url=self._extract_landing_page_url(item),
            paper_type=item.get("type"),
            raw={
                "openalex_id": item.get("id"),
                "concepts": concept_terms,
                "topics": topic_terms,
                "primary_topic": primary_topic_name,
                "subjects": [],
                "matched_queries": [],
                "source_priority": "high",
                "best_oa_location": best_oa_location,
                "open_access": open_access,
                "source": source,
                "origin": item,
            },
        )

    def _restore_abstract(self, inverted_index: dict[str, list[int]] | None) -> str | None:
        if not inverted_index:
            return None
        position_word_pairs: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word_pairs.append((pos, word))
        if not position_word_pairs:
            return None
        position_word_pairs.sort(key=lambda x: x[0])
        return " ".join(word for _, word in position_word_pairs)

    def _extract_pdf_url(self, item: dict[str, Any]) -> str | None:
        best_oa_location = item.get("best_oa_location") or {}
        if best_oa_location.get("url_for_pdf"):
            return best_oa_location.get("url_for_pdf")
        primary_location = item.get("primary_location") or {}
        if primary_location.get("pdf_url"):
            return primary_location.get("pdf_url")
        return None

    def _extract_landing_page_url(self, item: dict[str, Any]) -> str | None:
        primary_location = item.get("primary_location") or {}
        if primary_location.get("landing_page_url"):
            return primary_location.get("landing_page_url")
        best_oa_location = item.get("best_oa_location") or {}
        return best_oa_location.get("landing_page_url") or best_oa_location.get("url")

    def _build_query_pool(self, expanded_query: ExpandedQuery) -> list[str]:
        queries: list[str] = [expanded_query.original_topic]
        if expanded_query.english_topic:
            queries.append(expanded_query.english_topic)
        if expanded_query.chinese_topic:
            queries.append(expanded_query.chinese_topic)
        queries.extend(expanded_query.phrase_queries)
        queries.extend(expanded_query.loose_queries)
        queries.extend(expanded_query.boolean_queries)
        return self._dedupe([q for q in queries if q])[:24]

    def _sanitize_filter_text(self, query: str) -> str:
        clean = query.replace(",", " ").replace(":", " ")
        clean = " ".join(clean.split())
        return clean

    def _attach_match_trace(self, paper: PaperMetadata, query: str, search_mode: str) -> None:
        matched = paper.raw.get("matched_queries", [])
        if not isinstance(matched, list):
            matched = []
        trace = f"{search_mode}:{query}"
        if trace not in matched:
            matched.append(trace)
        paper.raw["matched_queries"] = matched

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            norm = " ".join(str(value).strip().split())
            key = norm.lower()
            if norm and key not in seen:
                seen.add(key)
                result.append(norm)
        return result
