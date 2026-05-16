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
from litbase_ai.utils.text import normalize_doi, strip_html


logger = get_logger(__name__)


class CrossrefClient(BaseSearchClient):
    """Crossref works client for broad recall without API key."""

    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, mailto: str | None = None, rows_per_page: int = 100):
        self.mailto = mailto
        self.rows_per_page = min(max(20, rows_per_page), 200)
        self.timeout = 15.0
        self.user_agent = "LitBase-AI/0.2 (+legal-oa-only)"
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
        queries = self._build_query_pool(expanded_query)
        if not queries:
            queries = [expanded_query.original_topic]
        self.last_search_stats["queries"] = len(queries)
        per_query_limit = max(8, min(60, math.ceil(limit / len(queries)) + 6))
        papers: list[PaperMetadata] = []
        task_id = progress.task("Searching Crossref queries", total=len(queries)) if progress else None
        consecutive_failures = 0
        for query in queries:
            if len(papers) >= limit:
                break
            failed_before = int(self.last_search_stats.get("failed_queries", 0))
            batch = self._search_single_query(query=query, limit=per_query_limit, year_from=year_from)
            papers.extend(batch)
            failed_after = int(self.last_search_stats.get("failed_queries", 0))
            if failed_after > failed_before:
                consecutive_failures += 1
            else:
                consecutive_failures = 0 if batch else consecutive_failures
            if progress and task_id is not None:
                progress.update(task_id, advance=1, description=f"Crossref: {query[:60]} ({len(batch)})")
            if consecutive_failures >= 3:
                logger.warning("Crossref had %s consecutive failed queries. Stop early.", consecutive_failures)
                break
            time.sleep(0.12)
        logger.info("Crossref search returned %s candidates.", len(papers))
        final = papers[:limit]
        self.last_search_stats["returned"] = len(final)
        self.last_search_stats["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        if len(final) == 0 and int(self.last_search_stats.get("failed_queries", 0)) > 0:
            self.last_status = "failed"
            self.last_reason = "all queries failed"
        return final

    def _search_single_query(self, query: str, limit: int, year_from: int | None) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        cursor = "*"
        while len(papers) < limit and cursor:
            params = {
                "query.bibliographic": query,
                "rows": min(self.rows_per_page, max(1, limit - len(papers))),
                "cursor": cursor,
                "sort": "relevance",
                "select": (
                    "DOI,title,abstract,published-print,published-online,issued,author,"
                    "container-title,publisher,URL,type,subject,is-referenced-by-count,link"
                ),
            }
            filters = []
            if year_from:
                filters.append(f"from-pub-date:{year_from}-01-01")
            if filters:
                params["filter"] = ",".join(filters)

            if self.mailto:
                params["mailto"] = self.mailto
            try:
                payload = self._request_payload(params)
            except Exception as exc:  # pragma: no cover
                logger.warning("Crossref query failed | query=%s | err=%s", query, exc)
                self.last_search_stats["failed_queries"] = int(self.last_search_stats.get("failed_queries", 0)) + 1
                break

            message = payload.get("message") or {}
            items = message.get("items") or []
            for item in items:
                paper = self._parse_work(item)
                matched = paper.raw.get("matched_queries", [])
                if isinstance(matched, list):
                    matched.append(f"crossref:{query}")
                paper.raw["matched_queries"] = matched
                papers.append(paper)
                if len(papers) >= limit:
                    break

            next_cursor = message.get("next-cursor")
            if len(items) < params["rows"]:
                break
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            time.sleep(0.1)
        return papers

    def _parse_work(self, item: dict) -> PaperMetadata:
        title_list = item.get("title") or []
        title = title_list[0] if title_list else "Untitled"
        abstract = strip_html(item.get("abstract"))
        doi = normalize_doi(item.get("DOI"))
        year = self._extract_year(item)
        authors = []
        for author in item.get("author", []) or []:
            given = (author.get("given") or "").strip()
            family = (author.get("family") or "").strip()
            full_name = " ".join(part for part in [given, family] if part).strip()
            if full_name:
                authors.append(full_name)
        journal_list = item.get("container-title") or []
        journal = journal_list[0] if journal_list else None
        subjects = [str(subject) for subject in (item.get("subject") or []) if str(subject).strip()]
        link_items = item.get("link") or []
        pdf_url = None
        for link in link_items:
            if not isinstance(link, dict):
                continue
            content_type = (link.get("content-type") or "").lower()
            if "pdf" in content_type and link.get("URL"):
                pdf_url = link.get("URL")
                break

        return PaperMetadata(
            id=str(item.get("DOI") or item.get("URL") or title),
            title=title,
            abstract=abstract or None,
            keywords=subjects,
            authors=authors,
            year=year,
            doi=doi,
            journal=journal,
            publisher=item.get("publisher"),
            citation_count=item.get("is-referenced-by-count"),
            source_database="Crossref",
            open_access_status="unknown",
            pdf_url=pdf_url,
            landing_page_url=item.get("URL"),
            paper_type=item.get("type"),
            raw={
                "subjects": subjects,
                "concepts": [],
                "topics": [],
                "primary_topic": None,
                "matched_queries": [],
                "source_priority": "medium",
                "crossref_origin": item,
            },
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _request_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            return response.json()

    def _extract_year(self, item: dict[str, Any]) -> int | None:
        for key in ("issued", "published-print", "published-online", "created"):
            date_parts = ((item.get(key) or {}).get("date-parts") or [])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
                if isinstance(year, int):
                    return year
        return None

    def _build_query_pool(self, expanded_query: ExpandedQuery) -> list[str]:
        queries = [expanded_query.original_topic]
        if expanded_query.english_topic:
            queries.append(expanded_query.english_topic)
        if expanded_query.chinese_topic:
            queries.append(expanded_query.chinese_topic)
        queries.extend(expanded_query.loose_queries)
        queries.extend(expanded_query.phrase_queries)
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            norm = " ".join(query.split())
            key = norm.lower()
            if norm and key not in seen:
                seen.add(key)
                deduped.append(norm)
        return deduped[:16]
