from __future__ import annotations

import math
import time
from typing import Any

import httpx

from litbase_ai.models import ExpandedQuery, PaperMetadata
from litbase_ai.query.expander import QueryExpander
from litbase_ai.search.base import BaseSearchClient
from litbase_ai.utils.logging import get_logger
from litbase_ai.utils.text import normalize_doi


logger = get_logger(__name__)


class SemanticScholarClient(BaseSearchClient):
    """Semantic Scholar Graph API client with rate-limit and retry control."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = ",".join(
        [
            "title",
            "abstract",
            "year",
            "authors",
            "venue",
            "publicationVenue",
            "publicationTypes",
            "citationCount",
            "influentialCitationCount",
            "externalIds",
            "url",
            "openAccessPdf",
            "fieldsOfStudy",
            "s2FieldsOfStudy",
            "publicationDate",
        ]
    )

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit_seconds: float = 1.2,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout = timeout
        self.user_agent = "LitBase-AI/0.3"
        self._last_request_ts = 0.0
        self._disabled_due_auth = False
        self.last_status = "skipped"
        self.last_reason = ""
        self.last_search_stats: dict[str, float | int] = {
            "queries": 0,
            "returned": 0,
            "failed_queries": 0,
            "elapsed_seconds": 0.0,
            "rate_limit_seconds": rate_limit_seconds,
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
            "rate_limit_seconds": self.rate_limit_seconds,
        }
        if not self.api_key:
            logger.info("SEMANTIC_SCHOLAR_API_KEY missing. Semantic Scholar source skipped.")
            self.last_status = "skipped"
            self.last_reason = "missing SEMANTIC_SCHOLAR_API_KEY"
            return []

        queries = self._build_query_pool(expanded_query)
        if not queries:
            queries = [expanded_query.original_topic]
        self.last_search_stats["queries"] = len(queries)
        per_query_limit = max(10, min(120, math.ceil(limit / max(1, len(queries))) + 10))
        papers: list[PaperMetadata] = []
        task_id = progress.task("Searching Semantic Scholar queries", total=len(queries)) if progress else None
        consecutive_failures = 0
        for idx, query in enumerate(queries):
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
                progress.update(
                    task_id,
                    advance=1,
                    description=f"Semantic Scholar {idx + 1}/{len(queries)}: {query[:60]} ({len(batch)})",
                )
            if consecutive_failures >= 3:
                logger.warning("Semantic Scholar had %s consecutive failed queries. Stop early.", consecutive_failures)
                break
            if self._disabled_due_auth:
                self.last_status = "failed"
                self.last_reason = "authorization failed (401/403)"
                break
        final = papers[:limit]
        self.last_search_stats["returned"] = len(final)
        self.last_search_stats["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        if len(final) == 0 and int(self.last_search_stats.get("failed_queries", 0)) > 0:
            self.last_status = "failed"
            if not self.last_reason:
                self.last_reason = "query failures or empty response"
        logger.info("Semantic Scholar returned %s candidates.", len(final))
        return final

    def _search_single_query(self, query: str, limit: int, year_from: int | None) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        offset = 0
        page_size = min(100, max(10, limit))
        while len(papers) < limit and offset < 1000:
            params: dict[str, Any] = {
                "query": query,
                "limit": min(page_size, limit - len(papers)),
                "offset": offset,
                "fields": self.FIELDS,
            }
            payload = self._request_with_rate_limit(url=self.BASE_URL, params=params)
            if payload is None:
                self.last_search_stats["failed_queries"] = int(self.last_search_stats.get("failed_queries", 0)) + 1
                break
            items = payload.get("data") or []
            if not items:
                break
            for item in items:
                paper = self._parse_paper(item)
                if year_from and paper.year and paper.year < year_from:
                    continue
                matched = paper.raw.get("matched_queries", [])
                if isinstance(matched, list):
                    matched.append(f"semantic_scholar:{query}")
                paper.raw["matched_queries"] = matched
                papers.append(paper)
                if len(papers) >= limit:
                    break
            offset += len(items)
            if len(items) < params["limit"]:
                break
        return papers

    def _parse_paper(self, item: dict) -> PaperMetadata:
        external_ids = item.get("externalIds") or {}
        doi = normalize_doi(external_ids.get("DOI"))
        authors = [
            author.get("name")
            for author in item.get("authors", [])
            if isinstance(author, dict) and author.get("name")
        ]
        publication_types = item.get("publicationTypes") or []
        paper_type = publication_types[0] if publication_types else None
        open_access_pdf = item.get("openAccessPdf") or {}
        venue = item.get("venue")
        publication_venue = (item.get("publicationVenue") or {}).get("name")
        journal = publication_venue or venue

        fields_of_study = [str(x) for x in (item.get("fieldsOfStudy") or []) if str(x).strip()]
        s2_fields = []
        for entry in item.get("s2FieldsOfStudy") or []:
            if isinstance(entry, dict):
                cat = entry.get("category")
                if cat:
                    s2_fields.append(str(cat))
            elif str(entry).strip():
                s2_fields.append(str(entry))
        keywords = self._dedupe(fields_of_study + s2_fields)

        return PaperMetadata(
            id=str(item.get("paperId") or doi or item.get("url") or item.get("title") or ""),
            title=item.get("title") or "Untitled",
            abstract=item.get("abstract"),
            keywords=keywords,
            authors=authors,
            year=item.get("year"),
            doi=doi,
            journal=journal,
            publisher=None,
            citation_count=item.get("citationCount"),
            source_database="Semantic Scholar",
            open_access_status="open" if open_access_pdf.get("url") else "unknown",
            pdf_url=open_access_pdf.get("url"),
            landing_page_url=item.get("url"),
            paper_type=paper_type,
            raw={
                "semantic_scholar": item,
                "influentialCitationCount": item.get("influentialCitationCount"),
                "fieldsOfStudy": fields_of_study,
                "s2FieldsOfStudy": s2_fields,
                "subjects": keywords,
                "concepts": keywords,
                "topics": [],
                "primary_topic": None,
                "matched_queries": [],
                "source_priority": "high",
            },
        )

    def _request_with_rate_limit(self, url: str, params: dict) -> dict | None:
        if not self.api_key or self._disabled_due_auth:
            return None
        headers = {
            "x-api-key": self.api_key,
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        for attempt in range(1, 4):
            self._respect_rate_limit()
            try:
                with httpx.Client(timeout=self.timeout, headers=headers) as client:
                    response = client.get(url, params=params)
                self._last_request_ts = time.perf_counter()
                if response.status_code == 429:
                    wait_seconds = min(10, 5 + attempt * 2)
                    logger.warning("Semantic Scholar 429 rate limited. Retry in %ss", wait_seconds)
                    time.sleep(wait_seconds)
                    continue
                if response.status_code in (401, 403):
                    logger.warning(
                        "Semantic Scholar authorization failed (status=%s). "
                        "API key may be invalid or lacks permissions. Source will be skipped.",
                        response.status_code,
                    )
                    self._disabled_due_auth = True
                    self.last_status = "failed"
                    self.last_reason = f"authorization failed ({response.status_code})"
                    return None
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                return None
            except httpx.RequestError as exc:  # pragma: no cover
                logger.warning("Semantic Scholar request error (attempt %s/3): %s", attempt, exc)
                time.sleep(2 + attempt)
            except httpx.HTTPStatusError as exc:  # pragma: no cover
                logger.warning("Semantic Scholar HTTP error (attempt %s/3): %s", attempt, exc)
                time.sleep(2 + attempt)
            except Exception as exc:  # pragma: no cover
                logger.warning("Semantic Scholar unexpected error (attempt %s/3): %s", attempt, exc)
                time.sleep(2 + attempt)
        return None

    def _respect_rate_limit(self) -> None:
        elapsed = time.perf_counter() - self._last_request_ts
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)

    def _build_query_pool(self, expanded_query: ExpandedQuery) -> list[str]:
        queries: list[str] = [expanded_query.original_topic]
        if expanded_query.english_topic:
            queries.append(expanded_query.english_topic)
        if expanded_query.chinese_topic:
            queries.append(expanded_query.chinese_topic)
        queries.extend(expanded_query.phrase_queries[:6])
        queries.extend(expanded_query.loose_queries[:8])

        en_keywords = expanded_query.english_keywords[:8]
        zh_keywords = expanded_query.chinese_keywords[:8]
        for i in range(0, len(en_keywords), 2):
            queries.append(" ".join(en_keywords[i : i + 3]))
        for i in range(0, len(zh_keywords), 2):
            queries.append(" ".join(zh_keywords[i : i + 3]))
        return self._dedupe(queries)[:16]

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            norm = " ".join(str(value).split())
            key = norm.lower()
            if norm and key not in seen:
                seen.add(key)
                result.append(norm)
        return result
