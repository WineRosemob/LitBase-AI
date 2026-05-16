from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from litbase_ai.models import ExpandedQuery, PaperMetadata
from litbase_ai.query.expander import QueryExpander
from litbase_ai.search.base import BaseSearchClient
from litbase_ai.utils.logging import get_logger
from litbase_ai.utils.text import normalize_doi, strip_html


logger = get_logger(__name__)


class ArxivClient(BaseSearchClient):
    """arXiv API client for preprint recall."""

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, max_results_per_query: int = 100):
        self.max_results_per_query = min(max(10, max_results_per_query), 200)
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

    def search_works(self, topic: str, limit: int = 300, year_from: int | None = None) -> list[PaperMetadata]:
        expanded = QueryExpander(llm_scorer=None).expand(topic)
        return self.search_with_expanded_query(expanded_query=expanded, limit=limit, year_from=year_from)

    def search_with_expanded_query(
        self,
        expanded_query: ExpandedQuery,
        limit: int = 300,
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
        max_queries = 4 if limit <= 60 else 8
        queries = queries[:max_queries]
        self.last_search_stats["queries"] = len(queries)
        per_query_limit = max(10, min(self.max_results_per_query, math.ceil(limit / len(queries)) + 10))
        papers: list[PaperMetadata] = []
        task_id = progress.task("Searching arXiv queries", total=len(queries)) if progress else None
        consecutive_failures = 0
        for query in queries:
            if len(papers) >= limit:
                break
            batch = self._search_single_query(
                query=query,
                limit=min(per_query_limit, limit - len(papers)),
                year_from=year_from,
            )
            papers.extend(batch)
            consecutive_failures = consecutive_failures + 1 if not batch else 0
            if progress and task_id is not None:
                progress.update(task_id, advance=1, description=f"arXiv: {query[:60]} ({len(batch)})")
            if consecutive_failures >= 3:
                logger.warning("arXiv had %s consecutive empty/failed queries. Stop early.", consecutive_failures)
                break
            time.sleep(0.2)
        logger.info("arXiv search returned %s candidates.", len(papers))
        final = papers[:limit]
        self.last_search_stats["returned"] = len(final)
        self.last_search_stats["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        if len(final) == 0 and int(self.last_search_stats.get("failed_queries", 0)) > 0:
            self.last_status = "failed"
            self.last_reason = "all queries failed"
        return final

    def _search_single_query(self, query: str, limit: int, year_from: int | None) -> list[PaperMetadata]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(self.max_results_per_query, limit),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            xml_text = self._request_text(params)
            entries = self._parse_entries(xml_text)
        except Exception as exc:  # pragma: no cover
            logger.warning("arXiv query failed | query=%s | err=%s", query, exc)
            self.last_search_stats["failed_queries"] = int(self.last_search_stats.get("failed_queries", 0)) + 1
            return []

        papers: list[PaperMetadata] = []
        for entry in entries:
            paper = self._parse_entry(entry)
            if year_from and paper.year and paper.year < year_from:
                continue
            matched = paper.raw.get("matched_queries", [])
            if isinstance(matched, list):
                matched.append(f"arxiv:{query}")
            paper.raw["matched_queries"] = matched
            papers.append(paper)
            if len(papers) >= limit:
                break
        return papers

    def _parse_entry(self, entry) -> PaperMetadata:
        ns_atom = "{http://www.w3.org/2005/Atom}"
        ns_arxiv = "{http://arxiv.org/schemas/atom}"

        entry_id = (entry.findtext(f"{ns_atom}id") or "").strip()
        title = strip_html((entry.findtext(f"{ns_atom}title") or "").strip()) or "Untitled"
        abstract = strip_html((entry.findtext(f"{ns_atom}summary") or "").strip())
        published = (entry.findtext(f"{ns_atom}published") or "").strip()
        year = None
        if published:
            try:
                year = datetime.fromisoformat(published.replace("Z", "+00:00")).year
            except ValueError:
                year = None

        authors = []
        for author in entry.findall(f"{ns_atom}author"):
            name = (author.findtext(f"{ns_atom}name") or "").strip()
            if name:
                authors.append(name)

        categories = []
        for category in entry.findall(f"{ns_atom}category"):
            term = (category.attrib.get("term") or "").strip()
            if term:
                categories.append(term)

        doi = normalize_doi((entry.findtext(f"{ns_arxiv}doi") or "").strip() or None)
        pdf_url = None
        landing_page = entry_id
        for link in entry.findall(f"{ns_atom}link"):
            href = link.attrib.get("href")
            title_attr = (link.attrib.get("title") or "").lower()
            if href and "pdf" in title_attr:
                pdf_url = href
            if href and link.attrib.get("rel") == "alternate":
                landing_page = href
        if not pdf_url and entry_id:
            paper_id = entry_id.rsplit("/", 1)[-1]
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

        return PaperMetadata(
            id=entry_id or title,
            title=title,
            abstract=abstract or None,
            keywords=self._dedupe(categories),
            authors=authors,
            year=year,
            doi=doi,
            journal="arXiv",
            publisher="arXiv",
            citation_count=None,
            source_database="arXiv",
            open_access_status="open",
            pdf_url=pdf_url,
            landing_page_url=landing_page,
            paper_type="preprint",
            raw={
                "subjects": categories,
                "concepts": [],
                "topics": [],
                "primary_topic": None,
                "matched_queries": [],
                "source_priority": "medium",
            },
        )

    def _parse_entries(self, xml_text: str):
        root = ET.fromstring(xml_text)
        ns_atom = "{http://www.w3.org/2005/Atom}"
        return root.findall(f"{ns_atom}entry")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _request_text(self, params: dict[str, object]) -> str:
        headers = {"User-Agent": self.user_agent}
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            return response.text

    def _build_query_pool(self, expanded_query: ExpandedQuery) -> list[str]:
        queries = [expanded_query.original_topic]
        if expanded_query.english_topic:
            queries.append(expanded_query.english_topic)
        queries.extend(expanded_query.english_keywords[:10])
        queries.extend(expanded_query.loose_queries)
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = " ".join(query.split())
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                deduped.append(normalized)
        return deduped[:12]

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                result.append(value)
        return result
