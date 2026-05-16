from __future__ import annotations

import time
from urllib.parse import quote

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from litbase_ai.enrich.base import BaseEnricher
from litbase_ai.models import PaperMetadata
from litbase_ai.utils.logging import get_logger
from litbase_ai.utils.text import normalize_doi


logger = get_logger(__name__)


class UnpaywallClient(BaseEnricher):
    """Unpaywall client for legal open-access metadata enrichment."""

    BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(self, email: str | None):
        self.email = email
        self.user_agent = "LitBase-AI/0.1"
        self.timeout = 10.0
        self.last_stats: dict[str, int] = {
            "total_papers": 0,
            "doi_count": 0,
            "queried_count": 0,
            "oa_found": 0,
            "pdf_found": 0,
            "failed": 0,
        }

    def enrich(self, papers: list[PaperMetadata], progress=None) -> list[PaperMetadata]:
        self.last_stats = {
            "total_papers": len(papers),
            "doi_count": 0,
            "queried_count": 0,
            "oa_found": 0,
            "pdf_found": 0,
            "failed": 0,
        }
        if not self.email:
            logger.info("UNPAYWALL_EMAIL missing; skip Unpaywall enrichment.")
            return papers

        doi_papers = [paper for paper in papers if normalize_doi(paper.doi)]
        self.last_stats["doi_count"] = len(doi_papers)
        task_id = None
        if progress:
            task_id = progress.task(
                f"Enriching OA links with Unpaywall ({len(doi_papers)} DOI)",
                total=len(doi_papers),
            )

        enriched: list[PaperMetadata] = []
        for paper in papers:
            if not paper.doi:
                enriched.append(paper)
                continue

            doi = normalize_doi(paper.doi)
            if not doi:
                enriched.append(paper)
                continue
            try:
                result = self._query_doi(doi)
                self.last_stats["queried_count"] += 1
                if result:
                    paper = self._apply_unpaywall_result(paper, result)
                    if result.get("is_oa"):
                        self.last_stats["oa_found"] += 1
                    best_oa = result.get("best_oa_location") or {}
                    if best_oa.get("url_for_pdf"):
                        self.last_stats["pdf_found"] += 1
            except Exception as exc:  # pragma: no cover
                logger.warning("Unpaywall failed for DOI %s: %s", doi, exc)
                self.last_stats["failed"] += 1
            enriched.append(paper)
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            time.sleep(0.15)

        if progress:
            progress.log(
                "Unpaywall summary: "
                f"doi={self.last_stats['doi_count']} queried={self.last_stats['queried_count']} "
                f"oa={self.last_stats['oa_found']} pdf={self.last_stats['pdf_found']} failed={self.last_stats['failed']}"
            )
        return enriched

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _query_doi(self, doi: str) -> dict | None:
        encoded_doi = quote(doi, safe="")
        url = f"{self.BASE_URL}/{encoded_doi}"
        params = {"email": self.email}
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.get(url, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()

    def _apply_unpaywall_result(self, paper: PaperMetadata, result: dict) -> PaperMetadata:
        paper.raw["unpaywall"] = result
        is_oa = bool(result.get("is_oa"))
        best_oa_location = result.get("best_oa_location") or {}
        oa_status = result.get("oa_status")
        if is_oa and oa_status:
            paper.open_access_status = oa_status
        if is_oa and best_oa_location.get("url_for_pdf"):
            paper.pdf_url = best_oa_location.get("url_for_pdf")
        if is_oa and not paper.landing_page_url:
            paper.landing_page_url = (
                best_oa_location.get("url_for_landing_page")
                or best_oa_location.get("url")
                or paper.landing_page_url
            )
        return paper
