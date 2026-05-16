from __future__ import annotations

import copy
import random
import time
import urllib.parse
from difflib import SequenceMatcher
from typing import Any

import httpx

from litbase_ai.download.candidate_utils import dedupe_candidate_entries, extract_pdf_urls_from_html, is_plausible_pdf_url
from litbase_ai.models import PaperMetadata
from litbase_ai.utils.logging import get_logger
from litbase_ai.utils.text import normalize_doi, normalize_title


logger = get_logger(__name__)
TITLE_SIMILARITY_THRESHOLD = 0.76


class LegalPDFSourceResolver:
    """Discover additional legal OA download candidates before file download."""

    def __init__(
        self,
        openalex_mailto: str | None = None,
        unpaywall_email: str | None = None,
        core_api_key: str | None = None,
        *,
        enable_discovery: bool = True,
        enable_crossref_page_scrape: bool = True,
        proxy: str | None = None,
        connect_timeout: float = 15.0,
        read_timeout: float = 30.0,
        request_delay_min: float = 0.0,
        request_delay_max: float = 0.0,
        user_agent: str = "LitBase-AI/0.3 (+legal-oa-discovery)",
    ):
        self.openalex_mailto = openalex_mailto
        self.unpaywall_email = unpaywall_email
        self.core_api_key = core_api_key
        self.enable_discovery = enable_discovery
        self.enable_crossref_page_scrape = enable_crossref_page_scrape
        self.proxy = proxy
        self.timeout = httpx.Timeout(
            connect=max(1.0, float(connect_timeout)),
            read=max(1.0, float(read_timeout)),
            write=max(1.0, float(read_timeout)),
            pool=max(1.0, float(connect_timeout)),
        )
        self.request_delay_min = max(0.0, float(request_delay_min))
        self.request_delay_max = max(self.request_delay_min, float(request_delay_max))
        self.user_agent = user_agent
        self._cache: dict[str, dict[str, Any]] = {}
        self.last_stats: dict[str, int] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        self.last_stats = {
            "papers_seen": 0,
            "cache_hits": 0,
            "queries": 0,
            "sources_with_candidates": 0,
            "candidates_found": 0,
            "title_doi_resolved": 0,
        }

    def resolve(self, paper: PaperMetadata) -> list[dict[str, str]]:
        if not self.enable_discovery:
            return []

        self.last_stats["papers_seen"] += 1
        cache_key = self._paper_cache_key(paper)
        cached = self._cache.get(cache_key)
        if cached:
            self.last_stats["cache_hits"] += 1
            paper.raw["download_discovery"] = copy.deepcopy(cached)
            resolved_doi = cached.get("resolved_doi")
            if resolved_doi and not paper.doi:
                paper.doi = resolved_doi
            return list(cached.get("candidates") or [])

        discovery = {
            "cache_key": cache_key,
            "queried_at": int(time.time()),
            "resolved_doi": normalize_doi(paper.doi),
            "source_reports": [],
            "candidates": [],
        }
        candidates: list[dict[str, str]] = []

        resolved_doi = normalize_doi(paper.doi)
        if not resolved_doi and paper.title:
            resolved_doi = self._resolve_title_to_doi(paper.title)
            if resolved_doi:
                self.last_stats["title_doi_resolved"] += 1
                paper.doi = resolved_doi
                discovery["resolved_doi"] = resolved_doi
                discovery["resolved_doi_from_title"] = True

        try:
            client_kwargs: dict[str, Any] = {
                "timeout": self.timeout,
                "follow_redirects": True,
                "headers": {"User-Agent": self.user_agent, "Accept": "application/json,text/html,*/*"},
                "trust_env": not bool(self.proxy),
            }
            if self.proxy:
                client_kwargs["proxy"] = self.proxy
            with httpx.Client(**client_kwargs) as client:
                if resolved_doi:
                    candidates.extend(self._discover_from_unpaywall(client, resolved_doi, paper, discovery))
                    candidates.extend(self._discover_from_openalex(client, resolved_doi, paper, discovery))
                    candidates.extend(self._discover_from_crossref(client, resolved_doi, paper, discovery))
                    candidates.extend(self._discover_from_openaire(client, resolved_doi, discovery))
                    candidates.extend(self._discover_from_doaj(client, resolved_doi, discovery))
                    candidates.extend(self._discover_from_europepmc(client, resolved_doi, discovery))
                    candidates.extend(self._discover_from_pmc(client, resolved_doi, discovery))
                    candidates.extend(self._discover_from_core(client, resolved_doi, discovery))
                    if self.enable_crossref_page_scrape:
                        candidates.extend(self._discover_from_landing_page(client, resolved_doi, paper, discovery))
                elif paper.landing_page_url:
                    candidates.extend(self._scrape_landing_url(client, paper.landing_page_url, "metadata.landing_page_url", discovery))
        except Exception as exc:  # pragma: no cover
            logger.warning("Download discovery failed for '%s': %s", paper.title[:80], exc)
            discovery["source_reports"].append({"source": "resolver", "status": "error", "reason": str(exc)})

        deduped = dedupe_candidate_entries(candidates)
        final_cache_key = f"doi:{resolved_doi}" if resolved_doi else cache_key
        discovery["cache_key"] = final_cache_key
        discovery["candidates"] = deduped
        paper.raw["download_discovery"] = discovery
        cached_value = copy.deepcopy(discovery)
        self._cache[cache_key] = cached_value
        if final_cache_key != cache_key:
            self._cache[final_cache_key] = copy.deepcopy(discovery)
        self.last_stats["candidates_found"] += len(deduped)
        self.last_stats["sources_with_candidates"] += sum(
            1 for item in discovery["source_reports"] if int(item.get("candidate_count", 0)) > 0
        )
        if discovery.get("resolved_doi") and not paper.doi:
            paper.doi = str(discovery["resolved_doi"])
        return deduped

    def _discover_from_unpaywall(
        self,
        client: httpx.Client,
        doi: str,
        paper: PaperMetadata,
        discovery: dict[str, Any],
    ) -> list[dict[str, str]]:
        payload = paper.raw.get("unpaywall") if isinstance(paper.raw.get("unpaywall"), dict) else None
        status = "cached"
        if payload is None and self.unpaywall_email:
            payload = self._request_json(
                client,
                f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi, safe='')}",
                params={"email": self.unpaywall_email},
            )
            status = "queried"
            if payload is not None:
                paper.raw["unpaywall"] = payload
        if not isinstance(payload, dict):
            discovery["source_reports"].append({"source": "unpaywall", "status": "unavailable", "candidate_count": 0})
            return []

        candidates: list[dict[str, str]] = []
        best = payload.get("best_oa_location") or {}
        if isinstance(best, dict):
            self._add_candidate(candidates, best.get("url_for_pdf"), "unpaywall.best_oa_location.url_for_pdf")
            self._add_candidate(candidates, best.get("url"), "unpaywall.best_oa_location.url", from_landing=True)
            self._add_candidate(
                candidates,
                best.get("url_for_landing_page"),
                "unpaywall.best_oa_location.url_for_landing_page",
                from_landing=True,
            )
        for idx, loc in enumerate(payload.get("oa_locations") or []):
            if not isinstance(loc, dict):
                continue
            self._add_candidate(candidates, loc.get("url_for_pdf"), f"unpaywall.oa_locations[{idx}].url_for_pdf")
            self._add_candidate(candidates, loc.get("url"), f"unpaywall.oa_locations[{idx}].url", from_landing=True)
            self._add_candidate(
                candidates,
                loc.get("url_for_landing_page"),
                f"unpaywall.oa_locations[{idx}].url_for_landing_page",
                from_landing=True,
            )

        best_landing = best.get("url_for_landing_page") or best.get("url")
        if isinstance(best_landing, str) and best_landing and not paper.landing_page_url:
            paper.landing_page_url = best_landing
        if isinstance(best.get("url_for_pdf"), str) and best.get("url_for_pdf") and not paper.pdf_url:
            paper.pdf_url = best.get("url_for_pdf")

        discovery["source_reports"].append({"source": "unpaywall", "status": status, "candidate_count": len(candidates)})
        return candidates

    def _discover_from_openalex(
        self,
        client: httpx.Client,
        doi: str,
        paper: PaperMetadata,
        discovery: dict[str, Any],
    ) -> list[dict[str, str]]:
        payload = None
        raw_origin = paper.raw.get("origin")
        if paper.source_database == "OpenAlex" and isinstance(raw_origin, dict):
            payload = raw_origin
        status = "cached"
        if payload is None:
            url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='')}"
            params = {"mailto": self.openalex_mailto} if self.openalex_mailto else None
            payload = self._request_json(client, url, params=params)
            status = "queried"
            if isinstance(payload, dict):
                paper.raw.setdefault("download_openalex", payload)
        if not isinstance(payload, dict):
            discovery["source_reports"].append({"source": "openalex", "status": "unavailable", "candidate_count": 0})
            return []

        candidates: list[dict[str, str]] = []
        best = payload.get("best_oa_location") or {}
        if isinstance(best, dict):
            self._add_candidate(candidates, best.get("pdf_url"), "openalex.best_oa_location.pdf_url")
            self._add_candidate(candidates, best.get("url_for_pdf"), "openalex.best_oa_location.url_for_pdf")
            self._add_candidate(candidates, best.get("landing_page_url"), "openalex.best_oa_location.landing_page_url", from_landing=True)
            self._add_candidate(candidates, best.get("url"), "openalex.best_oa_location.url", from_landing=True)
        primary = payload.get("primary_location") or {}
        if isinstance(primary, dict):
            self._add_candidate(candidates, primary.get("pdf_url"), "openalex.primary_location.pdf_url")
            self._add_candidate(candidates, primary.get("landing_page_url"), "openalex.primary_location.landing_page_url", from_landing=True)
        open_access = payload.get("open_access") or {}
        if isinstance(open_access, dict):
            self._add_candidate(candidates, open_access.get("oa_url"), "openalex.open_access.oa_url")

        if isinstance(primary, dict) and primary.get("landing_page_url") and not paper.landing_page_url:
            paper.landing_page_url = primary.get("landing_page_url")
        if isinstance(primary, dict) and primary.get("pdf_url") and not paper.pdf_url:
            paper.pdf_url = primary.get("pdf_url")

        discovery["source_reports"].append({"source": "openalex", "status": status, "candidate_count": len(candidates)})
        return candidates

    def _discover_from_crossref(
        self,
        client: httpx.Client,
        doi: str,
        paper: PaperMetadata,
        discovery: dict[str, Any],
    ) -> list[dict[str, str]]:
        payload = paper.raw.get("crossref_origin") if isinstance(paper.raw.get("crossref_origin"), dict) else None
        status = "cached"
        if payload is None:
            wrapper = self._request_json(client, f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}")
            payload = (wrapper or {}).get("message") if isinstance(wrapper, dict) else None
            status = "queried"
            if isinstance(payload, dict):
                paper.raw.setdefault("download_crossref", payload)
        if not isinstance(payload, dict):
            discovery["source_reports"].append({"source": "crossref", "status": "unavailable", "candidate_count": 0})
            return []

        candidates: list[dict[str, str]] = []
        for idx, link in enumerate(payload.get("link") or []):
            if not isinstance(link, dict):
                continue
            ctype = (link.get("content-type") or "").lower()
            self._add_candidate(
                candidates,
                link.get("URL"),
                f"crossref.link[{idx}]" + (".pdf" if "pdf" in ctype else ""),
                from_landing=("pdf" not in ctype),
            )

        landing = payload.get("URL") or paper.landing_page_url
        if isinstance(landing, str) and landing and not paper.landing_page_url:
            paper.landing_page_url = landing
        discovery["source_reports"].append({"source": "crossref", "status": status, "candidate_count": len(candidates)})
        return candidates

    def _discover_from_openaire(
        self,
        client: httpx.Client,
        doi: str,
        discovery: dict[str, Any],
    ) -> list[dict[str, str]]:
        payload = self._request_json(
            client,
            "https://api.openaire.eu/search/publications",
            params={"doi": doi, "format": "json"},
        )
        if not isinstance(payload, dict):
            discovery["source_reports"].append({"source": "openaire", "status": "unavailable", "candidate_count": 0})
            return []

        candidates: list[dict[str, str]] = []
        results = (((payload.get("response") or {}).get("results") or {}).get("result") or [])
        if isinstance(results, dict):
            results = [results]
        for pub in results:
            main = (((pub.get("metadata") or {}).get("oaf:entity") or {}).get("oaf:result") or {})
            if not isinstance(main, dict):
                continue
            children = ((main.get("children") or {}).get("result") or [])
            if isinstance(children, dict):
                children = [children]
            for idx, child in enumerate(children):
                if not isinstance(child, dict):
                    continue
                instance = child.get("instance") or {}
                if not isinstance(instance, dict):
                    continue
                url_obj = instance.get("url") or {}
                if isinstance(url_obj, dict):
                    self._add_candidate(candidates, url_obj.get("$"), f"openaire.instance[{idx}].url", from_landing=True)
                webresource = instance.get("webresource") or {}
                if isinstance(webresource, dict):
                    inner_url = webresource.get("url") or {}
                    if isinstance(inner_url, dict):
                        self._add_candidate(candidates, inner_url.get("$"), f"openaire.instance[{idx}].webresource", from_landing=True)

        discovery["source_reports"].append({"source": "openaire", "status": "queried", "candidate_count": len(candidates)})
        return candidates

    def _discover_from_doaj(
        self,
        client: httpx.Client,
        doi: str,
        discovery: dict[str, Any],
    ) -> list[dict[str, str]]:
        payload = self._request_json(client, f"https://doaj.org/api/v2/search/articles/doi:{doi}")
        if not isinstance(payload, dict):
            discovery["source_reports"].append({"source": "doaj", "status": "unavailable", "candidate_count": 0})
            return []

        candidates: list[dict[str, str]] = []
        results = payload.get("results") or []
        for item_idx, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            bibjson = item.get("bibjson") or {}
            links = bibjson.get("link") or []
            for link_idx, link in enumerate(links):
                if not isinstance(link, dict):
                    continue
                self._add_candidate(
                    candidates,
                    link.get("url"),
                    f"doaj.link[{item_idx}:{link_idx}]",
                    from_landing=(link.get("type") != "fulltext"),
                )

        discovery["source_reports"].append({"source": "doaj", "status": "queried", "candidate_count": len(candidates)})
        return candidates

    def _discover_from_europepmc(
        self,
        client: httpx.Client,
        doi: str,
        discovery: dict[str, Any],
    ) -> list[dict[str, str]]:
        payload = self._request_json(
            client,
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": f"DOI:{doi}", "format": "json", "pageSize": 5},
        )
        if not isinstance(payload, dict):
            discovery["source_reports"].append({"source": "europepmc", "status": "unavailable", "candidate_count": 0})
            return []

        candidates: list[dict[str, str]] = []
        results = ((payload.get("resultList") or {}).get("result") or [])
        if isinstance(results, dict):
            results = [results]
        for result_idx, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            fulltext = result.get("fullTextUrlList") or {}
            urls = fulltext.get("fullTextUrl") or []
            if isinstance(urls, dict):
                urls = [urls]
            for url_idx, entry in enumerate(urls):
                if not isinstance(entry, dict):
                    continue
                candidate_url = str(entry.get("url") or "").strip()
                style = str(entry.get("documentStyle") or "").lower()
                if style == "pdf" or is_plausible_pdf_url(candidate_url):
                    self._add_candidate(
                        candidates,
                        candidate_url,
                        f"europepmc.fullTextUrl[{result_idx}:{url_idx}]",
                    )

        discovery["source_reports"].append({"source": "europepmc", "status": "queried", "candidate_count": len(candidates)})
        return candidates

    def _discover_from_pmc(
        self,
        client: httpx.Client,
        doi: str,
        discovery: dict[str, Any],
    ) -> list[dict[str, str]]:
        payload = self._request_json(
            client,
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
            params={"ids": doi, "format": "json"},
        )
        if not isinstance(payload, dict):
            discovery["source_reports"].append({"source": "pmc", "status": "unavailable", "candidate_count": 0})
            return []

        candidates: list[dict[str, str]] = []
        for record in payload.get("records") or []:
            if not isinstance(record, dict):
                continue
            pmcid = str(record.get("pmcid") or "").strip()
            if not pmcid:
                continue
            self._add_candidate(
                candidates,
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
                f"pmc.{pmcid}.pdf",
            )

        discovery["source_reports"].append({"source": "pmc", "status": "queried", "candidate_count": len(candidates)})
        return candidates

    def _discover_from_core(
        self,
        client: httpx.Client,
        doi: str,
        discovery: dict[str, Any],
    ) -> list[dict[str, str]]:
        headers = {"Authorization": f"Bearer {self.core_api_key}"} if self.core_api_key else None
        payload = self._request_json(
            client,
            "https://api.core.ac.uk/v3/search/works",
            params={"q": f'doi:"{doi}"', "limit": 5},
            headers=headers,
        )
        if not isinstance(payload, dict):
            discovery["source_reports"].append({"source": "core", "status": "unavailable", "candidate_count": 0})
            return []

        candidates: list[dict[str, str]] = []
        for idx, url in enumerate(self._iter_urls(payload)):
            if self._is_core_pdf_candidate(url):
                self._add_candidate(candidates, url, f"core.url[{idx}]")
                if len(candidates) >= 8:
                    break

        discovery["source_reports"].append({"source": "core", "status": "queried", "candidate_count": len(candidates)})
        return candidates

    def _discover_from_landing_page(
        self,
        client: httpx.Client,
        doi: str,
        paper: PaperMetadata,
        discovery: dict[str, Any],
    ) -> list[dict[str, str]]:
        landing_url = paper.landing_page_url or self._resolve_doi_landing_page(client, doi)
        if not landing_url:
            discovery["source_reports"].append({"source": "crossref_page", "status": "unavailable", "candidate_count": 0})
            return []
        if not paper.landing_page_url:
            paper.landing_page_url = landing_url
        return self._scrape_landing_url(client, landing_url, "crossref.page_scrape", discovery)

    def _scrape_landing_url(
        self,
        client: httpx.Client,
        landing_url: str,
        source_prefix: str,
        discovery: dict[str, Any],
    ) -> list[dict[str, str]]:
        html, final_url = self._request_text(client, landing_url)
        if not html:
            discovery["source_reports"].append({"source": source_prefix, "status": "unavailable", "candidate_count": 0})
            return []
        extracted = extract_pdf_urls_from_html(html, final_url)
        candidates: list[dict[str, str]] = []
        for idx, url in enumerate(extracted):
            self._add_candidate(candidates, url, f"{source_prefix}[{idx}]")
        discovery["source_reports"].append({"source": source_prefix, "status": "queried", "candidate_count": len(candidates)})
        return candidates

    def _resolve_title_to_doi(self, title: str) -> str | None:
        clean_title = title.strip()
        if len(clean_title) < 10:
            return None
        try:
            client_kwargs: dict[str, Any] = {
                "timeout": self.timeout,
                "headers": {"User-Agent": self.user_agent, "Accept": "application/json"},
                "trust_env": not bool(self.proxy),
            }
            if self.proxy:
                client_kwargs["proxy"] = self.proxy
            with httpx.Client(**client_kwargs) as client:
                params = {"search": clean_title, "per_page": 5}
                if self.openalex_mailto:
                    params["mailto"] = self.openalex_mailto
                payload = self._request_json(client, "https://api.openalex.org/works", params=params)
        except Exception:  # pragma: no cover
            payload = None
        if not isinstance(payload, dict):
            return None

        best_doi = None
        best_score = 0.0
        title_key = normalize_title(clean_title)
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            result_title = normalize_title(item.get("display_name") or item.get("title"))
            if not result_title:
                continue
            score = SequenceMatcher(None, title_key, result_title).ratio()
            if score > best_score:
                best_score = score
                best_doi = normalize_doi(item.get("doi"))
        if best_score >= TITLE_SIMILARITY_THRESHOLD:
            return best_doi
        return None

    def _request_json(
        self,
        client: httpx.Client,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        self.last_stats["queries"] += 1
        try:
            self._sleep_between_requests()
            request_headers = {"Accept": "application/json"}
            if headers:
                request_headers.update(headers)
            response = client.get(url, params=params, headers=request_headers)
            if response.status_code >= 400:
                return None
            return response.json()
        except Exception:
            return None

    def _request_text(self, client: httpx.Client, url: str) -> tuple[str, str]:
        self.last_stats["queries"] += 1
        try:
            self._sleep_between_requests()
            response = client.get(url, headers={"Accept": "text/html,application/xhtml+xml,*/*"})
            if response.status_code >= 400:
                return "", url
            return response.text or "", str(response.url)
        except Exception:
            return "", url

    def _resolve_doi_landing_page(self, client: httpx.Client, doi: str) -> str | None:
        self.last_stats["queries"] += 1
        try:
            self._sleep_between_requests()
            response = client.head(f"https://doi.org/{doi}", follow_redirects=True)
            if response.status_code >= 400:
                self._sleep_between_requests()
                response = client.get(f"https://doi.org/{doi}", headers={"Accept": "text/html,*/*"})
            if response.status_code >= 400:
                return None
            return str(response.url)
        except Exception:
            return None

    def _sleep_between_requests(self) -> None:
        if self.request_delay_max <= 0:
            return
        time.sleep(random.uniform(self.request_delay_min, self.request_delay_max))

    def _iter_urls(self, obj: Any) -> list[str]:
        urls: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, inner in value.items():
                    if isinstance(inner, str) and ("url" in key.lower() or inner.startswith("http")):
                        urls.append(inner)
                    else:
                        walk(inner)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(obj)
        return urls

    def _is_core_pdf_candidate(self, url: str) -> bool:
        if not is_plausible_pdf_url(url):
            return False
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lower()
        if "core.ac.uk" in host and "/data-providers/" in path:
            return False
        return True

    def _paper_cache_key(self, paper: PaperMetadata) -> str:
        doi = normalize_doi(paper.doi)
        if doi:
            return f"doi:{doi}"
        title = normalize_title(paper.title)
        if title:
            return f"title:{title}"
        return f"paper:{paper.id}"

    def _add_candidate(
        self,
        candidates: list[dict[str, str]],
        url: str | None,
        source: str,
        from_landing: bool = False,
    ) -> None:
        if not url:
            return
        cleaned = str(url).strip()
        if not cleaned:
            return
        if not from_landing and not is_plausible_pdf_url(cleaned):
            return
        candidates.append(
            {
                "url": cleaned,
                "source": source,
                "from_landing": "1" if from_landing else "0",
            }
        )
