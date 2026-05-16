from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from litbase_ai.download.arxiv_download import ArxivDownloader
from litbase_ai.download.base import BaseDownloader
from litbase_ai.download.candidate_utils import dedupe_candidate_entries, extract_pdf_urls_from_html
from litbase_ai.download.ezproxy import EZProxyClient
from litbase_ai.download.inst_proxy import InstProxyClient
from litbase_ai.download.legal_source_resolver import LegalPDFSourceResolver
from litbase_ai.download.libgen import LibGenClient
from litbase_ai.download.scihub import SciHubClient
from litbase_ai.download.source_scoring import DownloadSourceScorer, normalize_source_label
from litbase_ai.models import ScoredPaper
from litbase_ai.utils.logging import get_logger
from litbase_ai.utils.text import clean_filename, extract_first_author, is_http_url, short_title


logger = get_logger(__name__)


class PDFDownloader(BaseDownloader):
    """PDF downloader with OA sources + Sci-Hub/LibGen/arXiv fallback.

    Source priority: legal OA first, then Sci-Hub/LibGen/arXiv as fallback.
    Blocked domain policy is configurable via enable_scihub/enable_libgen.
    """

    BLOCKED_DOMAIN_KEYWORDS: set[str] = set()

    STATIC_SOURCE_PRIORITY = {
        "metadata.pdf_url": 1.20,
        "raw.pdf_url": 1.12,
        "ezproxy": 1.08,
        "inst_proxy": 1.06,
        "unpaywall": 1.05,
        "pmc": 1.02,
        "openalex": 1.00,
        "europepmc": 0.98,
        "semantic_scholar": 0.96,
        "doaj": 0.94,
        "core": 0.93,
        "openaire": 0.92,
        "crossref": 0.88,
        "arxiv": 0.86,
        "scihub": 0.72,
        "libgen": 0.64,
        "metadata.landing_page_url": 0.72,
        "raw": 0.68,
        "metadata": 0.66,
    }

    def __init__(
        self,
        output_dir: Path,
        threshold: float = 75,
        *,
        openalex_mailto: str | None = None,
        unpaywall_email: str | None = None,
        core_api_key: str | None = None,
        enable_discovery: bool = True,
        enable_crossref_page_scrape: bool = True,
        enable_scihub: bool = False,
        enable_libgen: bool = False,
        enable_arxiv_download: bool = False,
        enable_ezproxy: bool = False,
        enable_inst_proxy: bool = False,
        scihub_domains: list[str] | None = None,
        libgen_mirrors: list[str] | None = None,
        ezproxy_template: str | None = None,
        ezproxy_cookie_file: str | None = None,
        inst_proxy_mode: str = "http_proxy",
        inst_proxy_url: str | None = None,
        inst_proxy_cookie_file: str | None = None,
        inst_proxy_school: str | None = None,
        inst_proxy_disabled_host_keywords: list[str] | None = None,
        block_keywords: set[str] | None = None,
        proxy: str | None = None,
        connect_timeout: float = 15.0,
        read_timeout: float = 30.0,
        request_delay_min: float = 0.0,
        request_delay_max: float = 0.0,
    ):
        self.output_dir = output_dir
        self.pdf_dir = self.output_dir / "pdf"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.user_agent = "LitBase-AI/0.4 (+multi-source-discovery)"
        self.proxy = proxy
        self.timeout = httpx.Timeout(
            connect=max(1.0, float(connect_timeout)),
            read=max(1.0, float(read_timeout)),
            write=max(1.0, float(read_timeout)),
            pool=max(1.0, float(connect_timeout)),
        )
        self.request_delay_min = max(0.0, float(request_delay_min))
        self.request_delay_max = max(self.request_delay_min, float(request_delay_max))
        self.max_retries_per_url = 3
        self.max_html_bytes = 1_500_000

        # Additional source toggles
        self.enable_scihub = enable_scihub
        self.enable_libgen = enable_libgen
        self.enable_arxiv_download = enable_arxiv_download
        self.enable_ezproxy = enable_ezproxy
        self.enable_inst_proxy = enable_inst_proxy

        # Set blocked keywords (empty by default = no blocking; set to restore old behavior)
        if block_keywords is not None:
            self.BLOCKED_DOMAIN_KEYWORDS = block_keywords

        self.source_scorer = DownloadSourceScorer()
        self.resolver = LegalPDFSourceResolver(
            openalex_mailto=openalex_mailto,
            unpaywall_email=unpaywall_email,
            core_api_key=core_api_key,
            enable_discovery=enable_discovery,
            enable_crossref_page_scrape=enable_crossref_page_scrape,
            proxy=proxy,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            request_delay_min=request_delay_min,
            request_delay_max=request_delay_max,
            user_agent=self.user_agent,
        )

        # Initialize alternative source clients
        self.scihub_client: SciHubClient | None = None
        if enable_scihub:
            self.scihub_client = SciHubClient(
                domains=scihub_domains,
                proxy=proxy,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )

        self.libgen_client: LibGenClient | None = None
        if enable_libgen:
            self.libgen_client = LibGenClient(
                mirrors=libgen_mirrors,
                proxy=proxy,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )

        self.arxiv_downloader: ArxivDownloader | None = None
        if enable_arxiv_download:
            self.arxiv_downloader = ArxivDownloader(
                proxy=proxy,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )

        self.ezproxy_client: EZProxyClient | None = None
        if enable_ezproxy and ezproxy_template:
            self.ezproxy_client = EZProxyClient(
                proxy_template=ezproxy_template,
                cookie_file=ezproxy_cookie_file,
                proxy=proxy,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )

        self.inst_proxy_client: InstProxyClient | None = None
        if enable_inst_proxy and inst_proxy_url:
            self.inst_proxy_client = InstProxyClient(
                mode=inst_proxy_mode,
                proxy_url=inst_proxy_url,
                school_name=inst_proxy_school,
                cookie_file=inst_proxy_cookie_file,
                disabled_host_keywords=inst_proxy_disabled_host_keywords,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )

        self.last_stats = self._fresh_stats(selected=0)

    def _fresh_stats(self, selected: int) -> dict[str, Any]:
        return {
            "selected": selected,
            "has_pdf_url": 0,
            "has_candidate_url": 0,
            "to_download": 0,
            "downloaded": 0,
            "already_exists": 0,
            "skipped_no_pdf": 0,
            "skipped_restricted": 0,
            "skipped_score": 0,
            "failed": 0,
            "attempted_urls": 0,
            "resolved_from_landing": 0,
            "non_pdf_responses": 0,
            "blocked_by_policy": 0,
            "failure_reasons": {},
            "discovery_queries": 0,
            "discovery_candidates": 0,
            "discovery_cache_hits": 0,
            "title_doi_resolved": 0,
        }

    def download_batch(
        self,
        papers: list[ScoredPaper],
        progress=None,
        override_threshold: float | None = None,
    ) -> list[ScoredPaper]:
        threshold = self.threshold if override_threshold is None else override_threshold
        self.last_stats = self._fresh_stats(selected=len(papers))
        self.resolver.reset_stats()

        ready_count = 0
        failure_reason_counter: dict[str, int] = {}
        for paper in papers:
            if paper.metadata.pdf_url:
                self.last_stats["has_pdf_url"] += 1
            if self._collect_candidate_urls(paper):
                self.last_stats["has_candidate_url"] += 1
            status = self._precheck_status(paper, threshold=threshold)
            if status == "ready":
                ready_count += 1
            elif status == "skipped_no_pdf":
                self.last_stats["skipped_no_pdf"] += 1
            elif status == "skipped_restricted":
                self.last_stats["skipped_restricted"] += 1
            else:
                self.last_stats["skipped_score"] += 1
        self.last_stats["to_download"] = ready_count

        task_id = progress.task("Downloading PDFs", total=ready_count) if progress and ready_count > 0 else None
        downloaded: list[ScoredPaper] = []
        for paper in papers:
            updated, status = self.download_one(paper, override_threshold=threshold)
            downloaded.append(updated)
            if status == "downloaded":
                self.last_stats["downloaded"] += 1
                if progress and task_id is not None:
                    progress.update(task_id, advance=1, description=f"Downloading: {self._short_title(paper)}")
            elif status == "already_exists":
                self.last_stats["already_exists"] += 1
                if progress and task_id is not None:
                    progress.update(task_id, advance=1, description=f"Already exists: {self._short_title(paper)}")
            elif status == "failed":
                self.last_stats["failed"] += 1
                if progress and task_id is not None:
                    progress.update(task_id, advance=1, description=f"Failed: {self._short_title(paper)}")

            trace = paper.metadata.raw.get("download_trace") or []
            self.last_stats["attempted_urls"] = int(self.last_stats["attempted_urls"]) + sum(
                1
                for item in trace
                if isinstance(item, dict)
                and item.get("status") in {"attempt", "downloaded", "request_failed", "http_error", "non_pdf"}
            )
            self.last_stats["resolved_from_landing"] = int(self.last_stats["resolved_from_landing"]) + sum(
                1 for item in trace if isinstance(item, dict) and item.get("status") == "resolved_from_landing"
            )
            self.last_stats["non_pdf_responses"] = int(self.last_stats["non_pdf_responses"]) + sum(
                1 for item in trace if isinstance(item, dict) and item.get("status") == "non_pdf"
            )
            self.last_stats["blocked_by_policy"] = int(self.last_stats["blocked_by_policy"]) + sum(
                1 for item in trace if isinstance(item, dict) and item.get("status") == "blocked"
            )
            for item in trace:
                if not isinstance(item, dict):
                    continue
                reason = str(item.get("reason") or "")
                if reason:
                    failure_reason_counter[reason] = failure_reason_counter.get(reason, 0) + 1

        self.last_stats["failure_reasons"] = failure_reason_counter
        self.last_stats["discovery_queries"] = self.resolver.last_stats.get("queries", 0)
        self.last_stats["discovery_candidates"] = self.resolver.last_stats.get("candidates_found", 0)
        self.last_stats["discovery_cache_hits"] = self.resolver.last_stats.get("cache_hits", 0)
        self.last_stats["title_doi_resolved"] = self.resolver.last_stats.get("title_doi_resolved", 0)

        if progress:
            progress.log(
                "PDF download summary: "
                f"downloaded={self.last_stats['downloaded']} already_exists={self.last_stats['already_exists']} "
                f"skipped_no_pdf={self.last_stats['skipped_no_pdf']} skipped_restricted={self.last_stats['skipped_restricted']} "
                f"failed={self.last_stats['failed']} attempted_urls={self.last_stats['attempted_urls']} "
                f"resolved_from_landing={self.last_stats['resolved_from_landing']} discovery_queries={self.last_stats['discovery_queries']}"
            )
        return downloaded

    def download_one(self, paper: ScoredPaper, override_threshold: float | None = None) -> tuple[ScoredPaper, str]:
        threshold = self.threshold if override_threshold is None else override_threshold
        status = self._precheck_status(paper, threshold=threshold)
        if status != "ready":
            if status in {"skipped_no_pdf", "skipped_restricted", "skipped_score"}:
                paper.metadata.raw["download_trace"] = [{"status": status}]
            return paper, status

        candidates = self._collect_candidate_urls(paper)
        if not candidates:
            paper.metadata.raw["download_trace"] = [{"status": "skipped_no_candidate", "reason": "no_legal_candidate_url"}]
            return paper, "skipped_no_pdf"

        path = self.pdf_dir / self._build_filename(paper)
        if path.exists():
            logger.info("PDF already exists, skip: %s", path.name)
            paper.metadata.raw["local_pdf_path"] = str(path)
            paper.metadata.raw["download_trace"] = [{"status": "already_exists", "path": str(path)}]
            return paper, "already_exists"

        success, trace = self._download_from_candidates(candidates=candidates, path=path)
        paper.metadata.raw["download_trace"] = trace
        if success:
            paper.metadata.raw["local_pdf_path"] = str(path)
            if trace:
                last = next(
                    (item for item in reversed(trace) if isinstance(item, dict) and item.get("status") == "downloaded"),
                    trace[-1],
                )
                paper.metadata.raw["download_source"] = last.get("source")
                final_url = last.get("url")
                if isinstance(final_url, str) and final_url:
                    paper.metadata.raw["downloaded_from_url"] = final_url
            return paper, "downloaded"
        return paper, "failed"

    def _should_download(self, paper: ScoredPaper) -> bool:
        return self._precheck_status(paper, threshold=self.threshold) == "ready"

    def _precheck_status(self, paper: ScoredPaper, threshold: float) -> str:
        if (paper.score.final_score or 0) < threshold:
            return "skipped_score"
        if (paper.metadata.source_database or "").strip() == "CNKI":
            cnki_meta = paper.metadata.raw.get("cnki") or {}
            if cnki_meta.get("restricted", True):
                return "skipped_restricted"
        candidates = self._collect_candidate_urls(paper)
        if not candidates:
            return "skipped_no_pdf"
        if all(not self._is_allowed_pdf_url(item.get("url", ""))[0] for item in candidates):
            return "skipped_restricted"
        return "ready"

    def _build_filename(self, paper: ScoredPaper) -> str:
        year = str(paper.metadata.year or "unknown")
        first_author = extract_first_author(paper.metadata.authors)
        title_chunk = short_title(paper.metadata.title, max_words=5)
        doi_chunk = (paper.metadata.doi or "no_doi").replace("/", "_").replace(".", "_")
        filename = f"{year}_{first_author}_{title_chunk}_{doi_chunk}.pdf"
        return clean_filename(filename, max_len=180)

    def _is_allowed_pdf_url(self, url: str) -> tuple[bool, str]:
        if not is_http_url(url):
            return False, "non_http_scheme"
        parsed = urlparse(url)
        host_and_path = f"{parsed.netloc}{parsed.path}".lower()
        for blocked in self.BLOCKED_DOMAIN_KEYWORDS:
            if blocked in host_and_path:
                return False, "policy_blocked_domain"
        return True, ""

    def _short_title(self, paper: ScoredPaper) -> str:
        return (paper.metadata.title or "Untitled")[:80]

    def has_download_candidate(self, paper: ScoredPaper) -> bool:
        """Whether paper has at least one legal candidate URL for OA download attempts."""
        return bool(self._collect_candidate_urls(paper))

    def _collect_candidate_urls(self, paper: ScoredPaper) -> list[dict[str, str]]:
        raw = paper.metadata.raw or {}
        candidates: list[dict[str, str]] = []

        def add(url: str | None, source: str, from_landing: bool = False) -> None:
            if not url:
                return
            cleaned = str(url).strip()
            if not cleaned:
                return
            candidates.append(
                {
                    "url": cleaned,
                    "source": source,
                    "from_landing": "1" if from_landing else "0",
                }
            )

        add(paper.metadata.pdf_url, "metadata.pdf_url")
        add(raw.get("pdf_url"), "raw.pdf_url")

        unpaywall = raw.get("unpaywall") or {}
        if isinstance(unpaywall, dict):
            best_oa = unpaywall.get("best_oa_location") or {}
            if isinstance(best_oa, dict):
                add(best_oa.get("url_for_pdf"), "unpaywall.best_oa_location.url_for_pdf")
                add(best_oa.get("url"), "unpaywall.best_oa_location.url", from_landing=True)
                add(best_oa.get("url_for_landing_page"), "unpaywall.best_oa_location.url_for_landing_page", from_landing=True)
            for idx, loc in enumerate(unpaywall.get("oa_locations") or []):
                if not isinstance(loc, dict):
                    continue
                add(loc.get("url_for_pdf"), f"unpaywall.oa_locations[{idx}].url_for_pdf")
                add(loc.get("url"), f"unpaywall.oa_locations[{idx}].url", from_landing=True)
                add(loc.get("url_for_landing_page"), f"unpaywall.oa_locations[{idx}].url_for_landing_page", from_landing=True)

        openalex_best = raw.get("best_oa_location") or {}
        if isinstance(openalex_best, dict):
            add(openalex_best.get("url_for_pdf"), "openalex.best_oa_location.url_for_pdf")
            add(openalex_best.get("url"), "openalex.best_oa_location.url", from_landing=True)

        semantic_raw = raw.get("semantic_scholar") or {}
        if isinstance(semantic_raw, dict):
            oa_pdf = semantic_raw.get("openAccessPdf") or {}
            if isinstance(oa_pdf, dict):
                add(oa_pdf.get("url"), "semantic_scholar.openAccessPdf.url")
            add(semantic_raw.get("url"), "semantic_scholar.url", from_landing=True)

        crossref_origin = raw.get("crossref_origin") or {}
        if isinstance(crossref_origin, dict):
            for idx, link in enumerate(crossref_origin.get("link") or []):
                if not isinstance(link, dict):
                    continue
                link_url = link.get("URL")
                ctype = (link.get("content-type") or "").lower()
                add(link_url, f"crossref.link[{idx}]" + (".pdf" if "pdf" in ctype else ""), from_landing=("pdf" not in ctype))

        add(paper.metadata.landing_page_url, "metadata.landing_page_url", from_landing=True)

        # ── arXiv direct download ──────────────────────────────────────
        if self.enable_arxiv_download and self.arxiv_downloader is not None:
            arxiv_id = self._find_arxiv_id(paper)
            if arxiv_id:
                pdf_url = self.arxiv_downloader.resolve_pdf_url(arxiv_id)
                if pdf_url:
                    add(pdf_url, "arxiv")

        # ── Sci-Hub candidates ─────────────────────────────────────────
        if self.enable_scihub and self.scihub_client is not None:
            doi = paper.metadata.doi
            if doi and isinstance(doi, str) and doi.strip():
                # Do not resolve Sci-Hub URL here to avoid extra network calls
                # during candidate collection. The full client flow runs lazily
                # in _download_from_candidates when legal sources are exhausted.
                candidates.append({
                    "url": f"https://sci-hub.se/{doi}",
                    "source": "scihub",
                    "from_landing": "0",
                    "doi": doi.strip(),
                })

        # ── LibGen candidates ──────────────────────────────────────────
        if self.enable_libgen and self.libgen_client is not None:
            doi = paper.metadata.doi
            title = paper.metadata.title or ""
            if (isinstance(doi, str) and doi.strip()) or title:
                # Do not resolve LibGen URL here to avoid extra network calls
                # during candidate collection. Resolve lazily in fallback flow.
                candidates.append({
                    "url": f"https://libgen.li/scimag/ads.php?doi={doi}" if doi else "",
                    "source": "libgen",
                    "from_landing": "0",
                    "doi": (doi or "").strip() if isinstance(doi, str) else "",
                    "title": title,
                })

        # ── EZProxy candidates ────────────────────────────────────────
        if self.enable_ezproxy and self.ezproxy_client is not None:
            # Try proxying the direct PDF URL
            pdf_url = paper.metadata.pdf_url or raw.get("pdf_url")
            if pdf_url:
                proxy_url = self.ezproxy_client.resolve_pdf_url(pdf_url)
                if proxy_url:
                    add(proxy_url, "ezproxy")

            # Also try proxying the landing page
            landing = paper.metadata.landing_page_url
            if landing:
                proxy_landing = self.ezproxy_client.resolve_pdf_url(landing)
                if proxy_landing:
                    add(proxy_landing, "ezproxy", from_landing=True)

        # ── Institutional proxy candidates ────────────────────────────
        if self.enable_inst_proxy and self.inst_proxy_client is not None:
            pdf_url = paper.metadata.pdf_url or raw.get("pdf_url")
            if pdf_url:
                # Keep original publisher URL here.
                # URL rewrite/proxy routing and policy checks happen inside
                # InstProxyClient.try_download(), so disabled publisher policy
                # can inspect the true target host.
                candidates.append({
                    "url": str(pdf_url),
                    "source": "inst_proxy",
                    "from_landing": "0",
                })

        # ── Legal OA resolver ──────────────────────────────────────────
        for item in self.resolver.resolve(paper.metadata):
            if isinstance(item, dict):
                candidates.append({str(key): str(value) for key, value in item.items() if value is not None})

        deduped = dedupe_candidate_entries(candidates)
        return self._sort_candidates(deduped)

    def _sort_candidates(self, candidates: list[dict[str, str]]) -> list[dict[str, str]]:
        def key(item: dict[str, str]) -> tuple[float, float, str]:
            source = item.get("source", "unknown")
            bucket = normalize_source_label(source)
            direct_bonus = 0.14 if item.get("from_landing") != "1" else 0.0
            html_penalty = -0.04 if source.endswith(":html") else 0.0
            static_priority = self.STATIC_SOURCE_PRIORITY.get(source, self.STATIC_SOURCE_PRIORITY.get(bucket, 0.55))
            dynamic_priority = self.source_scorer.success_score(source)
            latency_ms = self.source_scorer.latency_ms(source)
            combined = static_priority + direct_bonus + html_penalty + dynamic_priority
            return (-combined, latency_ms, item.get("url", ""))

        return sorted(candidates, key=key)

    def _download_from_candidates(self, candidates: list[dict[str, str]], path: Path) -> tuple[bool, list[dict[str, str]]]:
        trace: list[dict[str, str]] = []
        headers = {"User-Agent": self.user_agent, "Accept": "application/pdf,application/octet-stream,text/html,*/*"}
        client_kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "follow_redirects": True,
            "headers": headers,
            "trust_env": not bool(self.proxy),
        }
        if self.proxy:
            client_kwargs["proxy"] = self.proxy
        with httpx.Client(**client_kwargs) as client:
            for candidate in candidates:
                url = candidate.get("url", "")
                source = candidate.get("source", "unknown")
                from_landing = candidate.get("from_landing") == "1"
                allowed, reason = self._is_allowed_pdf_url(url)
                if not allowed:
                    trace.append({"status": "blocked", "url": url, "source": source, "reason": reason})
                    continue

                trace.append({"status": "attempt", "url": url, "source": source})
                t0 = time.perf_counter()

                # ── Sci-Hub: use full client for domain-rotated download ──
                if self.enable_scihub and self.scihub_client is not None and source == "scihub":
                    doi = self._extract_doi_from_candidate(candidate)
                    if doi:
                        sci_ok, sci_info = self.scihub_client.try_download(doi=doi, output_path=path)
                        sci_trace = sci_info.get("trace") or []
                        trace.extend(sci_trace)
                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        self.source_scorer.record(source, sci_ok, latency_ms=elapsed_ms, reason="" if sci_ok else "scihub_failed")
                        if sci_ok:
                            return True, trace
                        continue

                # ── LibGen: use full client for mirror-rotated download ──
                if self.enable_libgen and self.libgen_client is not None and source == "libgen":
                    doi = self._extract_doi_from_candidate(candidate)
                    title = candidate.get("title", "")
                    if doi or title:
                        lg_ok, lg_info = self.libgen_client.try_download(doi=doi, output_path=path, title=title)
                        lg_trace = lg_info.get("trace") or []
                        trace.extend(lg_trace)
                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        self.source_scorer.record(source, lg_ok, latency_ms=elapsed_ms, reason="" if lg_ok else "libgen_failed")
                        if lg_ok:
                            return True, trace
                        continue

                # ── EZProxy: use EZProxy client for proxied download ──
                if self.enable_ezproxy and self.ezproxy_client is not None and source == "ezproxy":
                    # The URL is already a proxy URL, download through EZProxy
                    ez_ok, ez_info = self.ezproxy_client.try_download(original_url=url, output_path=path)
                    ez_trace = ez_info.get("trace") or []
                    trace.extend(ez_trace)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self.source_scorer.record(source, ez_ok, latency_ms=elapsed_ms, reason="" if ez_ok else "ezproxy_failed")
                    if ez_ok:
                        return True, trace
                    continue

                # ── InstProxy: use institutional proxy client ──────────
                if self.enable_inst_proxy and self.inst_proxy_client is not None and source == "inst_proxy":
                    ip_ok, ip_info = self.inst_proxy_client.try_download(original_url=url, output_path=path)
                    ip_trace = ip_info.get("trace") or []
                    trace.extend(ip_trace)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self.source_scorer.record(source, ip_ok, latency_ms=elapsed_ms, reason="" if ip_ok else "inst_proxy_failed")
                    if ip_ok:
                        return True, trace
                    continue

                # ── Standard HTTP download ──────────────────────────────
                ok, info, resolved = self._attempt_candidate(
                    client=client,
                    url=url,
                    path=path,
                    source=source,
                    from_landing=from_landing,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                trace.extend(info)
                if resolved:
                    trace.append({"status": "resolved_from_landing", "url": resolved, "source": source})
                reason = self._last_reason_from_info(info)
                self.source_scorer.record(source, ok, latency_ms=elapsed_ms, reason=reason)
                if ok:
                    return True, trace
        return False, trace

    def _extract_doi_from_candidate(self, candidate: dict[str, str]) -> str:
        """Try to extract DOI from candidate metadata or URL."""
        doi = candidate.get("doi", "")
        if doi:
            return str(doi)
        # Try the URL as last resort (Sci-Hub URLs may contain DOI)
        url = candidate.get("url", "")
        if "/doi.org/" in url:
            return url.split("/doi.org/")[-1].split("?")[0].split("#")[0]
        return ""

    def _find_arxiv_id(self, paper: ScoredPaper) -> str | None:
        """Extract arXiv ID from a ScoredPaper using multiple heuristics.

        Checks (in order):
        1. raw['arxiv_id'] or raw['arxivId']
        2. .pdf_url or .landing_page_url containing arxiv.org
        3. raw dict values containing arxiv URLs
        4. source_database == 'arXiv' and extracted from ID
        """
        if self.arxiv_downloader is None:
            return None

        raw = paper.metadata.raw or {}

        # 1. Direct arXiv ID in raw dict
        for key in ("arxiv_id", "arxivId", "ArXivId"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                cleaned = self.arxiv_downloader.extract_arxiv_id(val)
                if cleaned:
                    return cleaned

        # 2. Check pdf_url and landing_page_url
        for url in (paper.metadata.pdf_url, paper.metadata.landing_page_url):
            if url:
                aid = self.arxiv_downloader.extract_arxiv_id(url)
                if aid:
                    return aid

        # 3. Search raw dict for arXiv URLs
        for val in raw.values():
            if isinstance(val, str) and "arxiv.org" in val:
                aid = self.arxiv_downloader.extract_arxiv_id(val)
                if aid:
                    return aid

        # 4. If source is arXiv and id looks like an arXiv ID
        if paper.metadata.source_database == "arXiv":
            aid = self.arxiv_downloader.extract_arxiv_id(paper.metadata.id)
            if aid:
                return aid

        return None

    def _attempt_candidate(
        self,
        client: httpx.Client,
        url: str,
        path: Path,
        source: str,
        from_landing: bool,
    ) -> tuple[bool, list[dict[str, str]], str | None]:
        response = self._request_with_retry(client=client, url=url)
        if response is None:
            return False, [{"status": "request_failed", "url": url, "source": source, "reason": "network_or_timeout"}], None
        if response.status_code >= 400:
            return False, [{"status": "http_error", "url": str(response.url), "source": source, "reason": f"http_{response.status_code}"}], None

        content = response.content or b""
        if self._looks_like_pdf(response=response, content=content):
            if self._write_valid_pdf(content=content, path=path):
                return True, [{"status": "downloaded", "url": str(response.url), "source": source}], None
            return False, [{"status": "non_pdf", "url": str(response.url), "source": source, "reason": "invalid_pdf_file"}], None

        if not from_landing and not self._looks_like_html(response=response, content=content):
            return False, [{"status": "non_pdf", "url": str(response.url), "source": source, "reason": "content_not_pdf"}], None

        html_text = self._decode_html(content=content)
        if not html_text:
            return False, [{"status": "non_pdf", "url": str(response.url), "source": source, "reason": "empty_html"}], None
        extracted_urls = extract_pdf_urls_from_html(html=html_text, base_url=str(response.url))
        if not extracted_urls:
            return False, [{"status": "non_pdf", "url": str(response.url), "source": source, "reason": "no_pdf_link_in_html"}], None

        info: list[dict[str, str]] = []
        for extracted in extracted_urls:
            allowed, reason = self._is_allowed_pdf_url(extracted)
            if not allowed:
                info.append({"status": "blocked", "url": extracted, "source": f"{source}:html", "reason": reason})
                continue
            pdf_resp = self._request_with_retry(client=client, url=extracted, referer=str(response.url))
            if pdf_resp is None:
                info.append({"status": "request_failed", "url": extracted, "source": f"{source}:html", "reason": "network_or_timeout"})
                continue
            if pdf_resp.status_code >= 400:
                info.append({"status": "http_error", "url": str(pdf_resp.url), "source": f"{source}:html", "reason": f"http_{pdf_resp.status_code}"})
                continue
            pdf_content = pdf_resp.content or b""
            if not self._looks_like_pdf(response=pdf_resp, content=pdf_content):
                info.append({"status": "non_pdf", "url": str(pdf_resp.url), "source": f"{source}:html", "reason": "resolved_link_not_pdf"})
                continue
            if not self._write_valid_pdf(content=pdf_content, path=path):
                info.append({"status": "non_pdf", "url": str(pdf_resp.url), "source": f"{source}:html", "reason": "invalid_pdf_file"})
                continue
            info.append({"status": "downloaded", "url": str(pdf_resp.url), "source": f"{source}:html"})
            return True, info, extracted
        return False, info, None

    def _request_with_retry(
        self,
        client: httpx.Client,
        url: str,
        referer: str | None = None,
    ) -> httpx.Response | None:
        headers = {"Accept": "application/pdf,application/octet-stream,text/html,*/*"}
        if referer:
            headers["Referer"] = referer
        for attempt in range(1, self.max_retries_per_url + 1):
            try:
                self._sleep_between_requests()
                response = client.get(url, headers=headers)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries_per_url:
                    sleep_s = min(8.0, 1.2 * attempt)
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        sleep_s = min(10.0, float(retry_after))
                    time.sleep(sleep_s)
                    continue
                return response
            except Exception as exc:  # pragma: no cover
                logger.warning("Download request failed (%s/%s) for %s: %s", attempt, self.max_retries_per_url, url, exc)
                if attempt < self.max_retries_per_url:
                    time.sleep(min(6.0, 1.2 * attempt))
                    continue
                return None
        return None

    def _sleep_between_requests(self) -> None:
        if self.request_delay_max <= 0:
            return
        time.sleep(random.uniform(self.request_delay_min, self.request_delay_max))

    def _looks_like_pdf(self, response: httpx.Response, content: bytes) -> bool:
        content_type = (response.headers.get("Content-Type") or "").lower()
        head = content[:2048]
        if b"%PDF" in head:
            return True
        if "application/pdf" in content_type and len(content) > 1024:
            return True
        return False

    def _looks_like_html(self, response: httpx.Response, content: bytes) -> bool:
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            return True
        head = content[:300].lstrip().lower()
        return head.startswith(b"<!doctype html") or head.startswith(b"<html")

    def _decode_html(self, content: bytes) -> str:
        if not content:
            return ""
        if len(content) > self.max_html_bytes:
            content = content[: self.max_html_bytes]
        for encoding in ("utf-8", "latin-1"):
            try:
                return content.decode(encoding, errors="ignore")
            except Exception:  # pragma: no cover
                continue
        return ""

    def _write_valid_pdf(self, content: bytes, path: Path) -> bool:
        if not content or len(content) < 1024:
            return False
        tmp_path = path.with_suffix(path.suffix + ".part")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(content)
            if not self._is_pdf_file(tmp_path):
                tmp_path.unlink(missing_ok=True)
                return False
            tmp_path.replace(path)
            logger.info("Downloaded PDF: %s", path)
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed writing PDF %s: %s", path, exc)
            tmp_path.unlink(missing_ok=True)
            return False

    def _is_pdf_file(self, path: Path) -> bool:
        try:
            size = path.stat().st_size
            if size < 1000:
                return False
            with path.open("rb") as fh:
                header = fh.read(5)
                if header != b"%PDF-":
                    return False
                fh.seek(max(0, size - 2048))
                tail = fh.read()
            return b"%%EOF" in tail or size > 4_096
        except OSError:
            return False

    def _last_reason_from_info(self, info: list[dict[str, str]]) -> str:
        for item in reversed(info):
            if item.get("reason"):
                return str(item["reason"])
            status = item.get("status")
            if status:
                return str(status)
        return ""
