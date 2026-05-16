from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from litbase_ai.config import AppConfig
from litbase_ai.download.cookie_bootstrap import load_cookie_bootstrap_policy, publisher_host_keywords
from litbase_ai.download.pdf_downloader import PDFDownloader
from litbase_ai.enrich.journal_rank import JournalRankLookup
from litbase_ai.enrich.unpaywall_client import UnpaywallClient
from litbase_ai.models import ExpandedQuery, PaperMetadata, ScoredPaper
from litbase_ai.query.expander import QueryExpander
from litbase_ai.scoring.aggregator import FinalScoreAggregator
from litbase_ai.scoring.embedding_score import EmbeddingScorer
from litbase_ai.scoring.evidence_score import EvidenceBasedScorer
from litbase_ai.scoring.feedback import HumanFeedbackManager
from litbase_ai.scoring.llm_rubric_score import LLMRubricScorer
from litbase_ai.scoring.llm_score import LLMScorer
from litbase_ai.scoring.rule_score import RuleBasedScorer
from litbase_ai.search.arxiv_client import ArxivClient
from litbase_ai.search.cnki_client import CNKIClient
from litbase_ai.search.crossref_client import CrossrefClient
from litbase_ai.search.merge import PaperMerger
from litbase_ai.search.openalex_client import OpenAlexClient
from litbase_ai.search.semantic_scholar_client import SemanticScholarClient
from litbase_ai.storage.cards import LiteratureCardGenerator
from litbase_ai.storage.exporters import PaperExporter
from litbase_ai.utils.cache import CacheManager
from litbase_ai.utils.logging import LoggerFactory, get_logger
from litbase_ai.utils.progress import ProgressManager


logger = get_logger(__name__)


class LitBasePipeline:
    """Main orchestration pipeline for multi-source retrieval and enhanced scoring."""

    STAGE_NAMES = [
        "Initialize project",
        "Load config",
        "Expand query",
        "Search papers",
        "Merge and deduplicate",
        "Export raw metadata",
        "Enrich open-access links",
        "Rule-based scoring",
        "Evidence extraction",
        "Optional embedding scoring",
        "Select LLM candidates",
        "LLM rubric scoring",
        "Optional human feedback calibration",
        "Final score aggregation",
        "Select papers",
        "Download PDFs",
        "Generate literature cards",
        "Export cards table",
        "Re-export selected papers",
        "Generate summary report",
        "Export final results",
        "Finish",
    ]

    def __init__(
        self,
        config: AppConfig,
        output_dir: Path,
        topic: str,
        limit: int = 500,
        year_from: int | None = None,
        rule_threshold: float = 60,
        llm_threshold: float = 70,
        download_threshold: float = 75,
        candidate_multiplier: int = 3,
        enable_openalex: bool = True,
        enable_crossref: bool = True,
        enable_arxiv: bool = True,
        enable_semantic_scholar: bool | None = None,
        enable_llm: bool = True,
        enable_llm_query_expansion: bool = True,
        enable_cnki: bool = False,
        scoring_mode: str = "balanced",
        llm_rubric_max_papers: int = 100,
        llm_rubric_min_rule_score: float = 40,
        feedback_file: Path | None = None,
        enable_embedding_score: bool | None = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_candidate_mode: str = "hybrid",
        llm_candidate_top_k: int = 50,
        min_llm_candidates: int = 20,
        selection_mode: str = "hybrid",
        selection_top_k: int = 30,
        min_selected: int = 10,
        cards_from: str = "auto",
        top_k_cards: int = 30,
        card_threshold: float = 0,
        download_top_k_if_none: int = 5,
        enable_download_discovery: bool = True,
        enable_crossref_page_scrape: bool = True,
        cnki_headless: bool = True,
        cnki_max_pages: int = 5,
        cnki_limit: int = 200,
        progress_enabled: bool = True,
        progress_style: str = "rich",
        verbose: bool = False,
        quiet: bool = False,
    ):
        self.config = config
        self.output_dir = output_dir
        self.topic = topic
        self.limit = max(1, limit)
        self.year_from = year_from
        self.rule_threshold = float(rule_threshold)
        self.llm_threshold = float(llm_threshold)
        self.download_threshold = float(download_threshold)
        self.candidate_multiplier = max(1, candidate_multiplier)

        self.enable_openalex = enable_openalex
        self.enable_crossref = enable_crossref
        self.enable_arxiv = enable_arxiv
        self.enable_semantic_scholar = enable_semantic_scholar
        self.enable_llm = enable_llm
        self.enable_llm_query_expansion = enable_llm_query_expansion
        self.enable_cnki = enable_cnki
        self.scoring_mode = scoring_mode
        self.llm_rubric_max_papers = max(1, llm_rubric_max_papers)
        self.llm_rubric_min_rule_score = float(llm_rubric_min_rule_score)
        self.feedback_file = feedback_file
        self.embedding_model = embedding_model
        self.llm_candidate_mode = llm_candidate_mode
        self.llm_candidate_top_k = max(1, llm_candidate_top_k)
        self.min_llm_candidates = max(1, min_llm_candidates)
        self.selection_mode = selection_mode
        self.selection_top_k = max(1, selection_top_k)
        self.min_selected = max(1, min_selected)
        self.cards_from = cards_from
        self.top_k_cards = max(1, top_k_cards)
        self.card_threshold = float(card_threshold)
        self.download_top_k_if_none = max(1, download_top_k_if_none)
        self.enable_download_discovery = enable_download_discovery
        self.enable_crossref_page_scrape = enable_crossref_page_scrape
        self.cnki_headless = cnki_headless
        self.cnki_max_pages = cnki_max_pages
        self.cnki_limit = cnki_limit
        self.progress_enabled = progress_enabled
        self.progress_style = progress_style
        self.verbose = verbose
        self.quiet = quiet

        self.progress = ProgressManager(
            enabled=progress_enabled,
            style=progress_style,
            use_rich=progress_style == "rich",
            verbose=verbose,
            quiet=quiet,
        )

        self.openalex_client = OpenAlexClient(mailto=config.openalex_mailto, per_page=100)
        self.crossref_client = CrossrefClient(mailto=config.openalex_mailto, rows_per_page=100)
        self.arxiv_client = ArxivClient(max_results_per_query=80)
        self.semantic_client = SemanticScholarClient(api_key=config.semantic_scholar_api_key)
        self.paper_merger = PaperMerger(title_similarity_threshold=92)
        self.unpaywall_client = UnpaywallClient(email=config.unpaywall_email)

        self.journal_lookup = JournalRankLookup()
        self.journal_lookup.load()
        self.rule_scorer = RuleBasedScorer(
            scoring_config=config.scoring_config,
            journal_rank_lookup=self.journal_lookup,
        )

        # Query expansion LLM helper (existing simple scorer for compatibility)
        llm_api_key_for_expander = config.llm_api_key if enable_llm else None
        self.llm_query_scorer = LLMScorer(
            api_key=llm_api_key_for_expander,
            base_url=config.llm_base_url,
            model=config.llm_model,
            prompt_template_path=Path(__file__).resolve().parent / "prompts" / "llm_relevance_prompt.txt",
            connect_timeout=config.llm_connect_timeout,
            read_timeout=config.llm_read_timeout,
            max_retries=config.llm_max_retries,
            retry_backoff_seconds=config.llm_retry_backoff_seconds,
        )
        expander_llm = self.llm_query_scorer if self.enable_llm and self.enable_llm_query_expansion else None
        self.query_expander = QueryExpander(llm_scorer=expander_llm)

        self.cache_manager = CacheManager(self.output_dir / "cache")
        self.evidence_scorer = EvidenceBasedScorer(cache_manager=self.cache_manager, progress=self.progress)

        embed_cfg = config.scoring_config.get("embedding", {})
        if enable_embedding_score is None:
            if scoring_mode == "strict":
                resolved_embedding_enabled = True
            elif scoring_mode == "fast":
                resolved_embedding_enabled = False
            else:
                resolved_embedding_enabled = bool(embed_cfg.get("enabled", False))
        else:
            resolved_embedding_enabled = enable_embedding_score
        self.embedding_enabled = resolved_embedding_enabled
        self.embedding_scorer = EmbeddingScorer(
            model_name=embedding_model or str(embed_cfg.get("model_name", "")),
            enabled=self.embedding_enabled,
            cache_manager=self.cache_manager,
            progress=self.progress,
        )

        llm_rubric_enabled = self._llm_rubric_enabled()
        rubric_api_key = config.llm_api_key if llm_rubric_enabled else None
        self.llm_rubric_scorer = LLMRubricScorer(
            api_key=rubric_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
            prompt_template_path=Path(__file__).resolve().parent / "prompts" / "llm_rubric_scoring_prompt.txt",
            cache_manager=self.cache_manager,
            progress=self.progress,
            connect_timeout=config.llm_connect_timeout,
            read_timeout=config.llm_read_timeout,
            max_retries=config.llm_max_retries,
            retry_backoff_seconds=config.llm_retry_backoff_seconds,
        )
        self.feedback_manager = HumanFeedbackManager(feedback_file=feedback_file)
        self.aggregator = FinalScoreAggregator(
            weights=config.scoring_config.get("final_score_weights", {}),
            decision_thresholds=config.scoring_config.get("decision_thresholds", {}),
        )

        # ── Cookie bootstrap policy overrides ────────────────────────
        cookie_policy = load_cookie_bootstrap_policy()
        disabled_publisher_hosts = publisher_host_keywords(cookie_policy.disabled_publishers)
        enable_ezproxy = config.enable_ezproxy

        if cookie_policy.disable_inst_proxy:
            self.progress.log("[InitPolicy] InstProxy disabled by cookie bootstrap policy.")
        if cookie_policy.disable_ezproxy:
            self.progress.log("[InitPolicy] EZProxy disabled by cookie bootstrap policy.")
            enable_ezproxy = False

        # ── WebVPN / institutional proxy ─────────────────────────────
        inst_proxy_mode = config.inst_proxy_mode
        inst_proxy_url = config.inst_proxy_url
        inst_proxy_cookie_file = config.inst_proxy_cookie_file
        enable_inst_proxy = config.enable_inst_proxy
        if cookie_policy.disable_inst_proxy:
            enable_inst_proxy = False

        # If school is specified, look up host from database
        school_name = config.inst_proxy_school
        if school_name:
            from litbase_ai.download.school_db import get_host_for
            host = get_host_for(school_name)
            if host:
                inst_proxy_url = host
                inst_proxy_mode = "url_rewrite"
                self.progress.log(f"[InstProxy] School: {school_name} → {host}")
            else:
                self.progress.log(f"[InstProxy] School '{school_name}' not found in database")

        # WebVPN cookie — auto-enable if valid
        if (not cookie_policy.disable_inst_proxy) and config.webvpn_auto_login and config.webvpn_url:
            from litbase_ai.download.webvpn_login import WebVPNLoginClient

            wl = WebVPNLoginClient(vpn_url=config.webvpn_url)
            cookie_file = str(wl.cookie_file)

            ok, msg = wl.test_cookies()
            if ok:
                self.progress.log(f"[WebVPN] Using saved cookies ({msg})")
                enable_inst_proxy = True
                inst_proxy_mode = "url_rewrite"
                inst_proxy_url = config.webvpn_url
                inst_proxy_cookie_file = cookie_file
            else:
                self.progress.log(
                    f"[WebVPN] No valid cookies ({msg}). Login first:\n"
                    f"  python -m litbase_ai.download.webvpn_login --vpn-url {config.webvpn_url}"
                )
                # Try with school lookup
                if not school_name and not inst_proxy_url:
                    self.progress.log("[WebVPN] Set INST_PROXY_SCHOOL in your env file for school DB lookup")

        self.downloader = PDFDownloader(
            output_dir=output_dir,
            threshold=download_threshold,
            openalex_mailto=config.openalex_mailto,
            unpaywall_email=config.unpaywall_email,
            core_api_key=config.core_api_key,
            enable_discovery=enable_download_discovery,
            enable_crossref_page_scrape=enable_crossref_page_scrape,
            enable_scihub=config.enable_scihub,
            enable_libgen=config.enable_libgen,
            enable_arxiv_download=config.enable_arxiv_download,
            enable_ezproxy=enable_ezproxy,
            enable_inst_proxy=enable_inst_proxy,
            ezproxy_template=config.ezproxy_template,
            ezproxy_cookie_file=config.ezproxy_cookie_file,
            inst_proxy_mode=inst_proxy_mode,
            inst_proxy_url=inst_proxy_url,
            inst_proxy_cookie_file=inst_proxy_cookie_file,
            inst_proxy_school=school_name,
            inst_proxy_disabled_host_keywords=disabled_publisher_hosts,
            proxy=config.download_proxy,
            connect_timeout=config.download_connect_timeout,
            read_timeout=config.download_read_timeout,
            request_delay_min=config.download_request_delay_min,
            request_delay_max=config.download_request_delay_max,
        )
        self.exporter = PaperExporter(output_dir=output_dir)
        self.card_generator = LiteratureCardGenerator(output_dir=output_dir, threshold=0)

        self.diagnostics: dict[str, Any] = {}
        self._stage_counter = 0
        self._run_log_path: str | None = None
        self._no_results = False

    def run(self) -> None:
        raw_by_source: dict[str, list[PaperMetadata]] = {}
        merged_papers: list[PaperMetadata] = []
        enriched_papers: list[PaperMetadata] = []
        scored_papers: list[ScoredPaper] = []
        llm_candidates: list[ScoredPaper] = []
        selected_papers: list[ScoredPaper] = []
        downloaded_papers: list[ScoredPaper] = []
        cards_papers: list[ScoredPaper] = []
        expanded_query: ExpandedQuery | None = None

        try:
            self._stage("Initialize project", f"Output directory: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            log_file = LoggerFactory.setup(
                output_dir=self.output_dir,
                verbose=self.verbose,
                quiet=self.quiet,
                force=True,
            )
            if log_file:
                self._run_log_path = str(log_file)
                self.progress.log(f"run.log -> {log_file}")
            self._init_diagnostics()

            with self._stage_ctx("Load config"):
                self.progress.log(f"Config source: {self.config.env_source}")
                self.progress.log(f"Scoring mode: {self.scoring_mode}")
                self.progress.log(f"LLM rubric enabled: {self._llm_rubric_enabled()}")
                self.progress.log(
                    f"LLM timeout config: connect={self.config.llm_connect_timeout}s read={self.config.llm_read_timeout}s retries={self.config.llm_max_retries}"
                )
                self.progress.log(f"Search source workers: {self.config.search_source_workers}")
                self.progress.log(f"Embedding enabled: {self.embedding_enabled}")

            with self._stage_ctx("Expand query"):
                expanded_query = self.query_expander.expand(self.topic)
                self.exporter.export_expanded_query(expanded_query, progress=self.progress)
                self.diagnostics["expanded_query"] = {
                    "query_count": (
                        len(expanded_query.loose_queries)
                        + len(expanded_query.phrase_queries)
                        + len(expanded_query.boolean_queries)
                    ),
                    "english_keywords_count": len(expanded_query.english_keywords),
                    "chinese_keywords_count": len(expanded_query.chinese_keywords),
                }

            candidate_limit = max(self.limit, self.limit * self.candidate_multiplier)
            with self._stage_ctx("Search papers"):
                raw_by_source = self._search_sources(expanded_query=expanded_query, candidate_limit=candidate_limit)
                before = sum(len(v) for v in raw_by_source.values())
                self.diagnostics["search_before_dedup"] = before
                self.progress.log(
                    "Search recall summary: " + ", ".join(f"{name}={len(items)}" for name, items in raw_by_source.items())
                )
                if before == 0:
                    self._no_results = True
                    self.progress.log(
                        "All enabled data sources returned 0 papers. Exporting empty outputs with diagnostics.",
                        level="warning",
                    )

            with self._stage_ctx("Merge and deduplicate"):
                merged_papers = self.paper_merger.merge(list(raw_by_source.values()))
                if len(merged_papers) > candidate_limit:
                    merged_papers = merged_papers[:candidate_limit]
                before = int(self.diagnostics.get("search_before_dedup", 0))
                after = len(merged_papers)
                self.diagnostics["deduplication"] = {"before": before, "after": after, "removed": max(0, before - after)}
                if after == 0 and not self._no_results:
                    self._no_results = True
                    self.progress.log("No papers remained after deduplication.", level="warning")

            with self._stage_ctx("Export raw metadata"):
                self.exporter.export_raw_jsonl(merged_papers, progress=self.progress)
                self._safe_export_diagnostics()

            with self._stage_ctx("Enrich open-access links"):
                enriched_papers = self._enrich(merged_papers)
                self.diagnostics["unpaywall"] = dict(self.unpaywall_client.last_stats)

            with self._stage_ctx("Rule-based scoring"):
                scored_papers = self._score_rules(enriched_papers, expanded_query)
                self.diagnostics["scoring"] = self._rule_scoring_stats(scored_papers)

            with self._stage_ctx("Evidence extraction"):
                scored_papers = self.evidence_scorer.score_batch(scored_papers, self.topic, expanded_query=expanded_query)
                self._update_scoring_enhanced(
                    evidence_extracted=self.evidence_scorer.last_stats.get("evidence_extracted", 0),
                    evidence_missing_count=self.evidence_scorer.last_stats.get("evidence_missing_count", 0),
                )

            with self._stage_ctx("Optional embedding scoring"):
                if self.embedding_enabled and self.scoring_mode != "fast":
                    scored_papers = self.embedding_scorer.score_batch(scored_papers, self.topic, expanded_query=expanded_query)
                self._update_scoring_enhanced(
                    embedding_enabled=self.embedding_enabled and self.scoring_mode != "fast",
                    embedding_scored=self.embedding_scorer.last_stats.get("embedding_scored", 0),
                    embedding_failed=self.embedding_scorer.last_stats.get("embedding_failed", 0),
                )

            with self._stage_ctx("Select LLM candidates"):
                llm_candidates = self._select_llm_candidates(scored_papers)

            with self._stage_ctx("LLM rubric scoring"):
                if self._llm_rubric_enabled() and self.scoring_mode in {"balanced", "strict"} and llm_candidates:
                    self.llm_rubric_scorer.score_batch(
                        llm_candidates,
                        topic=self.topic,
                        expanded_query=expanded_query,
                        max_papers=self.llm_rubric_max_papers,
                    )
                else:
                    self.progress.log("LLM rubric scoring skipped by mode/key/candidate conditions.")
                self._update_scoring_enhanced(
                    llm_rubric_enabled=self._llm_rubric_enabled() and self.scoring_mode in {"balanced", "strict"},
                    llm_rubric_scored=self.llm_rubric_scorer.last_stats.get("llm_rubric_scored", 0),
                    llm_rubric_failed=self.llm_rubric_scorer.last_stats.get("llm_rubric_failed", 0),
                    llm_skipped=self.llm_rubric_scorer.last_stats.get("llm_skipped", 0),
                    llm_cache_hit=self.llm_rubric_scorer.last_stats.get("llm_cache_hit", 0),
                    llm_cache_miss=self.llm_rubric_scorer.last_stats.get("llm_cache_miss", 0),
                )

            with self._stage_ctx("Optional human feedback calibration"):
                if self.feedback_file and self.scoring_mode in {"balanced", "strict"}:
                    scored_papers = self.feedback_manager.apply_feedback(scored_papers)
                self._update_scoring_enhanced(
                    human_feedback_enabled=bool(self.feedback_file and Path(self.feedback_file).exists()),
                    feedback_applied_count=self.feedback_manager.last_stats.get("feedback_applied_count", 0),
                )

            with self._stage_ctx("Final score aggregation"):
                scored_papers = self.aggregator.aggregate_batch(scored_papers)
                for paper in scored_papers:
                    if paper.score.final_score is None:
                        paper.score.final_score = paper.score.rule_score
                scored_papers = sorted(scored_papers, key=lambda x: ((x.score.final_score or 0), (x.metadata.citation_count or 0)), reverse=True)
                self._update_scoring_enhanced(final_score_aggregated=self.aggregator.last_stats.get("final_score_aggregated", 0))
                self.exporter.export_scored_excel(scored_papers, progress=self.progress)

            with self._stage_ctx("Select papers"):
                selected_papers = self._select_papers(scored_papers)
                self.exporter.export_selected_excel(selected_papers, progress=self.progress)

            with self._stage_ctx("Download PDFs"):
                downloaded_papers = self._download_with_fallback(selected_papers)
                self.exporter.export_selected_excel(downloaded_papers, progress=self.progress)

            with self._stage_ctx("Generate literature cards"):
                cards_papers = self._select_cards_source(scored_papers, selected_papers, downloaded_papers)
                cards_papers = self.card_generator.generate_batch(cards_papers, progress=self.progress)
                self.diagnostics["cards"] = {
                    **self.diagnostics.get("cards", {}),
                    "cards_generated": len(cards_papers),
                    "index_path": self.card_generator.last_index_path,
                }

            with self._stage_ctx("Export cards table"):
                self.exporter.export_cards_excel(cards_papers, progress=self.progress)

            with self._stage_ctx("Re-export selected papers"):
                self.exporter.export_selected_excel(selected_papers, progress=self.progress)

            with self._stage_ctx("Generate summary report"):
                self.exporter.run_stats = {
                    "raw_count": len(merged_papers),
                    "rule_pass_count": self.diagnostics.get("scoring", {}).get("rule_passed", 0),
                    "selected_count": len(selected_papers),
                }
                self.exporter.export_summary_markdown(
                    selected_papers,
                    self.topic,
                    progress=self.progress,
                    scored_papers=scored_papers,
                    selected_papers=selected_papers,
                    cards_papers=cards_papers,
                    diagnostics=self.diagnostics,
                )
                self._safe_export_diagnostics()

            with self._stage_ctx("Export final results"):
                self.exporter.export_bibtex(selected_papers, progress=self.progress)
                self._safe_export_diagnostics()

            self._stage("Finish")
            self._finish_summary(merged_papers, llm_candidates, selected_papers, downloaded_papers, cards_papers)
        except Exception as exc:  # pragma: no cover
            logger.exception("Pipeline failed unexpectedly: %s", exc)
            self.progress.log(f"Pipeline error: {exc}", level="error")
            self._safe_export_diagnostics()
            raise
        finally:
            self.progress.close()

    def _search_sources(self, expanded_query: ExpandedQuery, candidate_limit: int) -> dict[str, list[PaperMetadata]]:
        source_limits = self._build_source_limits(candidate_limit)
        source_results: dict[str, list[PaperMetadata]] = {name: [] for name in ["OpenAlex", "Semantic Scholar", "Crossref", "arXiv", "CNKI"]}
        search_meta: dict[str, dict[str, Any]] = {}

        semantic_enabled, semantic_reason = self._semantic_enable_decision()
        cnki_client = (
            CNKIClient(
                headless=self.cnki_headless,
                timeout=30,
                max_pages=self.cnki_max_pages,
                allow_browser_automation=True,
            )
            if self.enable_cnki
            else None
        )
        specs: list[dict[str, Any]] = [
            {
                "label": "OpenAlex",
                "enabled": self.enable_openalex,
                "skip_reason": "disabled by CLI" if not self.enable_openalex else None,
                "search_fn": lambda: self.openalex_client.search_with_expanded_query(
                    expanded_query=expanded_query,
                    limit=source_limits["openalex"],
                    year_from=self.year_from,
                    progress=self.progress,
                ),
                "stats_supplier": lambda: self.openalex_client.last_search_stats,
                "status_supplier": lambda: (self.openalex_client.last_status, self.openalex_client.last_reason),
            },
            {
                "label": "Semantic Scholar",
                "enabled": semantic_enabled,
                "skip_reason": semantic_reason,
                "search_fn": lambda: self.semantic_client.search_with_expanded_query(
                    expanded_query=expanded_query,
                    limit=source_limits["semantic_scholar"],
                    year_from=self.year_from,
                    progress=self.progress,
                ),
                "stats_supplier": lambda: self.semantic_client.last_search_stats,
                "status_supplier": lambda: (self.semantic_client.last_status, self.semantic_client.last_reason),
            },
            {
                "label": "Crossref",
                "enabled": self.enable_crossref,
                "skip_reason": "disabled by CLI" if not self.enable_crossref else None,
                "search_fn": lambda: self.crossref_client.search_with_expanded_query(
                    expanded_query=expanded_query,
                    limit=source_limits["crossref"],
                    year_from=self.year_from,
                    progress=self.progress,
                ),
                "stats_supplier": lambda: self.crossref_client.last_search_stats,
                "status_supplier": lambda: (self.crossref_client.last_status, self.crossref_client.last_reason),
            },
            {
                "label": "arXiv",
                "enabled": self.enable_arxiv,
                "skip_reason": "disabled by CLI" if not self.enable_arxiv else None,
                "search_fn": lambda: self.arxiv_client.search_with_expanded_query(
                    expanded_query=expanded_query,
                    limit=source_limits["arxiv"],
                    year_from=self.year_from,
                    progress=self.progress,
                ),
                "stats_supplier": lambda: self.arxiv_client.last_search_stats,
                "status_supplier": lambda: (self.arxiv_client.last_status, self.arxiv_client.last_reason),
            },
            {
                "label": "CNKI",
                "enabled": self.enable_cnki,
                "skip_reason": None if self.enable_cnki else "disabled by CLI",
                "search_fn": (
                    lambda: cnki_client.search_with_expanded_query(
                        expanded_query=expanded_query,
                        limit=min(self.cnki_limit, source_limits["cnki"]),
                        year_from=self.year_from,
                        progress=self.progress,
                        output_dir=self.output_dir,
                    )
                )
                if cnki_client is not None
                else (lambda: []),
                "stats_supplier": (lambda: cnki_client.last_search_stats) if cnki_client is not None else (lambda: {}),
                "status_supplier": (
                    (lambda: (cnki_client.last_status, cnki_client.last_reason))
                    if cnki_client is not None
                    else (lambda: ("skipped", "disabled by CLI"))
                ),
            },
        ]

        workers = min(max(1, self.config.search_source_workers), len(specs))
        self.progress.log(f"Search dispatch: {len(specs)} sources, parallel workers={workers}")
        future_map: dict[Any, str] = {}
        results_by_label: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for spec in specs:
                future = pool.submit(
                    self._safe_search_source,
                    source_label=spec["label"],
                    enabled=spec["enabled"],
                    skip_reason=spec["skip_reason"],
                    search_fn=spec["search_fn"],
                    stats_supplier=spec["stats_supplier"],
                    status_supplier=spec["status_supplier"],
                )
                future_map[future] = str(spec["label"])

            self.progress.log("Search barrier: waiting for all source tasks to finish before LLM stage.")
            for future in as_completed(future_map):
                label = future_map[future]
                try:
                    results_by_label[label] = future.result()
                    self.progress.log(f"Search completed: {label}")
                except Exception as exc:  # pragma: no cover
                    self.progress.log(f"Search task crashed: {label} ({exc})", level="warning")
                    results_by_label[label] = {
                        "papers": [],
                        "meta": {
                            "enabled": True,
                            "available": False,
                            "attempted": True,
                            "status": "failed",
                            "queries": 0,
                            "returned": 0,
                            "failed_queries": 1,
                            "skipped_reason": str(exc),
                            "elapsed_seconds": 0.0,
                        },
                    }
        self.progress.log("Search barrier released: all source tasks finished.")

        for spec in specs:
            label = str(spec["label"])
            result = results_by_label.get(label) or {"papers": [], "meta": {}}
            source_results[label] = list(result.get("papers") or [])
            meta = dict(result.get("meta") or {})
            meta["execution_mode"] = "parallel"
            search_meta[label] = meta

        self.diagnostics["search_execution"] = {
            "mode": "parallel",
            "workers": workers,
            "sources_total": len(specs),
            "completed_sources": sum(1 for meta in search_meta.values() if meta.get("attempted") or meta.get("status") == "skipped"),
        }
        self.diagnostics["data_sources"] = search_meta
        self.diagnostics["search"] = search_meta
        return source_results

    def _safe_search_source(
        self,
        source_label: str,
        enabled: bool,
        skip_reason: str | None,
        search_fn: Callable[[], list[PaperMetadata]],
        stats_supplier: Callable[[], dict[str, Any]],
        status_supplier: Callable[[], tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        meta = {
            "enabled": enabled,
            "available": enabled,
            "attempted": False,
            "status": "skipped" if not enabled else "ok",
            "queries": 0,
            "returned": 0,
            "failed_queries": 0,
            "skipped_reason": skip_reason,
            "elapsed_seconds": 0.0,
        }
        if not enabled:
            self.progress.log(f"{source_label}: skipped" + (f" ({skip_reason})" if skip_reason else ""))
            return {"papers": [], "meta": meta}

        start = time.perf_counter()
        meta["attempted"] = True
        try:
            papers = search_fn()
            stats = stats_supplier() or {}
            status = "ok"
            reason = ""
            if status_supplier:
                try:
                    status, reason = status_supplier()
                except Exception:
                    status, reason = "ok", ""
            meta.update(
                {
                    "queries": int(stats.get("queries", 0)),
                    "returned": len(papers),
                    "failed_queries": int(stats.get("failed_queries", 0)),
                    "elapsed_seconds": round(float(stats.get("elapsed_seconds", time.perf_counter() - start)), 2),
                }
            )
            if "rate_limit_seconds" in stats:
                meta["rate_limit_seconds"] = float(stats.get("rate_limit_seconds", 0.0))
            if status in {"failed", "restricted", "skipped"}:
                meta["status"] = status
                meta["available"] = False
                meta["skipped_reason"] = reason or meta.get("skipped_reason")
            elif meta["failed_queries"] > 0 and len(papers) == 0:
                meta["status"] = "failed"
                meta["available"] = False
                if reason:
                    meta["skipped_reason"] = reason
            else:
                meta["status"] = "ok"
                meta["available"] = True
            self.progress.log(f"{source_label}: {len(papers)} papers")
            return {"papers": papers, "meta": meta}
        except Exception as exc:  # pragma: no cover
            meta["status"] = "failed"
            meta["available"] = False
            meta["failed_queries"] = max(1, int(meta.get("failed_queries", 0)))
            meta["skipped_reason"] = str(exc)
            meta["elapsed_seconds"] = round(time.perf_counter() - start, 2)
            self.progress.log(f"{source_label}: failed ({exc})", level="warning")
            return {"papers": [], "meta": meta}

    def _build_source_limits(self, candidate_limit: int) -> dict[str, int]:
        semantic_enabled, _ = self._semantic_enable_decision()
        if self.enable_cnki:
            base_weights = {"openalex": 0.35, "semantic_scholar": 0.25, "crossref": 0.15, "arxiv": 0.10, "cnki": 0.15}
        else:
            base_weights = {"openalex": 0.45, "semantic_scholar": 0.30, "crossref": 0.15, "arxiv": 0.10, "cnki": 0.0}
        enabled = {
            "openalex": self.enable_openalex,
            "semantic_scholar": semantic_enabled,
            "crossref": self.enable_crossref,
            "arxiv": self.enable_arxiv,
            "cnki": self.enable_cnki,
        }
        active_weight_sum = sum(weight for name, weight in base_weights.items() if enabled.get(name, False))
        if active_weight_sum <= 0:
            raise RuntimeError("No enabled search sources. Please enable at least one source.")

        limits: dict[str, int] = {}
        for name, weight in base_weights.items():
            if not enabled.get(name, False):
                limits[name] = 0
                continue
            normalized = weight / active_weight_sum
            base = int(candidate_limit * normalized)
            if name == "openalex":
                limits[name] = max(20, base)
            elif name == "semantic_scholar":
                limits[name] = max(15, base)
            elif name == "cnki":
                limits[name] = min(max(10, base), self.cnki_limit)
            else:
                limits[name] = max(10, base)
        return limits

    def _semantic_enable_decision(self) -> tuple[bool, str | None]:
        if self.enable_semantic_scholar is False:
            return False, "disabled by CLI"
        if self.enable_semantic_scholar is True and not self.config.semantic_scholar_api_key:
            return False, "missing SEMANTIC_SCHOLAR_API_KEY"
        if self.enable_semantic_scholar is None and not self.config.semantic_scholar_api_key:
            return False, "missing SEMANTIC_SCHOLAR_API_KEY"
        return True, None

    def _enrich(self, papers: list[PaperMetadata]) -> list[PaperMetadata]:
        return self.unpaywall_client.enrich(papers, progress=self.progress)

    def _score_rules(self, papers: list[PaperMetadata], expanded_query: ExpandedQuery) -> list[ScoredPaper]:
        scored: list[ScoredPaper] = []
        task_id = self.progress.task("Rule-based scoring", total=len(papers)) if papers else None
        for idx, paper in enumerate(papers):
            score = self.rule_scorer.score(paper, self.topic, expanded_query=expanded_query)
            score.final_score = score.rule_score
            scored.append(ScoredPaper(metadata=paper, score=score))
            if task_id is not None:
                self.progress.update(task_id, advance=1, description=f"Rule scoring {idx + 1}/{len(papers)}: {(paper.title or '')[:80]}")
        return scored

    def _rule_scoring_stats(self, papers: list[ScoredPaper]) -> dict[str, Any]:
        scores = [paper.score.rule_score for paper in papers]
        passed = [paper for paper in papers if paper.score.rule_score >= self.rule_threshold]
        return {
            "rule_scored": len(papers),
            "rule_threshold": self.rule_threshold,
            "rule_passed": len(passed),
            "rule_score_avg": round(mean(scores), 2) if scores else 0.0,
            "rule_score_max": round(max(scores), 2) if scores else 0.0,
        }

    def _select_llm_candidates(self, scored_papers: list[ScoredPaper]) -> list[ScoredPaper]:
        if not scored_papers:
            self.diagnostics["llm_candidate_selection"] = {
                "mode": self.llm_candidate_mode,
                "rule_threshold": self.rule_threshold,
                "threshold_candidates": 0,
                "fallback_topk_used": False,
                "llm_candidates": 0,
                "llm_candidate_top_k": self.llm_candidate_top_k,
                "min_llm_candidates": self.min_llm_candidates,
            }
            return []
        sorted_by_rule = sorted(scored_papers, key=lambda x: x.score.rule_score, reverse=True)
        threshold_base = max(self.rule_threshold, self.llm_rubric_min_rule_score)
        threshold_candidates = [paper for paper in sorted_by_rule if paper.score.rule_score >= threshold_base]
        fallback_used = False

        if self.llm_candidate_mode == "threshold":
            llm_candidates = threshold_candidates
        elif self.llm_candidate_mode == "topk":
            llm_candidates = sorted_by_rule[: self.llm_candidate_top_k]
        else:
            llm_candidates = list(threshold_candidates)
            if len(llm_candidates) < self.min_llm_candidates:
                fallback_used = True
                target = min(len(sorted_by_rule), max(self.min_llm_candidates, len(llm_candidates)))
                pool = sorted_by_rule[: max(self.llm_candidate_top_k, target)]
                seen = {id(p) for p in llm_candidates}
                for paper in pool:
                    if id(paper) not in seen:
                        llm_candidates.append(paper)
                        seen.add(id(paper))
                    if len(llm_candidates) >= target:
                        break
                if len(threshold_candidates) == 0:
                    self.progress.log(
                        f"Rule threshold produced 0 candidates. Falling back to top-{target} rule scored papers for LLM rubric scoring."
                    )

        self.diagnostics["llm_candidate_selection"] = {
            "mode": self.llm_candidate_mode,
            "rule_threshold": self.rule_threshold,
            "threshold_candidates": len(threshold_candidates),
            "fallback_topk_used": fallback_used,
            "llm_candidates": len(llm_candidates),
            "llm_candidate_top_k": self.llm_candidate_top_k,
            "min_llm_candidates": self.min_llm_candidates,
        }
        return llm_candidates

    def _select_papers(self, scored_papers: list[ScoredPaper]) -> list[ScoredPaper]:
        if not scored_papers:
            self.diagnostics["selection"] = {
                "mode": self.selection_mode,
                "llm_threshold": self.llm_threshold,
                "threshold_selected": 0,
                "fallback_topk_used": False,
                "selected_count": 0,
                "min_selected": self.min_selected,
                "selection_top_k": self.selection_top_k,
            }
            return []
        sorted_by_final = sorted(
            scored_papers,
            key=lambda x: ((x.score.final_score or 0), (x.metadata.citation_count or 0)),
            reverse=True,
        )
        threshold_selected = [paper for paper in sorted_by_final if (paper.score.final_score or 0) >= self.llm_threshold]
        fallback_used = False

        if self.selection_mode == "threshold":
            selected = threshold_selected
        elif self.selection_mode == "topk":
            selected = sorted_by_final[: self.selection_top_k]
        else:
            selected = list(threshold_selected)
            if len(selected) < self.min_selected:
                fallback_used = True
                pool = sorted_by_final[: max(self.selection_top_k, self.min_selected)]
                seen = {id(p) for p in selected}
                for paper in pool:
                    if id(paper) not in seen:
                        selected.append(paper)
                        seen.add(id(paper))
                    if len(selected) >= min(len(sorted_by_final), self.min_selected):
                        break
                if len(threshold_selected) == 0:
                    self.progress.log("Selection threshold produced 0 papers. Falling back to top scored papers.")
            selected = selected[: max(self.selection_top_k, self.min_selected)]

        selected = selected[: self.limit] if self.limit > 0 else selected
        self.diagnostics["selection"] = {
            "mode": self.selection_mode,
            "llm_threshold": self.llm_threshold,
            "threshold_selected": len(threshold_selected),
            "fallback_topk_used": fallback_used,
            "selected_count": len(selected),
            "min_selected": self.min_selected,
            "selection_top_k": self.selection_top_k,
        }
        return selected

    def _download_with_fallback(self, selected_papers: list[ScoredPaper]) -> list[ScoredPaper]:
        downloaded = self.downloader.download_batch(selected_papers, progress=self.progress)
        stats = dict(self.downloader.last_stats)
        stats["enabled"] = True
        stats["attempted"] = bool(selected_papers)

        fallback_used = False
        fallback_download_count = 0
        if stats.get("to_download", 0) == 0:
            with_pdf = [p for p in selected_papers if self.downloader.has_download_candidate(p)]
            if with_pdf:
                fallback_used = True
                self.progress.log(
                    f"No papers passed download_threshold, but {len(with_pdf)} selected papers have legal download candidates. "
                    f"Fallback downloading top-{self.download_top_k_if_none}."
                )
                fallback_candidates = sorted(
                    with_pdf,
                    key=lambda x: ((x.score.final_score or 0), (x.metadata.citation_count or 0)),
                    reverse=True,
                )[: self.download_top_k_if_none]
                self.downloader.download_batch(fallback_candidates, progress=self.progress, override_threshold=0)
                fb_stats = dict(self.downloader.last_stats)
                fallback_download_count = int(fb_stats.get("downloaded", 0) + fb_stats.get("already_exists", 0))
                for key in ("downloaded", "already_exists", "failed"):
                    stats[key] = int(stats.get(key, 0)) + int(fb_stats.get(key, 0))
                stats["to_download"] = int(stats.get("to_download", 0)) + int(fb_stats.get("to_download", 0))

        stats["download_fallback_used"] = fallback_used
        stats["download_top_k_if_none"] = self.download_top_k_if_none
        stats["fallback_download_count"] = fallback_download_count
        stats["download_proxy_set"] = bool(self.config.download_proxy)
        stats["source_scores"] = self.downloader.source_scorer.snapshot()
        self.diagnostics["download"] = stats
        return downloaded

    def _select_cards_source(
        self,
        scored_papers: list[ScoredPaper],
        selected_papers: list[ScoredPaper],
        downloaded_papers: list[ScoredPaper],
    ) -> list[ScoredPaper]:
        fallback_cards_used = False
        cards_source = self.cards_from

        scored_sorted = sorted(
            scored_papers,
            key=lambda x: ((x.score.final_score or x.score.rule_score), (x.metadata.citation_count or 0)),
            reverse=True,
        )
        selected_sorted = sorted(
            selected_papers,
            key=lambda x: ((x.score.final_score or x.score.rule_score), (x.metadata.citation_count or 0)),
            reverse=True,
        )
        downloaded_only = [p for p in downloaded_papers if p.metadata.raw.get("local_pdf_path")]

        if self.cards_from == "selected":
            cards = list(selected_sorted)
        elif self.cards_from == "scored":
            cards = [p for p in scored_sorted if (p.score.final_score or 0) >= self.card_threshold]
        elif self.cards_from == "downloaded":
            cards = list(downloaded_only)
        else:
            cards_source = "auto"
            if selected_sorted:
                cards = list(selected_sorted)
                if len(cards) < self.top_k_cards:
                    fallback_cards_used = True
                    selected_ids = {id(p) for p in cards}
                    for paper in scored_sorted:
                        if id(paper) in selected_ids:
                            continue
                        cards.append(paper)
                        selected_ids.add(id(paper))
                        if len(cards) >= self.top_k_cards:
                            break
            else:
                fallback_cards_used = True
                self.progress.log("selected_papers is empty, generating fallback cards from top scored papers.")
                cards = [p for p in scored_sorted if (p.score.final_score or 0) >= self.card_threshold]

        cards = cards[: self.top_k_cards]
        self.diagnostics["cards"] = {
            "cards_source": cards_source if not fallback_cards_used else "scored" if not selected_sorted else cards_source,
            "cards_generated": len(cards),
            "card_threshold": self.card_threshold,
            "top_k_cards": self.top_k_cards,
            "fallback_cards_used": fallback_cards_used,
        }
        return cards

    def _llm_rubric_enabled(self) -> bool:
        if not self.enable_llm:
            return False
        if self.scoring_mode == "fast":
            return False
        return bool(self.config.llm_api_key)

    def _init_diagnostics(self) -> None:
        self.diagnostics = {
            "topic": self.topic,
            "expanded_query": {},
            "data_sources": {},
            "search_execution": {},
            "deduplication": {},
            "unpaywall": {},
            "scoring": {},
            "scoring_enhanced": {
                "rule_scored": 0,
                "evidence_extracted": 0,
                "evidence_missing_count": 0,
                "embedding_enabled": False,
                "embedding_scored": 0,
                "embedding_failed": 0,
                "llm_rubric_enabled": False,
                "llm_rubric_scored": 0,
                "llm_rubric_failed": 0,
                "llm_skipped": 0,
                "llm_cache_hit": 0,
                "llm_cache_miss": 0,
                "llm_connect_timeout": self.config.llm_connect_timeout,
                "llm_read_timeout": self.config.llm_read_timeout,
                "llm_max_retries": self.config.llm_max_retries,
                "human_feedback_enabled": False,
                "feedback_applied_count": 0,
                "final_score_aggregated": 0,
            },
            "llm_candidate_selection": {},
            "selection": {},
            "cards": {},
            "download": {},
            "outputs": {},
        }

    def _update_scoring_enhanced(self, **kwargs) -> None:
        current = dict(self.diagnostics.get("scoring_enhanced", {}))
        for key, value in kwargs.items():
            current[key] = value
        current["rule_scored"] = int(self.diagnostics.get("scoring", {}).get("rule_scored", current.get("rule_scored", 0)))
        self.diagnostics["scoring_enhanced"] = current

    def _safe_export_diagnostics(self) -> None:
        existing_outputs = dict(self.diagnostics.get("outputs", {}))
        merged_outputs = {**existing_outputs, **self.exporter.last_outputs}
        if self._run_log_path:
            merged_outputs["run_log"] = self._run_log_path
        self.diagnostics["outputs"] = merged_outputs
        try:
            self.exporter.export_search_diagnostics(self.diagnostics, progress=self.progress)
            merged_outputs = {**self.diagnostics.get("outputs", {}), **self.exporter.last_outputs}
            if self._run_log_path:
                merged_outputs["run_log"] = self._run_log_path
            self.diagnostics["outputs"] = merged_outputs
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed exporting diagnostics: %s", exc)

    def _finish_summary(
        self,
        merged_papers: list[PaperMetadata],
        llm_candidates: list[ScoredPaper],
        selected_papers: list[ScoredPaper],
        downloaded_papers: list[ScoredPaper],
        cards_papers: list[ScoredPaper],
    ) -> None:
        downloaded_count = sum(1 for paper in downloaded_papers if paper.metadata.raw.get("local_pdf_path"))
        self.progress.log(f"Expanded query count: {self.diagnostics.get('expanded_query', {}).get('query_count', 0)}")
        source_counts = ", ".join(f"{k}: {v.get('returned', 0)}" for k, v in self.diagnostics.get("data_sources", {}).items())
        self.progress.log(f"Search source counts: {{ {source_counts} }}")
        self.progress.log(f"Deduped papers: {len(merged_papers)}")
        self.progress.log(f"LLM candidates: {len(llm_candidates)}")
        self.progress.log(f"Selected papers: {len(selected_papers)}")
        self.progress.log(f"Downloaded PDFs: {downloaded_count}")
        self.progress.log(f"Literature cards: {len(cards_papers)}")
        self.progress.log(f"Output directory: {self.output_dir}")
        self.progress.log("Output files:")
        if self._run_log_path:
            self.progress.log(f"  - run_log: {self._run_log_path}")
        for name, path in self.exporter.last_outputs.items():
            self.progress.log(f"  - {name}: {path}")

    def _stage(self, name: str, message: str | None = None) -> None:
        self._stage_counter += 1
        total = len(self.STAGE_NAMES)
        self.progress.log(f"[{self._stage_counter}/{total}] {name}" + (f"\n{message}" if message else ""))

    def _stage_ctx(self, name: str):
        self._stage(name)
        return self.progress.stage(name)
