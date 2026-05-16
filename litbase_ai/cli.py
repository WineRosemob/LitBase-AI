from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="litbase-ai",
        description="LitBase-AI CLI for literature retrieval, scoring, and compliant export workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search papers and build a literature workspace.")
    _add_search_args(search_parser)

    doctor_parser = subparsers.add_parser("doctor", help="Run environment and data source health checks.")
    doctor_parser.add_argument("--env-file", type=str, default=None, help="Optional .env file path.")
    doctor_parser.add_argument("--check-cnki", action="store_true", help="Include CNKI availability check.")
    doctor_parser.add_argument("--output-dir", type=str, default="outputs/doctor", help="Doctor output directory.")
    doctor_parser.add_argument("--no-progress", action="store_true", help="Disable dynamic progress bars.")
    doctor_parser.add_argument(
        "--progress-style",
        choices=["rich", "tqdm", "plain"],
        default="rich",
        help="Progress rendering style.",
    )
    doctor_parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    doctor_parser.add_argument("--quiet", action="store_true", help="Quiet mode (errors only in terminal).")

    init_parser = subparsers.add_parser("init", help="Bootstrap cookies and persist download policy.")
    init_parser.add_argument("--env-file", type=str, default=None, help="Optional .env file path.")
    init_parser.add_argument(
        "--publishers",
        type=str,
        default=None,
        help="Comma-separated publisher keys (default: elsevier,springer,wiley,nature,ieee).",
    )
    init_parser.add_argument("--cookie-dir", type=str, default=None, help="Override cookie directory path.")
    init_parser.add_argument("--policy-file", type=str, default=None, help="Override policy JSON file path.")
    init_parser.add_argument("--headless", action="store_true", help="Run Firefox in headless mode for login steps.")
    init_parser.add_argument("--non-interactive", action="store_true", help="No prompts; auto-disable missing cookie flows.")

    return parser.parse_args()


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topic", required=True, type=str, help="Research topic query string.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum papers to retrieve.")
    parser.add_argument("--year-from", type=int, default=None, help="Filter papers from given year.")
    parser.add_argument("--rule-threshold", type=float, default=60, help="Rule score threshold.")
    parser.add_argument("--llm-threshold", type=float, default=70, help="Final score threshold.")
    parser.add_argument("--download-threshold", type=float, default=75, help="PDF download threshold.")
    parser.add_argument("--candidate-multiplier", type=int, default=3, help="Wide-recall candidate multiplier.")
    parser.add_argument(
        "--disable-download-discovery",
        action="store_true",
        help="Disable extra OA source discovery during PDF download.",
    )
    parser.add_argument(
        "--disable-crossref-page-scrape",
        action="store_true",
        help="Disable DOI landing-page scraping when resolving PDF links.",
    )

    parser.add_argument("--disable-openalex", action="store_true", help="Disable OpenAlex source.")
    parser.add_argument("--disable-crossref", action="store_true", help="Disable Crossref source.")
    parser.add_argument("--disable-arxiv", action="store_true", help="Disable arXiv source.")

    semantic_group = parser.add_mutually_exclusive_group()
    semantic_group.add_argument(
        "--enable-semantic-scholar",
        action="store_true",
        help="Enable Semantic Scholar source (requires API key).",
    )
    semantic_group.add_argument(
        "--disable-semantic-scholar",
        action="store_true",
        help="Force disable Semantic Scholar source.",
    )

    cnki_group = parser.add_mutually_exclusive_group()
    cnki_group.add_argument("--enable-cnki", action="store_true", help="Enable optional CNKI metadata source.")
    cnki_group.add_argument("--disable-cnki", action="store_true", help="Force disable CNKI source.")

    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM scoring and LLM query expansion.")
    parser.add_argument(
        "--scoring-mode",
        choices=["fast", "balanced", "strict"],
        default="balanced",
        help="Scoring mode preset.",
    )
    parser.add_argument("--llm-rubric-max-papers", type=int, default=100, help="Max papers for LLM rubric scoring.")
    parser.add_argument(
        "--llm-rubric-min-rule-score",
        type=float,
        default=40,
        help="Minimum rule score for rubric threshold mode.",
    )
    parser.add_argument("--feedback-file", type=str, default=None, help="Optional feedback.csv path.")
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument("--enable-embedding-score", action="store_true", help="Enable optional embedding scoring.")
    embedding_group.add_argument("--disable-embedding-score", action="store_true", help="Disable optional embedding scoring.")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model name for embedding scoring.",
    )
    parser.add_argument(
        "--llm-candidate-mode",
        choices=["threshold", "topk", "hybrid"],
        default="hybrid",
        help="How to select LLM rubric candidates.",
    )
    parser.add_argument("--llm-candidate-top-k", type=int, default=50, help="Top-k for LLM candidates.")
    parser.add_argument("--min-llm-candidates", type=int, default=20, help="Minimum LLM candidate count in hybrid mode.")
    parser.add_argument(
        "--selection-mode",
        choices=["threshold", "topk", "hybrid"],
        default="hybrid",
        help="Final selected paper strategy.",
    )
    parser.add_argument("--selection-top-k", type=int, default=30, help="Top-k for selected papers.")
    parser.add_argument("--min-selected", type=int, default=10, help="Minimum selected papers in hybrid mode.")
    parser.add_argument(
        "--cards-from",
        choices=["selected", "scored", "downloaded", "auto"],
        default="auto",
        help="Literature card source.",
    )
    parser.add_argument("--top-k-cards", type=int, default=30, help="Maximum number of literature cards.")
    parser.add_argument("--card-threshold", type=float, default=0, help="Minimum score for cards-from scored mode.")
    parser.add_argument("--download-top-k-if-none", type=int, default=5, help="Fallback top-k download candidates if threshold yields none.")
    parser.add_argument(
        "--disable-llm-query-expansion",
        action="store_true",
        help="Disable LLM-assisted query expansion only.",
    )
    parser.add_argument(
        "--cnki-headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run CNKI browser in headless mode when CNKI is enabled.",
    )
    parser.add_argument("--cnki-max-pages", type=int, default=5, help="Maximum CNKI pages per query.")
    parser.add_argument("--cnki-limit", type=int, default=200, help="Maximum CNKI candidates.")

    parser.add_argument("--no-progress", action="store_true", help="Disable dynamic progress bars.")
    parser.add_argument(
        "--progress-style",
        choices=["rich", "tqdm", "plain"],
        default="rich",
        help="Progress rendering style.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (errors only in terminal).")
    parser.add_argument("--output-dir", type=str, default="outputs/demo", help="Output directory path.")
    parser.add_argument("--env-file", type=str, default=None, help="Optional .env file path.")


def _run_doctor(args: argparse.Namespace) -> None:
    from litbase_ai.config import AppConfig
    from litbase_ai.utils.healthcheck import (
        DataSourceHealthChecker,
        DownloadHealthChecker,
        dependency_report,
        runtime_report,
    )
    from litbase_ai.utils.logging import LoggerFactory
    from litbase_ai.utils.progress import ProgressManager

    output_dir = Path(args.output_dir)
    LoggerFactory.setup(output_dir=output_dir, verbose=args.verbose, quiet=args.quiet, force=True)
    progress = ProgressManager(
        enabled=not args.no_progress,
        style=args.progress_style,
        use_rich=args.progress_style == "rich",
        verbose=args.verbose,
        quiet=args.quiet,
    )

    try:
        config = AppConfig.load(env_file=args.env_file)
        checker = DataSourceHealthChecker(config=config, progress=progress, check_cnki=args.check_cnki)
        download_checker = DownloadHealthChecker(output_dir=output_dir, config=config, progress=progress)

        progress.log("[doctor] Running runtime checks ...")
        report: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime_report(),
            "dependency": dependency_report(),
            "config": {
                "env_source": config.env_source,
                "openalex_mailto_set": bool(config.openalex_mailto),
                "unpaywall_email_set": bool(config.unpaywall_email),
                "core_api_key_set": bool(config.core_api_key),
                "llm_key_set": bool(config.llm_api_key),
                "llm_base_url": config.llm_base_url,
                "llm_model": config.llm_model,
                "llm_connect_timeout": config.llm_connect_timeout,
                "llm_read_timeout": config.llm_read_timeout,
                "llm_max_retries": config.llm_max_retries,
                "llm_retry_backoff_seconds": config.llm_retry_backoff_seconds,
                "semantic_scholar_key_set": bool(config.semantic_scholar_api_key),
                "download_proxy_set": bool(config.download_proxy),
                "download_connect_timeout": config.download_connect_timeout,
                "download_read_timeout": config.download_read_timeout,
                "search_source_workers": config.search_source_workers,
                "enable_scihub": config.enable_scihub,
                "enable_libgen": config.enable_libgen,
                "enable_arxiv_download": config.enable_arxiv_download,
                "enable_ezproxy": config.enable_ezproxy,
                "enable_inst_proxy": config.enable_inst_proxy,
            },
        }

        progress.log("[doctor] Checking data sources ...")
        report["data_sources"] = checker.check_all()

        progress.log("[doctor] Checking download capability ...")
        report["download"] = {
            "writable_pdf_dir": download_checker.check_writable_pdf_dir(),
            "arxiv_pdf_test": download_checker.test_arxiv_download(),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "doctor_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        progress.log(f"[doctor] Report written to {report_path}")
    finally:
        progress.close()


def _run_init(args: argparse.Namespace) -> None:
    from litbase_ai.config import AppConfig
    from litbase_ai.download.cookie_bootstrap import CookieBootstrapper, parse_publishers_arg

    config = AppConfig.load(env_file=args.env_file)
    publishers = parse_publishers_arg(args.publishers)
    bootstrapper = CookieBootstrapper(
        config=config,
        publishers=publishers,
        cookie_dir=args.cookie_dir,
        policy_path=args.policy_file,
        headless=args.headless,
    )
    report = bootstrapper.run(interactive=not args.non_interactive)

    policy = report.get("policy") or {}
    print("Cookie bootstrap completed.")
    print(f"- policy_file: {report.get('policy_path')}")
    print(f"- disable_inst_proxy: {policy.get('disable_inst_proxy')}")
    print(f"- disable_ezproxy: {policy.get('disable_ezproxy')}")
    print(f"- disabled_publishers: {', '.join(policy.get('disabled_publishers') or []) or '(none)'}")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    if args.command == "doctor":
        _run_doctor(args)
        return
    if args.command == "init":
        _run_init(args)
        return

    from litbase_ai.config import AppConfig
    from litbase_ai.pipeline import LitBasePipeline
    from litbase_ai.utils.logging import LoggerFactory

    LoggerFactory.setup(verbose=args.verbose, quiet=args.quiet, force=True)
    config = AppConfig.load(env_file=args.env_file)

    enable_semantic: bool | None = None
    if args.enable_semantic_scholar:
        enable_semantic = True
    elif args.disable_semantic_scholar:
        enable_semantic = False

    if args.enable_cnki:
        enable_cnki = True
    elif args.disable_cnki:
        enable_cnki = False
    else:
        enable_cnki = False

    if args.enable_embedding_score:
        enable_embedding = True
    elif args.disable_embedding_score:
        enable_embedding = False
    else:
        enable_embedding = None

    pipeline = LitBasePipeline(
        config=config,
        output_dir=Path(args.output_dir),
        topic=args.topic,
        limit=args.limit,
        year_from=args.year_from,
        rule_threshold=args.rule_threshold,
        llm_threshold=args.llm_threshold,
        download_threshold=args.download_threshold,
        candidate_multiplier=args.candidate_multiplier,
        enable_openalex=not args.disable_openalex,
        enable_crossref=not args.disable_crossref,
        enable_arxiv=not args.disable_arxiv,
        enable_semantic_scholar=enable_semantic,
        enable_llm=not args.disable_llm,
        enable_llm_query_expansion=not args.disable_llm_query_expansion and not args.disable_llm,
        enable_cnki=enable_cnki,
        scoring_mode=args.scoring_mode,
        llm_rubric_max_papers=args.llm_rubric_max_papers,
        llm_rubric_min_rule_score=args.llm_rubric_min_rule_score,
        feedback_file=Path(args.feedback_file) if args.feedback_file else None,
        enable_embedding_score=enable_embedding,
        embedding_model=args.embedding_model,
        llm_candidate_mode=args.llm_candidate_mode,
        llm_candidate_top_k=args.llm_candidate_top_k,
        min_llm_candidates=args.min_llm_candidates,
        selection_mode=args.selection_mode,
        selection_top_k=args.selection_top_k,
        min_selected=args.min_selected,
        cards_from=args.cards_from,
        top_k_cards=args.top_k_cards,
        card_threshold=args.card_threshold,
        download_top_k_if_none=args.download_top_k_if_none,
        enable_download_discovery=not args.disable_download_discovery,
        enable_crossref_page_scrape=not args.disable_crossref_page_scrape,
        cnki_headless=args.cnki_headless,
        cnki_max_pages=args.cnki_max_pages,
        cnki_limit=args.cnki_limit,
        progress_enabled=not args.no_progress,
        progress_style=args.progress_style,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
