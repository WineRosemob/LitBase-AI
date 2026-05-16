from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from litbase_ai.utils.logging import get_logger


logger = get_logger(__name__)


DEFAULT_SCORING_CONFIG: dict[str, Any] = {
    "weights": {
        "relevance": 0.40,
        "year": 0.15,
        "citation": 0.15,
        "journal": 0.15,
        "access": 0.10,
        "type": 0.05,
    },
    "year_score": {
        "recent_3_years": 100,
        "recent_5_years": 85,
        "recent_10_years": 65,
        "older": 35,
    },
    "citation_score": {"use_citation_per_year": True},
    "access_score": {
        "has_pdf": 100,
        "has_oa_landing_page": 70,
        "has_doi_only": 40,
        "no_access": 0,
    },
    "journal_score": {
        "Q1": 100,
        "Q2": 80,
        "Q3": 55,
        "Q4": 35,
        "unknown": 50,
    },
    "type_score": {
        "journal_article": 100,
        "proceedings_article": 80,
        "preprint": 75,
        "book_chapter": 60,
        "unknown": 50,
    },
    "final_score_weights": {
        "rule_score": 0.40,
        "llm_rubric_score": 0.45,
        "embedding_score": 0.10,
        "human_feedback_score": 0.05,
    },
    "llm_rubric": {
        "max_papers": 100,
        "min_rule_score": 40,
        "min_confidence": 50,
    },
    "embedding": {
        "enabled": False,
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    "decision_thresholds": {
        "core": 85,
        "important": 75,
        "background": 60,
        "peripheral": 40,
    },
    "fallback": {
        "llm_candidate_mode": "hybrid",
        "llm_candidate_top_k": 50,
        "min_llm_candidates": 20,
        "selection_mode": "hybrid",
        "selection_top_k": 30,
        "min_selected": 10,
        "cards_from": "auto",
        "top_k_cards": 30,
        "card_threshold": 0,
        "download_top_k_if_none": 5,
    },
}

PLACEHOLDER_VALUES = {
    "your_email@example.com",
    "your_llm_api_key_here",
    "your_api_key_here",
    "https://your-institution-proxy.example.com",
}


def _mask_secret(value: str | None) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}****{value[-4:]}"


def _is_placeholder_value(value: str) -> bool:
    return value.strip().lower() in PLACEHOLDER_VALUES


@dataclass
class AppConfig:
    """Application configuration loaded from env files and YAML."""

    openalex_mailto: str | None
    unpaywall_email: str | None
    core_api_key: str | None
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    semantic_scholar_api_key: str | None
    llm_connect_timeout: float
    llm_read_timeout: float
    llm_max_retries: int
    llm_retry_backoff_seconds: float
    download_proxy: str | None
    download_connect_timeout: float
    download_read_timeout: float
    download_request_delay_min: float
    download_request_delay_max: float
    enable_scihub: bool
    enable_libgen: bool
    enable_arxiv_download: bool
    enable_ezproxy: bool
    enable_inst_proxy: bool
    ezproxy_template: str | None
    ezproxy_cookie_file: str | None
    inst_proxy_mode: str
    inst_proxy_url: str | None
    inst_proxy_cookie_file: str | None
    inst_proxy_school: str | None
    webvpn_url: str | None
    webvpn_username: str | None
    webvpn_password: str | None
    webvpn_auto_login: bool
    search_source_workers: int
    scoring_config: dict[str, Any]
    scoring_config_path: Path
    env_source: str

    @classmethod
    def load(cls, env_file: str | Path | None = None) -> "AppConfig":
        package_dir = Path(__file__).resolve().parent
        project_dir = package_dir.parent
        scoring_path = package_dir / "config" / "scoring.yaml"
        scoring_config = cls.load_scoring_config(scoring_path)

        lookup_chain: list[dict[str, str]]
        env_source: str
        if env_file:
            file_path = Path(env_file).expanduser().resolve()
            file_values = cls._load_env_file(file_path)
            lookup_chain = [file_values, dict(os.environ)]
            env_source = str(file_path)
        else:
            env_values_dotenv = cls._load_env_file(project_dir / ".env")
            env_values_deepseek = cls._load_env_file(project_dir / ".env.deepseek")
            lookup_chain = [env_values_dotenv, env_values_deepseek, dict(os.environ)]
            env_source = "default(.env -> .env.deepseek [legacy] -> system env)"

        def pick(key: str, default: str | None = None) -> str | None:
            for source in lookup_chain:
                value = source.get(key)
                if value is None:
                    continue
                value_str = str(value).strip()
                if value_str:
                    if _is_placeholder_value(value_str):
                        continue
                    return value_str
            return default

        def pick_float(key: str, default: float) -> float:
            raw = pick(key)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                logger.warning("Invalid float for %s=%r, fallback to %s", key, raw, default)
                return default

        def pick_int(key: str, default: int) -> int:
            raw = pick(key)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                logger.warning("Invalid int for %s=%r, fallback to %s", key, raw, default)
                return default

        def pick_bool(key: str, default: bool) -> bool:
            raw = pick(key)
            if raw is None:
                return default
            val = str(raw).strip().lower()
            return val in ("1", "true", "yes", "on")

        llm_key = pick("LLM_API_KEY")
        if not llm_key:
            llm_key = pick("DEEPSEEK_API_KEY")

        config = cls(
            openalex_mailto=pick("OPENALEX_MAILTO"),
            unpaywall_email=pick("UNPAYWALL_EMAIL"),
            core_api_key=pick("CORE_API_KEY"),
            llm_api_key=llm_key,
            llm_base_url=pick("LLM_BASE_URL", "https://api.deepseek.com") or "https://api.deepseek.com",
            llm_model=pick("LLM_MODEL", "deepseek-chat") or "deepseek-chat",
            semantic_scholar_api_key=pick("SEMANTIC_SCHOLAR_API_KEY"),
            llm_connect_timeout=pick_float("LLM_CONNECT_TIMEOUT", 15.0),
            llm_read_timeout=pick_float("LLM_READ_TIMEOUT", 45.0),
            llm_max_retries=pick_int("LLM_MAX_RETRIES", 3),
            llm_retry_backoff_seconds=pick_float("LLM_RETRY_BACKOFF_SECONDS", 2.0),
            download_proxy=pick("DOWNLOAD_PROXY"),
            download_connect_timeout=pick_float("DOWNLOAD_CONNECT_TIMEOUT", 15.0),
            download_read_timeout=pick_float("DOWNLOAD_READ_TIMEOUT", 30.0),
            download_request_delay_min=pick_float("DOWNLOAD_REQUEST_DELAY_MIN", 0.0),
            download_request_delay_max=pick_float("DOWNLOAD_REQUEST_DELAY_MAX", 0.0),
            enable_scihub=pick_bool("ENABLE_SCIHUB", False),
            enable_libgen=pick_bool("ENABLE_LIBGEN", False),
            enable_arxiv_download=pick_bool("ENABLE_ARXIV_DOWNLOAD", True),
            enable_ezproxy=pick_bool("ENABLE_EZPROXY", False),
            enable_inst_proxy=pick_bool("ENABLE_INST_PROXY", False),
            ezproxy_template=pick("EZPROXY_TEMPLATE"),
            ezproxy_cookie_file=pick("EZPROXY_COOKIE_FILE"),
            inst_proxy_mode=pick("INST_PROXY_MODE", "http_proxy") or "http_proxy",
            inst_proxy_url=pick("INST_PROXY_URL"),
            inst_proxy_cookie_file=pick("INST_PROXY_COOKIE_FILE"),
            inst_proxy_school=pick("INST_PROXY_SCHOOL"),
            webvpn_url=pick("WEBVPN_URL"),
            webvpn_username=pick("WEBVPN_USERNAME"),
            webvpn_password=pick("WEBVPN_PASSWORD"),
            webvpn_auto_login=pick_bool("WEBVPN_AUTO_LOGIN", False),
            search_source_workers=pick_int("SEARCH_SOURCE_WORKERS", 4),
            scoring_config=scoring_config,
            scoring_config_path=scoring_path,
            env_source=env_source,
        )
        config.validate()
        return config

    @staticmethod
    def _load_env_file(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        try:
            values = dotenv_values(path)
            return {
                str(k): str(v).strip()
                for k, v in values.items()
                if k is not None and v is not None and str(v).strip()
            }
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed loading env file %s: %s", path, exc)
            return {}

    @staticmethod
    def load_scoring_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            logger.warning("scoring.yaml not found at %s; using default scoring config.", path)
            return DEFAULT_SCORING_CONFIG
        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
            merged = DEFAULT_SCORING_CONFIG.copy()
            for key, value in data.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
            return merged
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load scoring config, fallback to default: %s", exc)
            return DEFAULT_SCORING_CONFIG

    def validate(self) -> None:
        logger.info("Config source: %s", self.env_source)
        if not self.openalex_mailto:
            logger.warning("OPENALEX_MAILTO is empty. OpenAlex requests will still run.")
        if not self.unpaywall_email:
            logger.warning("UNPAYWALL_EMAIL is empty. Unpaywall enrichment will be skipped.")
        if self.core_api_key:
            logger.info("CORE_API_KEY loaded: %s", _mask_secret(self.core_api_key))
        if not self.llm_api_key:
            logger.warning("LLM_API_KEY/DEEPSEEK_API_KEY is empty. LLM scoring will be skipped.")
        else:
            logger.info("LLM key loaded: %s", _mask_secret(self.llm_api_key))
        if not self.semantic_scholar_api_key:
            logger.info("SEMANTIC_SCHOLAR_API_KEY is empty. Semantic Scholar source will be optional.")
        else:
            logger.info("Semantic Scholar key loaded: %s", _mask_secret(self.semantic_scholar_api_key))
        if self.llm_connect_timeout <= 0:
            logger.warning("LLM_CONNECT_TIMEOUT must be positive. Reset to 15.0 seconds.")
            self.llm_connect_timeout = 15.0
        if self.llm_read_timeout <= 0:
            logger.warning("LLM_READ_TIMEOUT must be positive. Reset to 45.0 seconds.")
            self.llm_read_timeout = 45.0
        if self.llm_max_retries < 1:
            logger.warning("LLM_MAX_RETRIES must be >= 1. Reset to 3.")
            self.llm_max_retries = 3
        if self.llm_retry_backoff_seconds < 0:
            logger.warning("LLM_RETRY_BACKOFF_SECONDS must be >= 0. Reset to 2.0 seconds.")
            self.llm_retry_backoff_seconds = 2.0
        if self.download_proxy:
            logger.info("DOWNLOAD_PROXY is set.")
        if self.download_connect_timeout <= 0:
            logger.warning("DOWNLOAD_CONNECT_TIMEOUT must be positive. Reset to 15.0 seconds.")
            self.download_connect_timeout = 15.0
        if self.download_read_timeout <= 0:
            logger.warning("DOWNLOAD_READ_TIMEOUT must be positive. Reset to 30.0 seconds.")
            self.download_read_timeout = 30.0
        if self.download_request_delay_min < 0:
            logger.warning("DOWNLOAD_REQUEST_DELAY_MIN must be >= 0. Reset to 0.")
            self.download_request_delay_min = 0.0
        if self.download_request_delay_max < self.download_request_delay_min:
            logger.warning(
                "DOWNLOAD_REQUEST_DELAY_MAX is smaller than DOWNLOAD_REQUEST_DELAY_MIN. Reset to match min delay."
            )
            self.download_request_delay_max = self.download_request_delay_min
        if self.search_source_workers < 1:
            logger.warning("SEARCH_SOURCE_WORKERS must be >= 1. Reset to 4.")
            self.search_source_workers = 4
        if self.enable_scihub:
            logger.warning("ENABLE_SCIHUB is enabled. Sci-Hub support is legacy/experimental and disabled by default.")
        if self.enable_libgen:
            logger.warning("ENABLE_LIBGEN is enabled. LibGen support is legacy/experimental and disabled by default.")
