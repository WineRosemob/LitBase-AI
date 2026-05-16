from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from litbase_ai.config import AppConfig
from litbase_ai.utils.logging import get_logger


logger = get_logger(__name__)


class DataSourceHealthChecker:
    """Lightweight health checks for each data source and API integration."""

    def __init__(self, config: AppConfig, progress=None, check_cnki: bool = False):
        self.config = config
        self.progress = progress
        self.check_cnki_enabled = check_cnki
        self.timeout = httpx.Timeout(
            connect=max(5.0, float(config.download_connect_timeout)),
            read=max(10.0, float(config.download_read_timeout)),
            write=max(10.0, float(config.download_read_timeout)),
            pool=max(5.0, float(config.download_connect_timeout)),
        )

    def check_all(self) -> dict[str, dict[str, Any]]:
        checks = {
            "OpenAlex": self.check_openalex(),
            "Semantic Scholar": self.check_semantic_scholar(),
            "Crossref": self.check_crossref(),
            "arXiv": self.check_arxiv(),
            "Unpaywall": self.check_unpaywall(),
            "OpenAIRE": self.check_openaire(),
            "DOAJ": self.check_doaj(),
            "Europe PMC": self.check_europepmc(),
            "PMC": self.check_pmc(),
            "CORE": self.check_core(),
            "DeepSeek": self.check_deepseek(),
            "CNKI": self.check_cnki() if self.check_cnki_enabled else self._skipped("CNKI", "CNKI check not requested"),
        }
        return checks

    def _client(self, headers: dict[str, str] | None = None) -> httpx.Client:
        merged_headers = {"User-Agent": "LitBase-AI/doctor"}
        if headers:
            merged_headers.update(headers)
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "headers": merged_headers,
            "follow_redirects": True,
            "trust_env": not bool(self.config.download_proxy),
        }
        if self.config.download_proxy:
            kwargs["proxy"] = self.config.download_proxy
        return httpx.Client(**kwargs)

    def check_openalex(self) -> dict[str, Any]:
        start = time.perf_counter()
        params = {"search": "carbon neutrality", "per-page": 1}
        if self.config.openalex_mailto:
            params["mailto"] = self.config.openalex_mailto
        try:
            with self._client() as client:
                response = client.get("https://api.openalex.org/works", params=params)
            response.raise_for_status()
            payload = response.json()
            ok = isinstance(payload, dict) and isinstance(payload.get("results"), list)
            return self._result("OpenAlex", True, ok, "ok" if ok else "failed", "" if ok else "Unexpected payload", start)
        except Exception as exc:  # pragma: no cover
            return self._result("OpenAlex", True, False, "failed", str(exc), start)

    def check_semantic_scholar(self) -> dict[str, Any]:
        start = time.perf_counter()
        if not self.config.semantic_scholar_api_key:
            return self._skipped("Semantic Scholar", "missing SEMANTIC_SCHOLAR_API_KEY", start=start)
        headers = {
            "x-api-key": self.config.semantic_scholar_api_key,
            "Accept": "application/json",
        }
        params = {"query": "carbon neutrality", "limit": 1, "fields": "title"}
        try:
            with self._client(headers=headers) as client:
                response = client.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params)
            if response.status_code == 401:
                return self._result("Semantic Scholar", True, False, "failed", "401 unauthorized", start)
            if response.status_code == 403:
                return self._result("Semantic Scholar", True, False, "failed", "403 forbidden", start)
            if response.status_code == 429:
                return self._result("Semantic Scholar", True, False, "failed", "429 rate limited", start)
            response.raise_for_status()
            payload = response.json()
            ok = isinstance(payload, dict) and isinstance(payload.get("data"), list)
            return self._result("Semantic Scholar", True, ok, "ok" if ok else "failed", "" if ok else "Unexpected payload", start)
        except Exception as exc:  # pragma: no cover
            return self._result("Semantic Scholar", True, False, "failed", str(exc), start)

    def check_crossref(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            with self._client() as client:
                response = client.get("https://api.crossref.org/works", params={"query": "carbon neutrality", "rows": 1})
            response.raise_for_status()
            payload = response.json()
            ok = isinstance(payload, dict) and isinstance((payload.get("message") or {}).get("items"), list)
            return self._result("Crossref", True, ok, "ok" if ok else "failed", "" if ok else "Unexpected payload", start)
        except Exception as exc:  # pragma: no cover
            return self._result("Crossref", True, False, "failed", str(exc), start)

    def check_arxiv(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            with self._client() as client:
                response = client.get(
                    "https://export.arxiv.org/api/query",
                    params={"search_query": "all:machine learning climate change", "start": 0, "max_results": 1},
                )
            response.raise_for_status()
            text = response.text
            ok = "<entry>" in text or "entry>" in text
            return self._result("arXiv", True, ok, "ok" if ok else "failed", "" if ok else "No entry found", start)
        except Exception as exc:  # pragma: no cover
            return self._result("arXiv", True, False, "failed", str(exc), start)

    def check_unpaywall(self) -> dict[str, Any]:
        start = time.perf_counter()
        if not self.config.unpaywall_email:
            return self._skipped("Unpaywall", "missing UNPAYWALL_EMAIL", start=start)
        doi = "10.1038/nature12373"
        try:
            with self._client() as client:
                response = client.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": self.config.unpaywall_email})
            response.raise_for_status()
            payload = response.json()
            ok = isinstance(payload, dict) and "is_oa" in payload
            return self._result("Unpaywall", True, ok, "ok" if ok else "failed", "" if ok else "Unexpected payload", start)
        except Exception as exc:  # pragma: no cover
            return self._result("Unpaywall", True, False, "failed", str(exc), start)

    def check_openaire(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            with self._client() as client:
                response = client.get(
                    "https://api.openaire.eu/search/publications",
                    params={"doi": "10.1038/nature12373", "format": "json"},
                )
            response.raise_for_status()
            payload = response.json()
            ok = isinstance(payload, dict) and isinstance(payload.get("response"), dict)
            return self._result("OpenAIRE", True, ok, "ok" if ok else "failed", "" if ok else "Unexpected payload", start)
        except Exception as exc:  # pragma: no cover
            return self._result("OpenAIRE", True, False, "failed", str(exc), start)

    def check_doaj(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            with self._client() as client:
                response = client.get("https://doaj.org/api/v2/search/articles/doi:10.1038/nature12373")
            response.raise_for_status()
            payload = response.json()
            ok = isinstance(payload, dict) and isinstance(payload.get("results"), list)
            return self._result("DOAJ", True, ok, "ok" if ok else "failed", "" if ok else "Unexpected payload", start)
        except Exception as exc:  # pragma: no cover
            return self._result("DOAJ", True, False, "failed", str(exc), start)

    def check_europepmc(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            with self._client() as client:
                response = client.get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    params={"query": "DOI:10.1038/nature12373", "format": "json", "pageSize": 1},
                )
            response.raise_for_status()
            payload = response.json()
            ok = isinstance(payload, dict) and isinstance((payload.get("resultList") or {}).get("result"), list)
            return self._result("Europe PMC", True, ok, "ok" if ok else "failed", "" if ok else "Unexpected payload", start)
        except Exception as exc:  # pragma: no cover
            return self._result("Europe PMC", True, False, "failed", str(exc), start)

    def check_pmc(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            with self._client() as client:
                response = client.get(
                    "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                    params={"ids": "10.1038/nature12373", "format": "json"},
                )
            response.raise_for_status()
            payload = response.json()
            ok = isinstance(payload, dict) and isinstance(payload.get("records"), list)
            return self._result("PMC", True, ok, "ok" if ok else "failed", "" if ok else "Unexpected payload", start)
        except Exception as exc:  # pragma: no cover
            return self._result("PMC", True, False, "failed", str(exc), start)

    def check_core(self) -> dict[str, Any]:
        start = time.perf_counter()
        headers = {"Accept": "application/json"}
        if self.config.core_api_key:
            headers["Authorization"] = f"Bearer {self.config.core_api_key}"
        try:
            with self._client(headers=headers) as client:
                response = client.get(
                    "https://api.core.ac.uk/v3/search/works",
                    params={"q": 'doi:"10.1038/nature12373"', "limit": 1},
                )
            response.raise_for_status()
            payload = response.json()
            ok = isinstance(payload, dict) and any(key in payload for key in ("results", "totalHits", "data"))
            return self._result("CORE", True, ok, "ok" if ok else "failed", "" if ok else "Unexpected payload", start)
        except Exception as exc:  # pragma: no cover
            return self._result("CORE", True, False, "failed", str(exc), start)

    def check_deepseek(self) -> dict[str, Any]:
        start = time.perf_counter()
        if not self.config.llm_api_key:
            return self._skipped("DeepSeek", "missing LLM_API_KEY/DEEPSEEK_API_KEY", start=start)
        url = f"{self.config.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.llm_model,
            "messages": [{"role": "user", "content": "Reply with JSON only: {\"ok\":true}"}],
            "temperature": 0,
            "max_tokens": 32,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json",
        }
        try:
            with self._client(headers=headers) as client:
                response = client.post(url, json=payload)
            if response.status_code in (401, 403):
                return self._result("DeepSeek", True, False, "failed", f"{response.status_code} auth failed", start)
            response.raise_for_status()
            data = response.json()
            ok = isinstance(data, dict) and bool(data.get("choices"))
            return self._result("DeepSeek", True, ok, "ok" if ok else "failed", "" if ok else "Unexpected payload", start)
        except Exception as exc:  # pragma: no cover
            return self._result("DeepSeek", True, False, "failed", str(exc), start)

    def check_cnki(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception:
            return self._result("CNKI", True, False, "skipped", "Playwright not installed", start)

        try:
            with sync_playwright() as p:
                browser = p.firefox.launch(headless=True)
                page = browser.new_page()
                page.goto("https://kns.cnki.net/kns8s/basicSearch", timeout=15000)
                title = page.title() or ""
                browser.close()
            lower_title = title.lower()
            if any(x in title for x in ("登录", "验证码", "访问受限")):
                return self._result("CNKI", True, False, "restricted", f"page title: {title}", start)
            if any(x in lower_title for x in ("forbidden", "captcha", "blocked")):
                return self._result("CNKI", True, False, "restricted", f"page title: {title}", start)
            return self._result("CNKI", True, True, "ok", "", start)
        except Exception as exc:  # pragma: no cover
            reason = str(exc)
            if "Executable doesn't exist" in reason or "playwright install" in reason.lower():
                return self._result("CNKI", True, False, "skipped", "Playwright browser executable is missing", start)
            return self._result("CNKI", True, False, "failed", reason, start)

    def _result(
        self,
        name: str,
        enabled: bool,
        available: bool,
        status: str,
        reason: str,
        start: float | None = None,
    ) -> dict[str, Any]:
        elapsed_seconds = round(time.perf_counter() - (start or time.perf_counter()), 3)
        result = {
            "name": name,
            "enabled": enabled,
            "available": available,
            "status": status,
            "reason": reason,
            "elapsed_seconds": elapsed_seconds,
        }
        if self.progress:
            self.progress.log(f"[doctor] {name}: {status}" + (f" ({reason})" if reason else ""))
        return result

    def _skipped(self, name: str, reason: str, start: float | None = None) -> dict[str, Any]:
        return self._result(name=name, enabled=False, available=False, status="skipped", reason=reason, start=start)


class DownloadHealthChecker:
    """Health checks for download readiness and legal OA PDF download capability."""

    TEST_ARXIV_PDF = "https://arxiv.org/pdf/1706.03762.pdf"

    def __init__(self, output_dir: Path, config: AppConfig, progress=None):
        self.output_dir = output_dir
        self.config = config
        self.progress = progress
        self.timeout = httpx.Timeout(
            connect=max(5.0, float(config.download_connect_timeout)),
            read=max(20.0, float(config.download_read_timeout)),
            write=max(20.0, float(config.download_read_timeout)),
            pool=max(5.0, float(config.download_connect_timeout)),
        )

    def _client(self, headers: dict[str, str] | None = None) -> httpx.Client:
        merged_headers = {"User-Agent": "LitBase-AI/doctor"}
        if headers:
            merged_headers.update(headers)
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "follow_redirects": True,
            "headers": merged_headers,
            "trust_env": not bool(self.config.download_proxy),
        }
        if self.config.download_proxy:
            kwargs["proxy"] = self.config.download_proxy
        return httpx.Client(**kwargs)

    def check_writable_pdf_dir(self) -> dict[str, Any]:
        start = time.perf_counter()
        target = self.output_dir / "pdf_test"
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return {
                "status": "ok",
                "path": str(target),
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            }
        except Exception as exc:  # pragma: no cover
            return {
                "status": "failed",
                "path": str(target),
                "reason": str(exc),
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            }

    def test_arxiv_download(self) -> dict[str, Any]:
        return self.test_direct_pdf_download(self.TEST_ARXIV_PDF)

    def test_direct_pdf_download(self, pdf_url: str) -> dict[str, Any]:
        start = time.perf_counter()
        target_dir = self.output_dir / "pdf_test"
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / "doctor_test.pdf"
        headers = {"Accept": "application/pdf,*/*"}
        try:
            with self._client(headers=headers) as client:
                response = client.get(pdf_url)
            response.raise_for_status()
            content_type = (response.headers.get("Content-Type") or "").lower()
            content = response.content
            file_path.write_bytes(content)
            is_pdf_header = content.startswith(b"%PDF")
            ok = file_path.exists() and file_path.stat().st_size > 0 and (is_pdf_header or "application/pdf" in content_type)
            result = {
                "attempted": True,
                "status": "ok" if ok else "failed",
                "file": str(file_path),
                "bytes": file_path.stat().st_size if file_path.exists() else 0,
                "content_type": content_type,
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            }
            if not ok:
                result["reason"] = "Downloaded content is not recognized as PDF."
            return result
        except Exception as exc:  # pragma: no cover
            return {
                "attempted": True,
                "status": "failed",
                "file": str(file_path),
                "reason": str(exc),
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            }


def dependency_report() -> dict[str, bool]:
    """Check optional/runtime dependencies are importable."""
    import importlib.util

    packages = [
        "httpx",
        "requests",
        "pandas",
        "openpyxl",
        "pydantic",
        "dotenv",
        "yaml",
        "tqdm",
        "rich",
        "tenacity",
        "rapidfuzz",
        "bibtexparser",
        "feedparser",
        "bs4",
        "lxml",
        "playwright",
        "sentence_transformers",
        "Crypto",
        "pytest",
    ]
    report: dict[str, bool] = {}
    for name in packages:
        report[name] = importlib.util.find_spec(name) is not None
    return report


def runtime_report() -> dict[str, Any]:
    return {
        "python": os.sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cwd": str(Path.cwd()),
    }
