from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from litbase_ai.models import ExpandedQuery, PaperMetadata
from litbase_ai.query.expander import QueryExpander
from litbase_ai.search.base import BaseSearchClient
from litbase_ai.utils.logging import get_logger


logger = get_logger(__name__)


try:  # pragma: no cover
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover
    PlaywrightTimeoutError = Exception
    sync_playwright = None

try:  # pragma: no cover
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None


class CNKIClient(BaseSearchClient):
    """Optional CNKI metadata retriever under strict legal constraints."""

    SEARCH_URL = "https://kns.cnki.net/kns8s/basicSearch"

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30,
        max_pages: int = 5,
        allow_browser_automation: bool = False,
    ):
        self.headless = headless
        self.timeout = timeout
        self.max_pages = max_pages
        self.allow_browser_automation = allow_browser_automation
        self.last_status = "skipped"
        self.last_reason = "not started"
        self.last_search_stats: dict[str, float | int] = {
            "queries": 0,
            "returned": 0,
            "failed_queries": 0,
            "elapsed_seconds": 0.0,
        }

    def search_works(self, topic: str, limit: int = 200, year_from: int | None = None) -> list[PaperMetadata]:
        expanded = QueryExpander(llm_scorer=None).expand(topic)
        return self.search_with_expanded_query(expanded_query=expanded, limit=limit, year_from=year_from)

    def search_with_expanded_query(
        self,
        expanded_query: ExpandedQuery,
        limit: int = 200,
        year_from: int | None = None,
        progress=None,
        output_dir: Path | None = None,
    ) -> list[PaperMetadata]:
        start = time.perf_counter()
        self.last_search_stats = {
            "queries": 0,
            "returned": 0,
            "failed_queries": 0,
            "elapsed_seconds": 0.0,
        }
        self.last_status = "ok"
        self.last_reason = ""
        if not self.allow_browser_automation:
            logger.info("CNKI client disabled (allow_browser_automation=False).")
            self.last_status = "skipped"
            self.last_reason = "allow_browser_automation=False"
            return []
        queries = self._build_query_pool(expanded_query)
        self.last_search_stats["queries"] = len(queries)
        papers: list[PaperMetadata] = []
        task_id = progress.task("Searching CNKI queries", total=len(queries)) if progress else None
        for query in queries:
            if len(papers) >= limit:
                break
            result = self._search_single_query(
                query=query,
                limit=min(50, limit - len(papers)),
                year_from=year_from,
                output_dir=output_dir,
            )
            papers.extend(result)
            if progress and task_id is not None:
                progress.update(task_id, advance=1, description=f"CNKI: {query[:30]} ({len(result)})")
            if self.last_status in {"skipped", "restricted"}:
                break
            if self.last_status == "failed" and "Executable doesn't exist" in str(self.last_reason):
                self.last_status = "skipped"
                self.last_reason = "Playwright browser executable is missing"
                break
            time.sleep(1.0)
        logger.info("CNKI client finished with %s metadata records.", len(papers))
        final = papers[:limit]
        self.last_search_stats["returned"] = len(final)
        self.last_search_stats["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        return final

    def _search_single_query(
        self,
        query: str,
        limit: int,
        year_from: int | None,
        output_dir: Path | None = None,
    ) -> list[PaperMetadata]:
        if sync_playwright is None:
            logger.warning("Playwright is not installed. Skip CNKI query: %s", query)
            self.last_status = "skipped"
            self.last_reason = "Playwright is not installed"
            return []
        if BeautifulSoup is None:
            logger.warning("beautifulsoup4 is not installed. Skip CNKI query: %s", query)
            self.last_status = "skipped"
            self.last_reason = "beautifulsoup4 is not installed"
            return []

        papers: list[PaperMetadata] = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.firefox.launch(headless=self.headless)
                page = browser.new_page()
                self._open_search_page(page, query)
                if self._is_blocked_by_login_or_captcha(page):
                    self._safe_stop_if_restricted(page=page, output_dir=output_dir)
                    browser.close()
                    self.last_status = "restricted"
                    self.last_reason = "login/captcha/permission page detected"
                    return []

                for _ in range(self.max_pages):
                    if len(papers) >= limit:
                        break
                    current = self._parse_result_list(page)
                    for paper in current:
                        if year_from and paper.year and paper.year < year_from:
                            continue
                        paper.raw.setdefault("cnki", {})
                        paper.raw["cnki"]["matched_query"] = query
                        matched = paper.raw.get("matched_queries", [])
                        if isinstance(matched, list):
                            matched.append(f"cnki:{query}")
                        paper.raw["matched_queries"] = matched
                        papers.append(paper)
                        if len(papers) >= limit:
                            break

                    next_clicked = self._click_next_page(page)
                    if not next_clicked:
                        break
                    if self._is_blocked_by_login_or_captcha(page):
                        self._safe_stop_if_restricted(page=page, output_dir=output_dir)
                        self.last_status = "restricted"
                        self.last_reason = "login/captcha/permission page detected"
                        break
                browser.close()
        except PlaywrightTimeoutError as exc:  # pragma: no cover
            logger.warning("CNKI query timeout for '%s': %s", query, exc)
            self.last_search_stats["failed_queries"] = int(self.last_search_stats.get("failed_queries", 0)) + 1
            self.last_status = "failed"
            self.last_reason = f"timeout: {exc}"
        except Exception as exc:  # pragma: no cover
            logger.warning("CNKI query failed for '%s': %s", query, exc)
            self.last_search_stats["failed_queries"] = int(self.last_search_stats.get("failed_queries", 0)) + 1
            reason = str(exc)
            if "Executable doesn't exist" in reason:
                self.last_status = "skipped"
                self.last_reason = "Playwright browser executable is missing"
            else:
                self.last_status = "failed"
                self.last_reason = reason
            self._save_debug_html(page=None, output_dir=output_dir)
        return papers

    def _open_search_page(self, page, query: str):
        page.goto(self.SEARCH_URL, timeout=self.timeout * 1000)
        page.wait_for_timeout(1200)
        input_selectors = [
            "input#txt_SearchText",
            "input[placeholder*='主题']",
            "input[placeholder*='检索']",
            "input[type='text']",
        ]
        entered = False
        for selector in input_selectors:
            locator = page.locator(selector)
            if locator.count() > 0:
                locator.first.fill(query)
                entered = True
                break
        if not entered:
            raise RuntimeError("CNKI search input not found.")

        button_selectors = [
            "button.search-btn",
            "button:has-text('检索')",
            "a:has-text('检索')",
            "button[type='submit']",
        ]
        clicked = False
        for selector in button_selectors:
            locator = page.locator(selector)
            if locator.count() > 0:
                locator.first.click()
                clicked = True
                break
        if not clicked:
            page.keyboard.press("Enter")
        page.wait_for_timeout(1800)

    def _parse_result_list(self, page) -> list[PaperMetadata]:
        if BeautifulSoup is None:
            return []
        html = page.content()
        soup = BeautifulSoup(html, "lxml")
        papers: list[PaperMetadata] = []
        # CNKI page structures may change; keep parser permissive.
        candidate_nodes = soup.select("tr, div.result-table-list div.item, div.briefBox")
        for node in candidate_nodes[:80]:
            paper = self._parse_result_item(node)
            if paper:
                papers.append(paper)
        return papers

    def _parse_result_item(self, item) -> PaperMetadata | None:
        title = ""
        url = None
        title_link = item.select_one("a.fz14, a.name, a")
        if title_link:
            title = title_link.get_text(" ", strip=True)
            url = title_link.get("href")
        if not title:
            return None
        title = re.sub(r"\s+", " ", title).strip()

        text = item.get_text(" ", strip=True)
        year = self._extract_year(text)
        journal = None
        authors: list[str] = []
        keywords: list[str] = []
        abstract = None

        paper = PaperMetadata(
            id=f"CNKI::{title}",
            title=title,
            abstract=abstract,
            keywords=keywords,
            authors=authors,
            year=year,
            doi=None,
            journal=journal,
            publisher=None,
            citation_count=self._extract_count(text, r"被引[：:\s]*(\d+)"),
            source_database="CNKI",
            open_access_status="restricted",
            pdf_url=None,
            landing_page_url=url,
            paper_type="unknown",
            raw={
                "concepts": [],
                "topics": [],
                "primary_topic": None,
                "subjects": [],
                "matched_queries": [],
                "source_priority": "optional",
                "cnki": {
                    "source": "CNKI",
                    "download_count": self._extract_count(text, r"下载[：:\s]*(\d+)"),
                    "database": None,
                    "funding": None,
                    "degree_type": None,
                    "institution": None,
                    "matched_query": None,
                    "restricted": True,
                },
            },
        )
        return paper

    def _extract_detail_metadata(self, paper: PaperMetadata) -> PaperMetadata:
        # Optional extension point for CNKI detail parsing.
        return paper

    def _is_blocked_by_login_or_captcha(self, page) -> bool:
        page_text = page.content()
        blocked_markers = ["验证码", "登录", "机构", "权限", "购买", "收费", "访问受限"]
        return any(marker in page_text for marker in blocked_markers)

    def _safe_stop_if_restricted(self, page=None, output_dir: Path | None = None) -> None:
        logger.warning(
            "CNKI access is restricted (login/captcha/permission). "
            "Stop fulltext actions and keep metadata-only behavior."
        )
        self._save_debug_html(page=page, output_dir=output_dir)

    def _click_next_page(self, page) -> bool:
        selectors = [
            "a:has-text('下一页')",
            "a:has-text('下页')",
            "button:has-text('下一页')",
        ]
        for selector in selectors:
            locator = page.locator(selector)
            if locator.count() > 0 and locator.first.is_visible():
                locator.first.click()
                page.wait_for_timeout(1500)
                return True
        return False

    def _build_query_pool(self, expanded_query: ExpandedQuery) -> list[str]:
        queries: list[str] = []
        if expanded_query.chinese_topic:
            queries.append(expanded_query.chinese_topic)
        queries.append(expanded_query.original_topic)
        zh_keywords = expanded_query.chinese_keywords[:8]
        for i in range(0, len(zh_keywords), 2):
            chunk = zh_keywords[i : i + 3]
            if chunk:
                queries.append(" ".join(chunk))
        queries.extend(
            [
                "碳中和 电力投资",
                "碳价格 电力部门 中国",
                "综合评估模型 中国",
                "能源系统模型 碳中和",
            ]
        )
        deduped = []
        seen = set()
        for query in queries:
            norm = " ".join(str(query).split())
            key = norm.lower()
            if norm and key not in seen:
                seen.add(key)
                deduped.append(norm)
        return deduped[:10]

    def _extract_year(self, text: str) -> int | None:
        matches = re.findall(r"(19\d{2}|20\d{2})", text)
        if not matches:
            return None
        try:
            return int(matches[0])
        except ValueError:
            return None

    def _extract_count(self, text: str, pattern: str) -> int | None:
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    def _save_debug_html(self, page=None, output_dir: Path | None = None) -> None:
        if output_dir is None:
            return
        try:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            html_path = debug_dir / "cnki_failed.html"
            content = ""
            if page is not None:
                try:
                    content = page.content()
                except Exception:
                    content = ""
            if not content:
                content = "<html><body><h1>CNKI debug placeholder</h1><p>No page content captured.</p></body></html>"
            html_path.write_text(content, encoding="utf-8")
        except Exception:  # pragma: no cover
            return
