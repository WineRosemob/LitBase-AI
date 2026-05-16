from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from litbase_ai.models import EvidenceItem, ExpandedQuery, RubricScore, ScoredPaper
from litbase_ai.utils.cache import CacheManager
from litbase_ai.utils.logging import get_logger


logger = get_logger(__name__)


class LLMRubricScorer:
    """LLM rubric scorer with JSON parsing, cache and graceful fallback."""

    PROMPT_VERSION = "rubric_v1"

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        prompt_template_path: Path,
        cache_manager: CacheManager | None = None,
        progress=None,
        connect_timeout: float = 15.0,
        read_timeout: float = 45.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt_template_path = prompt_template_path
        self.cache_manager = cache_manager
        self.progress = progress
        self.timeout = httpx.Timeout(
            connect=max(1.0, float(connect_timeout)),
            read=max(1.0, float(read_timeout)),
            write=max(1.0, float(read_timeout)),
            pool=max(1.0, float(connect_timeout)),
        )
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.user_agent = "LitBase-AI/0.1"
        self.prompt_template = self._load_prompt_template()
        self.last_stats: dict[str, int | bool] = {
            "llm_candidates": 0,
            "llm_rubric_enabled": bool(api_key),
            "llm_rubric_scored": 0,
            "llm_rubric_failed": 0,
            "llm_skipped": 0,
            "llm_cache_hit": 0,
            "llm_cache_miss": 0,
            "fatal_fallback": 0,
        }

    def score_one(
        self,
        paper: ScoredPaper,
        topic: str,
        expanded_query: ExpandedQuery | None = None,
    ) -> ScoredPaper:
        if not self.api_key:
            self.last_stats["llm_skipped"] = int(self.last_stats.get("llm_skipped", 0)) + 1
            return paper

        cache_key = self._build_cache_key(paper, topic, expanded_query)
        if self.cache_manager:
            cached = self.cache_manager.get("llm_rubric", cache_key)
            if cached:
                rubric = self._to_rubric(cached.get("rubric"))
                if rubric:
                    self.last_stats["llm_cache_hit"] = int(self.last_stats.get("llm_cache_hit", 0)) + 1
                    return self._apply_rubric(paper, rubric)
        self.last_stats["llm_cache_miss"] = int(self.last_stats.get("llm_cache_miss", 0)) + 1

        prompt = self._build_prompt(paper, topic, expanded_query)
        response = self._call_api(prompt)
        parsed = self._parse_response(response) if response else None
        if parsed is None:
            response_retry = self._call_api(prompt)
            parsed = self._parse_response(response_retry) if response_retry else None
        if parsed is None:
            self.last_stats["llm_rubric_failed"] = int(self.last_stats.get("llm_rubric_failed", 0)) + 1
            return paper

        rubric = self._to_rubric(parsed)
        if rubric is None:
            self.last_stats["llm_rubric_failed"] = int(self.last_stats.get("llm_rubric_failed", 0)) + 1
            return paper

        if self.cache_manager:
            self.cache_manager.set("llm_rubric", cache_key, {"rubric": rubric.model_dump(), "version": self.PROMPT_VERSION})
        self.last_stats["llm_rubric_scored"] = int(self.last_stats.get("llm_rubric_scored", 0)) + 1
        return self._apply_rubric(paper, rubric)

    def score_batch(
        self,
        papers: list[ScoredPaper],
        topic: str,
        expanded_query: ExpandedQuery | None = None,
        max_papers: int = 100,
    ) -> list[ScoredPaper]:
        self.last_stats = {
            "llm_candidates": len(papers),
            "llm_rubric_enabled": bool(self.api_key),
            "llm_rubric_scored": 0,
            "llm_rubric_failed": 0,
            "llm_skipped": 0,
            "llm_cache_hit": 0,
            "llm_cache_miss": 0,
            "fatal_fallback": 0,
        }
        if not papers:
            return papers
        if not self.api_key:
            self.last_stats["llm_skipped"] = len(papers)
            if self.progress:
                self.progress.log("LLM rubric scoring skipped: no API key found.")
            return papers

        task_total = min(len(papers), max_papers)
        task_id = self.progress.task("LLM rubric scoring", total=task_total) if self.progress else None
        fatal_mode = False
        consecutive_failures = 0
        for idx, paper in enumerate(papers[:max_papers]):
            if fatal_mode:
                self.last_stats["llm_skipped"] = int(self.last_stats.get("llm_skipped", 0)) + 1
                if task_id is not None:
                    self.progress.update(task_id, advance=1, description=f"Rubric skipped {idx + 1}/{task_total}")
                continue
            scored = self.score_one(paper, topic, expanded_query=expanded_query)
            papers[idx] = scored
            if scored.score.rubric_score and scored.score.rubric_score.final_llm_score is not None:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    fatal_mode = True
                    self.last_stats["fatal_fallback"] = int(self.last_stats.get("fatal_fallback", 0)) + 1
            if task_id is not None:
                self.progress.update(task_id, advance=1, description=f"Rubric {idx + 1}/{task_total}: {(paper.metadata.title or '')[:70]}")
        if len(papers) > max_papers:
            self.last_stats["llm_skipped"] = int(self.last_stats.get("llm_skipped", 0)) + (len(papers) - max_papers)
        return papers

    def _build_prompt(self, paper: ScoredPaper, topic: str, expanded_query: ExpandedQuery | None) -> str:
        expanded_keywords = []
        if expanded_query:
            expanded_keywords = expanded_query.english_keywords[:10] + expanded_query.chinese_keywords[:10]
        mapping = {
            "{topic}": topic or "",
            "{expanded_keywords}": ", ".join(expanded_keywords),
            "{title}": paper.metadata.title or "",
            "{year}": str(paper.metadata.year or ""),
            "{journal}": paper.metadata.journal or "",
            "{citation_count}": str(paper.metadata.citation_count or 0),
            "{keywords}": ", ".join(paper.metadata.keywords),
            "{abstract}": paper.metadata.abstract or "",
        }
        prompt = self.prompt_template
        for placeholder, value in mapping.items():
            prompt = prompt.replace(placeholder, value)
        return prompt

    def _call_api(self, prompt: str) -> dict[str, Any] | None:
        if not self.api_key:
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        url = f"{self.base_url}/chat/completions"
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, headers=headers) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                body = exc.response.text[:200] if exc.response is not None else ""
                logger.warning(
                    "LLM rubric HTTP error attempt=%s/%s status=%s body=%s",
                    attempt,
                    self.max_retries,
                    status,
                    body,
                )
                if status in (400, 401, 403, 404):
                    return None
                if attempt >= self.max_retries:
                    return None
            except httpx.TimeoutException as exc:
                logger.warning("LLM rubric timeout attempt=%s/%s: %s", attempt, self.max_retries, exc)
                if attempt >= self.max_retries:
                    return None
            except Exception as exc:  # pragma: no cover
                logger.warning("LLM rubric call failed attempt=%s/%s: %s", attempt, self.max_retries, exc)
                if attempt >= self.max_retries:
                    return None
            self._sleep_before_retry(attempt)
        return None

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff_seconds <= 0:
            return
        time.sleep(min(20.0, self.retry_backoff_seconds * attempt))

    def _parse_response(self, response: dict[str, Any]) -> dict[str, Any] | None:
        text = (
            (response.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else None
        except Exception:
            left = cleaned.find("{")
            right = cleaned.rfind("}")
            if left >= 0 and right > left:
                try:
                    data = json.loads(cleaned[left : right + 1])
                    return data if isinstance(data, dict) else None
                except Exception:
                    return None
            return None

    def _to_rubric(self, payload: dict[str, Any] | None) -> RubricScore | None:
        if not isinstance(payload, dict):
            return None
        try:
            rubric = RubricScore(
                topic_relevance=self._safe_float(payload.get("topic_relevance")),
                object_relevance=self._safe_float(payload.get("object_relevance")),
                method_relevance=self._safe_float(payload.get("method_relevance")),
                data_relevance=self._safe_float(payload.get("data_relevance")),
                novelty=self._safe_float(payload.get("novelty")),
                citation_value=self._safe_float(payload.get("citation_value")),
                writing_value=self._safe_float(payload.get("writing_value")),
                policy_relevance=self._safe_float(payload.get("policy_relevance")),
                confidence=self._safe_float(payload.get("confidence")),
                final_llm_score=self._safe_float(payload.get("final_llm_score")),
                decision=(str(payload.get("decision")).strip() if payload.get("decision") else None),
                reason=(str(payload.get("reason")).strip() if payload.get("reason") else None),
                evidence=[str(x).strip() for x in (payload.get("evidence") or []) if str(x).strip()],
                usable_for=[str(x).strip() for x in (payload.get("usable_for") or []) if str(x).strip()],
                labels=[str(x).strip() for x in (payload.get("labels") or []) if str(x).strip()],
            )
            return rubric
        except Exception:
            return None

    def _apply_rubric(self, paper: ScoredPaper, rubric: RubricScore) -> ScoredPaper:
        paper.score.rubric_score = rubric
        paper.score.llm_score = rubric.final_llm_score
        paper.score.llm_reason = rubric.reason
        merged_labels = list(dict.fromkeys(paper.score.labels + rubric.labels))
        paper.score.labels = merged_labels

        if rubric.decision:
            paper.score.final_decision = rubric.decision
        if rubric.confidence is not None:
            paper.score.final_confidence = rubric.confidence

        for ev in rubric.evidence[:3]:
            item = EvidenceItem(source="llm_rubric", text=ev, relevance="llm", confidence=rubric.confidence)
            paper.score.evidence_items.append(item)

        paper.metadata.raw["llm_rubric"] = rubric.model_dump()
        return paper

    def _build_cache_key(self, paper: ScoredPaper, topic: str, expanded_query: ExpandedQuery | None) -> str:
        abstract = paper.metadata.abstract or ""
        abstract_hash = hashlib.md5(abstract.encode("utf-8")).hexdigest()  # noqa: S324
        eq = expanded_query.model_dump() if expanded_query else {}
        payload = {
            "topic": topic,
            "paper_id": paper.metadata.id,
            "title": paper.metadata.title,
            "abstract_hash": abstract_hash,
            "model": self.model,
            "prompt_version": self.PROMPT_VERSION,
            "expanded_query": eq,
        }
        return str(payload)

    def _load_prompt_template(self) -> str:
        if self.prompt_template_path.exists():
            return self.prompt_template_path.read_text(encoding="utf-8")
        return (
            "Topic: {topic}\nExpanded keywords: {expanded_keywords}\nTitle: {title}\n"
            "Abstract: {abstract}\nReturn strict JSON with rubric fields and final_llm_score."
        )

    def _safe_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            number = float(value)
            return max(0.0, min(100.0, number))
        except Exception:
            return None
