from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from litbase_ai.models import ScoredPaper
from litbase_ai.utils.logging import get_logger


logger = get_logger(__name__)


class LLMScorer:
    """LLM-based secondary scorer using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        prompt_template_path: Path,
        *,
        connect_timeout: float = 15.0,
        read_timeout: float = 45.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt_template_path = prompt_template_path
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
        self._fatal_error = False
        self._fatal_reason = ""
        self.last_stats: dict[str, int] = {
            "candidate_count": 0,
            "scored_count": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }

    def score_batch(
        self,
        papers: list[ScoredPaper],
        topic: str,
        max_papers: int = 100,
        progress=None,
    ) -> list[ScoredPaper]:
        self.last_stats = {
            "candidate_count": len(papers),
            "scored_count": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }
        self._fatal_error = False
        self._fatal_reason = ""
        if not papers:
            return papers
        if not self.api_key:
            message = "LLM scoring skipped: no API key found. final_score = rule_score"
            if progress:
                progress.log(message)
            else:
                logger.info(message)
            for paper in papers:
                paper.score.final_score = paper.score.rule_score
            self.last_stats["skipped"] = len(papers)
            return papers

        scored: list[ScoredPaper] = []
        consecutive_failures = 0
        task_id = progress.task("LLM scoring papers", total=min(len(papers), max_papers)) if progress else None
        for paper in papers[:max_papers]:
            if self._fatal_error:
                paper.score.final_score = paper.score.rule_score
                scored.append(paper)
                self.last_stats["skipped"] += 1
                if progress and task_id is not None:
                    progress.update(task_id, advance=1, description="LLM disabled after fatal error")
                continue
            short_title = (paper.metadata.title or "")[:80]
            updated = self.score_one(paper, topic)
            scored.append(updated)
            self.last_stats["scored_count"] += 1
            if updated.score.llm_score is not None:
                self.last_stats["success"] += 1
                consecutive_failures = 0
            else:
                self.last_stats["failed"] += 1
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    self._fatal_error = True
                    if not self._fatal_reason:
                        self._fatal_reason = "3 consecutive LLM failures"
            if progress and task_id is not None:
                progress.update(task_id, advance=1, description=f"LLM scoring: {short_title}")
        for paper in papers[max_papers:]:
            paper.score.final_score = paper.score.rule_score
            scored.append(paper)
            self.last_stats["skipped"] += 1
        if progress:
            progress.log(
                "LLM scoring summary: "
                f"success={self.last_stats['success']} failed={self.last_stats['failed']} skipped={self.last_stats['skipped']}"
            )
            if self._fatal_error:
                progress.log(f"LLM scoring fallback activated: {self._fatal_reason}", level="warning")
        return scored

    def score_one(self, paper: ScoredPaper, topic: str) -> ScoredPaper:
        try:
            prompt = self._build_prompt(paper, topic)
            result = self._call_api(prompt)
            if not result:
                paper.score.final_score = paper.score.rule_score
                return paper

            message_text = (
                (result.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            parsed = self._parse_response(message_text)
            if not parsed:
                paper.score.final_score = paper.score.rule_score
                paper.metadata.raw["llm_raw_result"] = message_text
                return paper
            paper.metadata.raw["llm_raw_result"] = {"raw_text": message_text, "parsed": parsed}
            return self._update_scored_paper(paper, parsed)
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM score_one failed, fallback to rule_score: %s", exc)
            paper.score.final_score = paper.score.rule_score
            return paper

    def _build_prompt(self, paper: ScoredPaper, topic: str) -> str:
        metadata = paper.metadata
        mapping = {
            "{topic}": str(topic or ""),
            "{title}": str(metadata.title or ""),
            "{year}": str(metadata.year or ""),
            "{journal}": str(metadata.journal or ""),
            "{citation_count}": str(metadata.citation_count or 0),
            "{keywords}": ", ".join(metadata.keywords),
            "{abstract}": str(metadata.abstract or ""),
        }
        prompt = self.prompt_template
        for placeholder, value in mapping.items():
            prompt = prompt.replace(placeholder, value)
        return prompt

    def _call_api(self, prompt: str) -> dict | None:
        if not self.api_key:
            return None
        url = f"{self.base_url}/chat/completions"
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
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, headers=headers) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                body = exc.response.text[:300] if exc.response is not None else ""
                logger.warning(
                    "LLM scoring HTTP error attempt=%s/%s status=%s body=%s",
                    attempt,
                    self.max_retries,
                    status,
                    body,
                )
                if status in (400, 401, 403, 404):
                    self._fatal_error = True
                    self._fatal_reason = f"HTTP {status}"
                    return None
                if attempt >= self.max_retries:
                    return None
            except httpx.TimeoutException as exc:
                logger.warning("LLM scoring timeout attempt=%s/%s: %s", attempt, self.max_retries, exc)
                if attempt >= self.max_retries:
                    return None
            except Exception as exc:  # pragma: no cover
                logger.warning("LLM scoring API call failed attempt=%s/%s: %s", attempt, self.max_retries, exc)
                if attempt >= self.max_retries:
                    return None
            self._sleep_before_retry(attempt)
        return None

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff_seconds <= 0:
            return
        time.sleep(min(20.0, self.retry_backoff_seconds * attempt))

    def _parse_response(self, text: str) -> dict | None:
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
            if "llm_score" not in data:
                return None
            return data
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON.")
            return None

    def _update_scored_paper(self, paper: ScoredPaper, llm_result: dict) -> ScoredPaper:
        llm_score_raw = llm_result.get("llm_score")
        try:
            llm_score = float(llm_score_raw)
        except (TypeError, ValueError):
            llm_score = None

        if llm_score is None:
            paper.score.final_score = paper.score.rule_score
            return paper
        llm_score = max(0.0, min(100.0, llm_score))
        paper.score.llm_score = round(llm_score, 2)
        paper.score.llm_reason = llm_result.get("reason")
        final = 0.6 * paper.score.rule_score + 0.4 * llm_score
        paper.score.final_score = round(final, 2)

        new_labels = llm_result.get("labels") or []
        if isinstance(new_labels, list):
            merged_labels = list(dict.fromkeys(paper.score.labels + [str(label) for label in new_labels]))
            paper.score.labels = merged_labels
        paper.metadata.raw["llm_structured"] = llm_result
        if llm_result.get("paper_type"):
            paper.metadata.paper_type = llm_result.get("paper_type")
        return paper

    def _load_prompt_template(self) -> str:
        if not self.prompt_template_path.exists():
            logger.warning("Prompt template not found: %s", self.prompt_template_path)
            return (
                "Topic: {topic}\nTitle: {title}\nAbstract: {abstract}\n"
                "Return JSON with llm_score, paper_type, research_object, method, usable_for, reason, labels."
            )
        return self.prompt_template_path.read_text(encoding="utf-8")
