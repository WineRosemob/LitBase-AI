from __future__ import annotations

import hashlib
from typing import Any

from litbase_ai.models import EmbeddingScore, ExpandedQuery, PaperMetadata, ScoredPaper
from litbase_ai.utils.cache import CacheManager
from litbase_ai.utils.logging import get_logger


logger = get_logger(__name__)


class EmbeddingScorer:
    """Optional embedding-based similarity scorer."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        enabled: bool = False,
        cache_manager: CacheManager | None = None,
        progress=None,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.cache_manager = cache_manager
        self.progress = progress
        self._model = None
        self._model_load_error: str | None = None
        self.last_stats: dict[str, int | bool] = {
            "embedding_enabled": enabled,
            "embedding_scored": 0,
            "embedding_failed": 0,
            "cache_hit": 0,
            "cache_miss": 0,
        }

    def score(
        self,
        paper: PaperMetadata,
        topic: str,
        expanded_query: ExpandedQuery | None = None,
    ) -> EmbeddingScore:
        if not self.enabled:
            return EmbeddingScore(enabled=False, model_name=self.model_name)

        cache_key = self._build_cache_key(paper, topic, expanded_query)
        if self.cache_manager:
            cached = self.cache_manager.get("embedding", cache_key)
            if cached:
                self.last_stats["cache_hit"] = int(self.last_stats.get("cache_hit", 0)) + 1
                return EmbeddingScore(**cached)
        self.last_stats["cache_miss"] = int(self.last_stats.get("cache_miss", 0)) + 1

        model = self._load_model()
        if model is None:
            return EmbeddingScore(enabled=False, model_name=self.model_name)

        try:
            title_sim = self._similarity(model, topic, paper.title or "")
            abstract_sim = self._similarity(model, topic, paper.abstract or "")
            keyword_text = ", ".join(paper.keywords)
            keyword_sim = self._similarity(model, topic, keyword_text)
            values = [x for x in [title_sim, abstract_sim, keyword_sim] if x is not None]
            combined = sum(values) / len(values) if values else None
            score = EmbeddingScore(
                title_similarity=title_sim,
                abstract_similarity=abstract_sim,
                keyword_similarity=keyword_sim,
                combined_similarity=combined,
                model_name=self.model_name,
                enabled=True,
            )
            if self.cache_manager:
                self.cache_manager.set("embedding", cache_key, score.model_dump())
            return score
        except Exception as exc:  # pragma: no cover
            logger.warning("Embedding scoring failed: %s", exc)
            return EmbeddingScore(enabled=False, model_name=self.model_name)

    def score_batch(
        self,
        papers: list[ScoredPaper],
        topic: str,
        expanded_query: ExpandedQuery | None = None,
    ) -> list[ScoredPaper]:
        self.last_stats = {
            "embedding_enabled": self.enabled,
            "embedding_scored": 0,
            "embedding_failed": 0,
            "cache_hit": 0,
            "cache_miss": 0,
        }
        if not papers:
            return papers
        if not self.enabled:
            return papers
        task_id = self.progress.task("Embedding scoring", total=len(papers)) if self.progress else None
        for idx, paper in enumerate(papers):
            emb = self.score(paper.metadata, topic, expanded_query=expanded_query)
            paper.score.embedding_score = emb
            if emb.enabled and emb.combined_similarity is not None:
                self.last_stats["embedding_scored"] = int(self.last_stats.get("embedding_scored", 0)) + 1
            else:
                self.last_stats["embedding_failed"] = int(self.last_stats.get("embedding_failed", 0)) + 1
            if task_id is not None:
                self.progress.update(task_id, advance=1, description=f"Embedding {idx + 1}/{len(papers)}")
        return papers

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._model_load_error:
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            self._model_load_error = "sentence-transformers not installed"
            logger.warning("Embedding score skipped: sentence-transformers not installed.")
            return None
        try:
            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception as exc:  # pragma: no cover
            self._model_load_error = str(exc)
            logger.warning("Embedding model load failed: %s", exc)
            return None

    def _similarity(self, model, left: str, right: str) -> float | None:
        left = (left or "").strip()
        right = (right or "").strip()
        if not left or not right:
            return None
        vectors = model.encode([left, right], normalize_embeddings=True)
        a = vectors[0]
        b = vectors[1]
        sim = float((a * b).sum())
        score = max(0.0, min(100.0, (sim + 1.0) * 50.0))
        return round(score, 2)

    def _build_cache_key(self, paper: PaperMetadata, topic: str, expanded_query: ExpandedQuery | None) -> str:
        abstract_hash = hashlib.md5((paper.abstract or "").encode("utf-8")).hexdigest()  # noqa: S324
        eq = expanded_query.model_dump() if expanded_query else {}
        payload: dict[str, Any] = {
            "topic": topic,
            "paper_id": paper.id,
            "title": paper.title,
            "abstract_hash": abstract_hash,
            "model_name": self.model_name,
            "expanded_query": eq,
            "version": "embedding_v1",
        }
        return str(payload)
