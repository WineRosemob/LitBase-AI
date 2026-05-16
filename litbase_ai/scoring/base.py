from __future__ import annotations

from abc import ABC, abstractmethod

from litbase_ai.models import ExpandedQuery, PaperMetadata, PaperScore


class BaseScorer(ABC):
    """Abstract base class for paper scoring."""

    @abstractmethod
    def score(
        self,
        paper: PaperMetadata,
        topic: str,
        expanded_query: ExpandedQuery | None = None,
    ) -> PaperScore:
        """Score a paper under a research topic."""
