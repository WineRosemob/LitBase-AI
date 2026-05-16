from __future__ import annotations

from abc import ABC, abstractmethod

from litbase_ai.models import PaperMetadata


class BaseEnricher(ABC):
    """Abstract base class for metadata enrichers."""

    @abstractmethod
    def enrich(self, papers: list[PaperMetadata], progress=None) -> list[PaperMetadata]:
        """Enrich papers with additional metadata."""
