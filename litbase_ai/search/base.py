from __future__ import annotations

from abc import ABC, abstractmethod

from litbase_ai.models import ExpandedQuery, PaperMetadata


class BaseSearchClient(ABC):
    """Abstract interface for literature search clients."""

    @abstractmethod
    def search_works(
        self,
        topic: str,
        limit: int = 500,
        year_from: int | None = None,
    ) -> list[PaperMetadata]:
        """Search metadata by topic and return unified paper list."""

    def search_with_expanded_query(
        self,
        expanded_query: ExpandedQuery,
        limit: int = 500,
        year_from: int | None = None,
        progress=None,
        **kwargs,
    ) -> list[PaperMetadata]:
        """Search using expanded query object; fallback to original topic search."""
        return self.search_works(
            topic=expanded_query.original_topic,
            limit=limit,
            year_from=year_from,
        )
