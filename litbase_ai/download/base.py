from __future__ import annotations

from abc import ABC, abstractmethod

from litbase_ai.models import ScoredPaper


class BaseDownloader(ABC):
    """Abstract base class for paper downloaders."""

    @abstractmethod
    def download_batch(self, papers: list[ScoredPaper], progress=None) -> list[ScoredPaper]:
        """Download artifacts for a list of scored papers."""
