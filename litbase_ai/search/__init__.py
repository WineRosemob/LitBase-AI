"""Search clients for scholarly metadata sources."""

from litbase_ai.search.arxiv_client import ArxivClient
from litbase_ai.search.cnki_client import CNKIClient
from litbase_ai.search.crossref_client import CrossrefClient
from litbase_ai.search.openalex_client import OpenAlexClient
from litbase_ai.search.semantic_scholar_client import SemanticScholarClient

__all__ = [
    "OpenAlexClient",
    "CrossrefClient",
    "ArxivClient",
    "SemanticScholarClient",
    "CNKIClient",
]
