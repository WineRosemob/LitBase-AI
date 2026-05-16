from __future__ import annotations

import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from litbase_ai.models import ExpandedQuery, PaperMetadata
from litbase_ai.pipeline import LitBasePipeline


class _DummyProgress:
    def log(self, message: str, level: str = "info") -> None:
        return None

    def task(self, description: str, total: int | None = None):
        return None

    def update(self, task_id, advance: int = 1, description: str | None = None) -> None:
        return None


class _SleepClient:
    def __init__(self, label: str, sleep_seconds: float):
        self.label = label
        self.sleep_seconds = sleep_seconds
        self.last_status = "ok"
        self.last_reason = ""
        self.last_search_stats = {"queries": 1, "returned": 0, "failed_queries": 0, "elapsed_seconds": 0.0}

    def search_with_expanded_query(self, expanded_query, limit: int, year_from=None, progress=None):
        start = time.perf_counter()
        time.sleep(self.sleep_seconds)
        self.last_search_stats = {
            "queries": 1,
            "returned": 1,
            "failed_queries": 0,
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        }
        return [
            PaperMetadata(
                id=f"{self.label}-1",
                title=f"{self.label} paper",
                source_database=self.label,
                raw={},
            )
        ]


class PipelineSearchParallelTest(unittest.TestCase):
    def test_search_sources_waits_for_all_parallel_tasks(self) -> None:
        pipeline = LitBasePipeline.__new__(LitBasePipeline)
        pipeline.enable_openalex = True
        pipeline.enable_crossref = True
        pipeline.enable_arxiv = True
        pipeline.enable_cnki = False
        pipeline.enable_semantic_scholar = True
        pipeline.year_from = None
        pipeline.cnki_headless = True
        pipeline.cnki_max_pages = 1
        pipeline.cnki_limit = 10
        pipeline.output_dir = Path(".")
        pipeline.progress = _DummyProgress()
        pipeline.diagnostics = {}
        pipeline.config = SimpleNamespace(search_source_workers=4, semantic_scholar_api_key="s2-key")
        pipeline.openalex_client = _SleepClient("OpenAlex", 0.2)
        pipeline.semantic_client = _SleepClient("Semantic Scholar", 0.2)
        pipeline.crossref_client = _SleepClient("Crossref", 0.2)
        pipeline.arxiv_client = _SleepClient("arXiv", 0.2)

        expanded = ExpandedQuery(original_topic="climate")

        start = time.perf_counter()
        results = pipeline._search_sources(expanded_query=expanded, candidate_limit=40)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.65)
        self.assertEqual(len(results["OpenAlex"]), 1)
        self.assertEqual(len(results["Semantic Scholar"]), 1)
        self.assertEqual(len(results["Crossref"]), 1)
        self.assertEqual(len(results["arXiv"]), 1)
        self.assertEqual(results["CNKI"], [])
        self.assertEqual(pipeline.diagnostics["search_execution"]["mode"], "parallel")


if __name__ == "__main__":
    unittest.main()
