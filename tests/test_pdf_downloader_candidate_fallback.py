from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from litbase_ai.download.pdf_downloader import PDFDownloader
from litbase_ai.models import PaperMetadata, PaperScore, ScoredPaper


class _NeverResolveSciHub:
    def __init__(self) -> None:
        self.resolve_calls = 0

    def resolve_pdf_url(self, doi: str) -> str | None:
        self.resolve_calls += 1
        raise AssertionError("resolve_pdf_url should not be called during candidate collection")

    def try_download(self, doi: str, output_path: Path):
        return False, {"trace": []}


class _NeverResolveLibGen:
    def __init__(self) -> None:
        self.resolve_calls = 0

    def resolve_pdf_url(self, doi: str, title: str = "") -> str | None:
        self.resolve_calls += 1
        raise AssertionError("resolve_pdf_url should not be called during candidate collection")

    def try_download(self, doi: str, output_path: Path, title: str = ""):
        return False, {"trace": []}


class PDFDownloaderCandidateFallbackTest(unittest.TestCase):
    def test_collect_candidates_adds_lazy_fallback_without_network_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = PDFDownloader(output_dir=Path(tmp_dir))
            fake_scihub = _NeverResolveSciHub()
            fake_libgen = _NeverResolveLibGen()

            downloader.enable_scihub = True
            downloader.enable_libgen = True
            downloader.scihub_client = fake_scihub
            downloader.libgen_client = fake_libgen

            paper = ScoredPaper(
                metadata=PaperMetadata(
                    id="paper-1",
                    title="Sample Power System Study",
                    doi="10.1000/test-doi",
                    source_database="OpenAlex",
                    raw={},
                ),
                score=PaperScore(
                    paper_id="paper-1",
                    relevance_score=80,
                    year_score=80,
                    citation_score=70,
                    journal_score=70,
                    access_score=70,
                    type_score=70,
                    rule_score=75,
                ),
            )

            candidates = downloader._collect_candidate_urls(paper)  # noqa: SLF001
            sources = [item.get("source") for item in candidates]

            self.assertIn("scihub", sources)
            self.assertIn("libgen", sources)
            self.assertEqual(fake_scihub.resolve_calls, 0)
            self.assertEqual(fake_libgen.resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
