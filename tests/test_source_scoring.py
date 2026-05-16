from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from litbase_ai.download.source_scoring import DownloadSourceScorer, normalize_source_label


class DownloadSourceScorerTest(unittest.TestCase):
    def test_normalize_source_label(self) -> None:
        self.assertEqual(normalize_source_label("unpaywall.oa_locations[0].url_for_pdf"), "unpaywall")
        self.assertEqual(normalize_source_label("crossref.page_scrape[0]:html"), "crossref")
        self.assertEqual(normalize_source_label("metadata.pdf_url"), "metadata.pdf_url")

    def test_record_persists_and_updates_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scorer = DownloadSourceScorer(Path(tmp) / "scores.json")
            scorer.record("openalex.primary_location.pdf_url", True, latency_ms=1000)
            scorer.record("openalex.primary_location.pdf_url", False, reason="http_403")
            self.assertGreaterEqual(scorer.success_score("openalex.best_oa_location.url_for_pdf"), 0.0)
            self.assertGreater(scorer.latency_ms("openalex.best_oa_location.url_for_pdf"), 0.0)
            snapshot = scorer.snapshot()
            self.assertIn("openalex", snapshot)
            self.assertEqual(snapshot["openalex"]["last_error"], "http_403")


if __name__ == "__main__":
    unittest.main()
