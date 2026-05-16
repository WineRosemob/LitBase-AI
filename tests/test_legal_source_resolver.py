from __future__ import annotations

import unittest
from unittest.mock import patch

from litbase_ai.download.legal_source_resolver import LegalPDFSourceResolver
from litbase_ai.models import PaperMetadata


class LegalPDFSourceResolverTest(unittest.TestCase):
    def test_resolve_discovers_candidates_and_caches_result(self) -> None:
        resolver = LegalPDFSourceResolver(
            openalex_mailto="user@example.com",
            unpaywall_email="user@example.com",
            enable_discovery=True,
            enable_crossref_page_scrape=True,
        )
        paper = PaperMetadata(
            id="paper-1",
            title="A strong title for climate adaptation modeling",
            source_database="Crossref",
            raw={},
        )

        def fake_request_json(client, url, params=None, headers=None):
            if "unpaywall" in url:
                return {
                    "best_oa_location": {
                        "url_for_pdf": "https://repo.example.org/unpaywall.pdf",
                        "url_for_landing_page": "https://repo.example.org/landing",
                    },
                    "oa_locations": [],
                }
            if "openalex.org/works/doi:" in url:
                return {
                    "best_oa_location": {"pdf_url": "https://openalex.example.org/openalex.pdf"},
                    "primary_location": {"landing_page_url": "https://publisher.example.org/article"},
                    "open_access": {"oa_url": "https://openalex.example.org/openalex.pdf"},
                }
            if "crossref.org/works/" in url:
                return {
                    "message": {
                        "URL": "https://publisher.example.org/article",
                        "link": [{"URL": "https://publisher.example.org/direct.pdf", "content-type": "application/pdf"}],
                    }
                }
            if "api.openaire.eu" in url:
                return {"response": {"results": {"result": []}}}
            if "doaj.org" in url:
                return {"results": []}
            if "europepmc" in url:
                return {
                    "resultList": {
                        "result": [
                            {
                                "fullTextUrlList": {
                                    "fullTextUrl": [
                                        {
                                            "url": "https://europepmc.example.org/fulltext.pdf",
                                            "documentStyle": "pdf",
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            if "pmc/utils/idconv" in url:
                return {"records": [{"pmcid": "PMC123456"}]}
            if "api.core.ac.uk" in url:
                return {
                    "results": [
                        {
                            "downloadUrl": "https://core.example.org/core.pdf",
                            "alternateUrl": "https://core.example.org/landing",
                        }
                    ]
                }
            return None

        with patch.object(resolver, "_resolve_title_to_doi", return_value="10.1000/test"), patch.object(
            resolver,
            "_request_json",
            side_effect=fake_request_json,
        ), patch.object(
            resolver,
            "_resolve_doi_landing_page",
            return_value="https://publisher.example.org/article",
        ), patch.object(
            resolver,
            "_request_text",
            return_value=(
                '<html><meta name="citation_pdf_url" content="/scraped.pdf"></html>',
                "https://publisher.example.org/article",
            ),
        ):
            candidates = resolver.resolve(paper)
            cached_candidates = resolver.resolve(paper)

        urls = [item["url"] for item in candidates]
        self.assertEqual(paper.doi, "10.1000/test")
        self.assertIn("https://repo.example.org/unpaywall.pdf", urls)
        self.assertIn("https://openalex.example.org/openalex.pdf", urls)
        self.assertIn("https://publisher.example.org/direct.pdf", urls)
        self.assertIn("https://publisher.example.org/scraped.pdf", urls)
        self.assertIn("https://europepmc.example.org/fulltext.pdf", urls)
        self.assertIn("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/", urls)
        self.assertIn("https://core.example.org/core.pdf", urls)
        self.assertEqual(candidates, cached_candidates)
        self.assertGreaterEqual(resolver.last_stats["cache_hits"], 1)
        self.assertEqual(paper.raw["download_discovery"]["resolved_doi"], "10.1000/test")


if __name__ == "__main__":
    unittest.main()
