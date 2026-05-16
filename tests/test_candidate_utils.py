from __future__ import annotations

import unittest

from litbase_ai.download.candidate_utils import extract_pdf_urls_from_html, is_plausible_pdf_url


class CandidateUtilsTest(unittest.TestCase):
    def test_extract_pdf_urls_from_html_collects_multiple_patterns(self) -> None:
        html = """
        <html>
          <head><meta name="citation_pdf_url" content="/paper.pdf"></head>
          <body>
            <a href="/download?format=pdf&id=1">download</a>
            <script>window.open('/files/final.pdf');</script>
          </body>
        </html>
        """
        urls = extract_pdf_urls_from_html(html, "https://example.org/article")
        self.assertIn("https://example.org/paper.pdf", urls)
        self.assertIn("https://example.org/download?format=pdf&id=1", urls)
        self.assertIn("https://example.org/files/final.pdf", urls)

    def test_is_plausible_pdf_url_filters_provider_pages(self) -> None:
        self.assertTrue(is_plausible_pdf_url("https://example.org/file.pdf"))
        self.assertFalse(is_plausible_pdf_url("https://doaj.org/subjects/climate"))


if __name__ == "__main__":
    unittest.main()
