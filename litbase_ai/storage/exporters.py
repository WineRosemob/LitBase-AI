from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from litbase_ai.models import ExpandedQuery, PaperMetadata, ScoredPaper
from litbase_ai.utils.logging import get_logger
from litbase_ai.utils.text import clean_filename, extract_first_author, short_title


logger = get_logger(__name__)


class PaperExporter:
    """Exporter for JSONL, Excel, BibTeX and markdown reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_stats: dict[str, Any] = {}
        self.last_outputs: dict[str, str] = {}

    def export_raw_jsonl(self, papers: list[PaperMetadata], filename: str = "papers_raw.jsonl", progress=None) -> Path:
        path = self.output_dir / filename
        if progress:
            progress.log(f"Exporting {filename} ...")
        with path.open("w", encoding="utf-8") as file:
            for paper in papers:
                file.write(json.dumps(paper.model_dump(), ensure_ascii=False) + "\n")
        if progress:
            progress.log(f"Exported {filename} -> {path}")
        self.last_outputs["papers_raw"] = str(path)
        return path

    def export_expanded_query(self, expanded_query: ExpandedQuery, filename: str = "expanded_query.json", progress=None) -> Path:
        path = self.output_dir / filename
        if progress:
            progress.log(f"Exporting {filename} ...")
        path.write_text(json.dumps(expanded_query.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        if progress:
            progress.log(f"Exported {filename} -> {path}")
        self.last_outputs["expanded_query"] = str(path)
        return path

    def export_search_diagnostics(self, diagnostics: dict[str, Any], filename: str = "search_diagnostics.json", progress=None) -> Path:
        path = self.output_dir / filename
        if progress:
            progress.log(f"Exporting {filename} ...")
        path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress:
            progress.log(f"Exported {filename} -> {path}")
        self.last_outputs["search_diagnostics"] = str(path)
        return path

    def export_scored_excel(self, papers: list[ScoredPaper], filename: str = "papers_scored.xlsx", progress=None) -> Path:
        return self._export_excel(papers, filename=filename, output_key="papers_scored", progress=progress)

    def export_selected_excel(self, papers: list[ScoredPaper], filename: str = "papers_selected.xlsx", progress=None) -> Path:
        return self._export_excel(papers, filename=filename, output_key="papers_selected", progress=progress)

    def export_cards_excel(self, papers: list[ScoredPaper], filename: str = "papers_cards.xlsx", progress=None) -> Path:
        return self._export_excel(papers, filename=filename, output_key="papers_cards", progress=progress)

    def _export_excel(self, papers: list[ScoredPaper], filename: str, output_key: str, progress=None) -> Path:
        path = self.output_dir / filename
        if progress:
            progress.log(f"Exporting {filename} ...")
        records = [self._to_flat_record(paper) for paper in papers]
        self._records_to_dataframe(records).to_excel(path, index=False)
        if progress:
            progress.log(f"Exported {filename} -> {path}")
        self.last_outputs[output_key] = str(path)
        return path

    def export_bibtex(self, papers: list[ScoredPaper], filename: str = "references.bib", progress=None) -> Path:
        path = self.output_dir / filename
        if progress:
            progress.log(f"Exporting {filename} ...")
        entries = [self._build_bibtex_entry(paper) for paper in papers]
        with path.open("w", encoding="utf-8") as file:
            file.write("\n\n".join(entry for entry in entries if entry))
        if progress:
            progress.log(f"Exported {filename} -> {path}")
        self.last_outputs["references_bib"] = str(path)
        return path

    def export_summary_markdown(
        self,
        papers: list[ScoredPaper],
        topic: str,
        filename: str = "summary_report.md",
        progress=None,
        scored_papers: list[ScoredPaper] | None = None,
        selected_papers: list[ScoredPaper] | None = None,
        cards_papers: list[ScoredPaper] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> Path:
        path = self.output_dir / filename
        if progress:
            progress.log(f"Exporting {filename} ...")

        all_scored = scored_papers or papers
        selected = selected_papers or papers
        cards = cards_papers or papers
        diagnostics = diagnostics or {}

        by_year = Counter(str(p.metadata.year or "unknown") for p in all_scored)
        by_journal = Counter((p.metadata.journal or "unknown") for p in all_scored)
        downloaded = [p for p in selected if p.metadata.raw.get("local_pdf_path")]
        with_pdf_url = [p for p in selected if p.metadata.pdf_url]

        rule_scores = [p.score.rule_score for p in all_scored]
        final_scores = [float(p.score.final_score or 0.0) for p in all_scored]
        llm_scores = [float((p.score.rubric_score.final_llm_score if p.score.rubric_score else p.score.llm_score) or 0.0) for p in all_scored]

        selection_diag = diagnostics.get("selection", {})
        cards_diag = diagnostics.get("cards", {})
        llm_candidate_diag = diagnostics.get("llm_candidate_selection", {})
        scoring_enhanced = diagnostics.get("scoring_enhanced", {})

        def _score_dist(values: list[float]) -> dict[str, int]:
            bins = {"0-39": 0, "40-59": 0, "60-74": 0, "75-84": 0, "85-100": 0}
            for score in values:
                if score < 40:
                    bins["0-39"] += 1
                elif score < 60:
                    bins["40-59"] += 1
                elif score < 75:
                    bins["60-74"] += 1
                elif score < 85:
                    bins["75-84"] += 1
                else:
                    bins["85-100"] += 1
            return bins

        top20 = sorted(all_scored, key=lambda x: (x.score.final_score or 0, x.metadata.citation_count or 0), reverse=True)[:20]
        top_core = [p for p in top20 if (p.score.final_decision or "") == "core"][:10]
        top_method = [p for p in top20 if "method" in ((p.score.rubric_score.decision if p.score.rubric_score else "") or "")][:10]
        top_data = [p for p in top20 if "data" in ((p.score.rubric_score.decision if p.score.rubric_score else "") or "")][:10]
        top_policy = [p for p in top20 if (p.score.rubric_score and (p.score.rubric_score.policy_relevance or 0) >= 70)][:10]
        low_conf_llm = [p for p in all_scored if p.score.rubric_score and (p.score.rubric_score.confidence or 0) < 50][:20]

        lines: list[str] = [
            "# LitBase-AI Summary Report",
            "",
            "## Research Topic",
            topic,
            "",
            "## Retrieval and Deduplication",
            f"- Retrieval total: {self.run_stats.get('raw_count', len(all_scored))}",
            f"- Deduplicated total: {len(all_scored)}",
            f"- Data source summary: {diagnostics.get('data_sources', {})}",
            "",
            "## OA Enrichment",
            f"- Unpaywall stats: {diagnostics.get('unpaywall', {})}",
            f"- Papers with OA PDF URL (selected): {len(with_pdf_url)}",
            "",
            "## Scoring Statistics",
            f"- Rule scored: {len(all_scored)}",
            f"- Rule score avg: {round(mean(rule_scores), 2) if rule_scores else 0.0}",
            f"- Final score avg: {round(mean(final_scores), 2) if final_scores else 0.0}",
            f"- Enhanced scoring stats: {scoring_enhanced}",
            "",
            "## Score Distributions",
            f"- Rule score distribution: {_score_dist(rule_scores)}",
            f"- LLM rubric score distribution: {_score_dist(llm_scores)}",
            f"- Final score distribution: {_score_dist(final_scores)}",
            "",
            "## Candidate Selection",
            f"- LLM candidate selection: {llm_candidate_diag}",
            f"- Selection strategy: {selection_diag}",
            f"- Selected papers count: {len(selected)}",
            "",
            "## Cards Generation",
            f"- Cards stats: {cards_diag}",
            f"- Cards generated: {len(cards)}",
            "",
        ]

        if not selected and cards_diag.get("fallback_cards_used", False):
            lines.extend(
                [
                    "No papers passed the selection threshold. Fallback cards were generated from the top scored papers.",
                    "",
                ]
            )

        lines.extend(
            [
                "## Top 20 Literature Cards",
                "",
                "| Rank | Final Score | Decision | Year | Title | Journal | DOI |",
                "|---:|---:|---|---:|---|---|---|",
            ]
        )
        for idx, paper in enumerate(top20, start=1):
            lines.append(
                f"| {idx} | {paper.score.final_score or 0:.2f} | {paper.score.final_decision or ''} | {paper.metadata.year or ''} "
                f"| {paper.metadata.title.replace('|', ' ')} | {(paper.metadata.journal or '').replace('|', ' ')} | {paper.metadata.doi or ''} |"
            )

        def _append_list_section(title: str, items: list[ScoredPaper]) -> None:
            lines.extend(["", f"## {title}"])
            if not items:
                lines.append("- None")
                return
            for paper in items[:10]:
                lines.append(
                    f"- [{paper.score.final_score or 0:.2f}] {paper.metadata.title} "
                    f"({paper.metadata.year or 'N/A'})"
                )

        _append_list_section("Top Core Papers", top_core)
        _append_list_section("Top Method Papers", top_method)
        _append_list_section("Top Data Papers", top_data)
        _append_list_section("Top Policy Papers", top_policy)
        _append_list_section("Papers with OA PDF", with_pdf_url[:20])
        _append_list_section("Papers downloaded", downloaded[:20])
        _append_list_section("Low-confidence LLM scoring list", low_conf_llm)

        lines.extend(
            [
                "",
                "## Human Feedback Summary",
                f"- Human feedback stats: {scoring_enhanced}",
                "",
                "## Suggested Reading Order",
                "1. Read core papers with highest final_score.",
                "2. Read method papers to align model and scenario design.",
                "3. Read data and policy papers for assumptions and discussion.",
            ]
        )

        lines.extend(["", "## Papers by Year"])
        lines.extend([f"- {year}: {count}" for year, count in sorted(by_year.items(), reverse=True)])
        lines.extend(["", "## Papers by Journal"])
        for journal, count in by_journal.most_common(20):
            lines.append(f"- {journal}: {count}")

        path.write_text("\n".join(lines), encoding="utf-8")
        if progress:
            progress.log(f"Exported {filename} -> {path}")
        self.last_outputs["summary_report"] = str(path)
        return path

    def _to_flat_record(self, paper: ScoredPaper) -> dict[str, Any]:
        metadata = paper.metadata
        score = paper.score
        rubric = score.rubric_score
        embedding = score.embedding_score

        evidence_items = score.evidence_items or []
        evidence_text = " | ".join(item.text for item in evidence_items if item.text)
        usable_for = rubric.usable_for if rubric else []
        labels = rubric.labels if rubric and rubric.labels else score.labels
        reason = (rubric.reason if rubric else None) or score.llm_reason
        download_trace = metadata.raw.get("download_trace") or []
        download_last = download_trace[-1] if download_trace else {}
        download_discovery = metadata.raw.get("download_discovery") or {}
        download_attempts = sum(
            1
            for item in download_trace
            if isinstance(item, dict)
            and item.get("status") in {"attempt", "downloaded", "request_failed", "http_error", "non_pdf"}
        )

        return {
            "paper_id": metadata.id,
            "final_score": score.final_score,
            "final_decision": score.final_decision,
            "final_confidence": score.final_confidence,
            "rule_score": score.rule_score,
            "llm_score": score.llm_score,
            "llm_final_score": (rubric.final_llm_score if rubric else None),
            "llm_confidence": (rubric.confidence if rubric else None),
            "topic_relevance": (rubric.topic_relevance if rubric else None),
            "object_relevance": (rubric.object_relevance if rubric else None),
            "method_relevance": (rubric.method_relevance if rubric else None),
            "data_relevance": (rubric.data_relevance if rubric else None),
            "novelty": (rubric.novelty if rubric else None),
            "citation_value": (rubric.citation_value if rubric else None),
            "writing_value": (rubric.writing_value if rubric else None),
            "policy_relevance": (rubric.policy_relevance if rubric else None),
            "evidence": evidence_text,
            "usable_for": "; ".join(usable_for),
            "reason": reason,
            "labels": "; ".join(labels),
            "human_label": score.human_label,
            "human_feedback_score": score.human_feedback_score,
            "embedding_combined_similarity": (embedding.combined_similarity if embedding else None),
            "title_similarity": (embedding.title_similarity if embedding else None),
            "abstract_similarity": (embedding.abstract_similarity if embedding else None),
            "keyword_similarity": (embedding.keyword_similarity if embedding else None),
            "title_match_score": score.title_match_score,
            "abstract_match_score": score.abstract_match_score,
            "keyword_match_score": score.keyword_match_score,
            "phrase_match_score": score.phrase_match_score,
            "fuzzy_match_score": score.fuzzy_match_score,
            "title": metadata.title,
            "year": metadata.year,
            "authors": ", ".join(metadata.authors),
            "journal": metadata.journal,
            "doi": metadata.doi,
            "citation_count": metadata.citation_count,
            "open_access_status": metadata.open_access_status,
            "pdf_url": metadata.pdf_url,
            "landing_page_url": metadata.landing_page_url,
            "llm_reason": score.llm_reason,
            "source_database": metadata.source_database,
            "paper_type": metadata.paper_type,
            "matched_queries": ", ".join([str(x) for x in (metadata.raw.get("matched_queries") or [])]),
            "keywords": ", ".join(metadata.keywords),
            "concepts": ", ".join([str(x) for x in (metadata.raw.get("concepts") or [])]),
            "topics": ", ".join([str(x) for x in (metadata.raw.get("topics") or [])]),
            "primary_topic": metadata.raw.get("primary_topic"),
            "source_count": metadata.raw.get("source_count", len(metadata.raw.get("merged_sources", [])) or 1),
            "card_path": metadata.raw.get("card_path"),
            "local_pdf_path": metadata.raw.get("local_pdf_path"),
            "download_source": metadata.raw.get("download_source"),
            "downloaded_from_url": metadata.raw.get("downloaded_from_url"),
            "download_candidate_count": len(download_discovery.get("candidates") or []),
            "download_discovery_resolved_doi": download_discovery.get("resolved_doi"),
            "download_attempts": download_attempts,
            "download_last_status": download_last.get("status"),
            "download_last_reason": download_last.get("reason"),
            "cnki_download_count": (metadata.raw.get("cnki") or {}).get("download_count"),
            "cnki_database": (metadata.raw.get("cnki") or {}).get("database"),
            "cnki_degree_type": (metadata.raw.get("cnki") or {}).get("degree_type"),
            "cnki_institution": (metadata.raw.get("cnki") or {}).get("institution"),
            "cnki_restricted": (metadata.raw.get("cnki") or {}).get("restricted"),
        }

    def _build_bibtex_entry(self, paper: ScoredPaper) -> str:
        metadata = paper.metadata
        title = self._escape_bibtex(metadata.title)
        author = " and ".join(metadata.authors) if metadata.authors else "Unknown"
        year = str(metadata.year or "n.d.")
        journal = self._escape_bibtex(metadata.journal or "")
        doi = metadata.doi or ""
        key = clean_filename(f"{extract_first_author(metadata.authors)}_{year}_{short_title(metadata.title, 3)}", 80)
        lines = [f"@article{{{key},"]
        lines.append(f"  title = {{{title}}},")
        lines.append(f"  author = {{{self._escape_bibtex(author)}}},")
        lines.append(f"  year = {{{year}}},")
        if journal:
            lines.append(f"  journal = {{{journal}}},")
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        if metadata.landing_page_url:
            lines.append(f"  url = {{{metadata.landing_page_url}}},")
        lines.append("}")
        return "\n".join(lines)

    def _escape_bibtex(self, text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    def _records_to_dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        columns = self._flat_columns()
        if not records:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(records)
        for column in columns:
            if column not in df.columns:
                df[column] = None
        return df[columns]

    def _flat_columns(self) -> list[str]:
        return [
            "paper_id",
            "final_score",
            "final_decision",
            "final_confidence",
            "rule_score",
            "llm_score",
            "llm_final_score",
            "llm_confidence",
            "topic_relevance",
            "object_relevance",
            "method_relevance",
            "data_relevance",
            "novelty",
            "citation_value",
            "writing_value",
            "policy_relevance",
            "evidence",
            "usable_for",
            "reason",
            "labels",
            "human_label",
            "human_feedback_score",
            "embedding_combined_similarity",
            "title_similarity",
            "abstract_similarity",
            "keyword_similarity",
            "title_match_score",
            "abstract_match_score",
            "keyword_match_score",
            "phrase_match_score",
            "fuzzy_match_score",
            "title",
            "year",
            "authors",
            "journal",
            "doi",
            "citation_count",
            "open_access_status",
            "pdf_url",
            "landing_page_url",
            "llm_reason",
            "source_database",
            "paper_type",
            "matched_queries",
            "keywords",
            "concepts",
            "topics",
            "primary_topic",
            "source_count",
            "card_path",
            "local_pdf_path",
            "download_source",
            "downloaded_from_url",
            "download_candidate_count",
            "download_discovery_resolved_doi",
            "download_attempts",
            "download_last_status",
            "download_last_reason",
            "cnki_download_count",
            "cnki_database",
            "cnki_degree_type",
            "cnki_institution",
            "cnki_restricted",
        ]
