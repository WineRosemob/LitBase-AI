from __future__ import annotations

from pathlib import Path

from litbase_ai.models import ScoredPaper
from litbase_ai.utils.logging import get_logger
from litbase_ai.utils.text import clean_filename, extract_first_author, short_title


logger = get_logger(__name__)


class LiteratureCardGenerator:
    """Generate ranked markdown cards plus index.md."""

    def __init__(self, output_dir: Path, threshold: float = 0):
        self.output_dir = output_dir
        self.threshold = threshold
        self.card_dir = self.output_dir / "literature_cards"
        self.card_dir.mkdir(parents=True, exist_ok=True)
        self.last_generated_count = 0
        self.last_index_path: str | None = None

    def generate_batch(self, papers: list[ScoredPaper], progress=None) -> list[ScoredPaper]:
        ranked = sorted(
            papers,
            key=lambda p: ((p.score.final_score or 0), (p.metadata.citation_count or 0)),
            reverse=True,
        )
        self.last_generated_count = 0
        generated: list[ScoredPaper] = []
        task_id = progress.task("Generating literature cards", total=len(ranked)) if progress else None
        for idx, paper in enumerate(ranked, start=1):
            path = self.generate_one(paper, rank=idx)
            paper.metadata.raw["card_path"] = str(path)
            generated.append(paper)
            self.last_generated_count += 1
            if progress and task_id is not None:
                progress.update(task_id, advance=1, description=f"Card: {(paper.metadata.title or '')[:60]}")
        self._build_index(generated)
        if progress:
            progress.log(f"Generated {self.last_generated_count} literature cards")
        return generated

    def generate_one(self, paper: ScoredPaper, rank: int) -> Path:
        filename = self._build_filename(paper, rank=rank)
        path = self.card_dir / filename
        content = self._build_markdown(paper)
        path.write_text(content, encoding="utf-8")
        return path

    def _build_markdown(self, paper: ScoredPaper) -> str:
        metadata = paper.metadata
        score = paper.score
        rubric = score.rubric_score
        embedding = score.embedding_score
        llm_score = rubric.final_llm_score if rubric and rubric.final_llm_score is not None else score.llm_score
        llm_confidence = rubric.confidence if rubric else score.final_confidence
        usable_for = (rubric.usable_for if rubric else []) or []
        labels = (rubric.labels if rubric and rubric.labels else score.labels) or []
        reason = (rubric.reason if rubric else None) or score.llm_reason or "N/A"
        evidence_items = score.evidence_items or []

        evidence_lines = []
        for item in evidence_items:
            evidence_lines.append(f"- [{item.source}] {item.text}")
        if not evidence_lines:
            evidence_lines.append("- No strong evidence extracted from metadata.")

        usage_map = {
            "introduction": "可用于引言背景铺垫与问题提出。",
            "literature_review": "可用于文献综述中的研究脉络与代表性工作总结。",
            "method_comparison": "可用于方法比较与研究设计论证。",
            "data_source": "可用于数据来源、指标和变量说明。",
            "model_design": "可用于模型框架与参数设定参考。",
            "result_discussion": "可用于结果解释与对比讨论。",
            "policy_discussion": "可用于政策启示与治理讨论。",
            "research_gap": "可用于提炼研究空白与未来方向。",
            "not_useful": "当前主题相关性较弱，可作为边缘参考。",
        }
        use_lines = [f"- {k}: {usage_map.get(k, '可作为补充参考。')}" for k in usable_for]
        if not use_lines:
            decision = score.final_decision or "background"
            if decision in {"core", "important"}:
                use_lines = ["- literature_review: 可作为核心综述文献。", "- method_comparison: 可用于方法与结论对比。"]
            elif decision in {"method", "data"}:
                use_lines = ["- method_comparison: 可用于方法细节对照。", "- data_source: 可用于数据与变量参考。"]
            else:
                use_lines = ["- literature_review: 可作为补充背景参考。"]

        cnki_section: list[str] = []
        if "CNKI" in metadata.source_database:
            cnki = metadata.raw.get("cnki") or {}
            cnki_section = [
                "",
                "## CNKI Information",
                "- Source: CNKI",
                f"- Download Count: {cnki.get('download_count', 'N/A')}",
                f"- Database: {cnki.get('database', 'N/A')}",
                f"- Restricted: {cnki.get('restricted', True)}",
                f"- Landing Page: {metadata.landing_page_url or 'N/A'}",
            ]

        return "\n".join(
            [
                f"# {metadata.title}",
                "",
                "## Basic Information",
                f"- Authors: {', '.join(metadata.authors) if metadata.authors else 'N/A'}",
                f"- Year: {metadata.year or 'N/A'}",
                f"- Journal / Venue: {metadata.journal or 'N/A'}",
                f"- DOI: {metadata.doi or 'N/A'}",
                f"- Source database: {metadata.source_database}",
                f"- Citation count: {metadata.citation_count or 0}",
                f"- Open access status: {metadata.open_access_status or 'unknown'}",
                f"- PDF URL: {metadata.pdf_url or 'N/A'}",
                f"- Local PDF path: {metadata.raw.get('local_pdf_path', 'N/A')}",
                "",
                "## Scores",
                f"- Final Score: {score.final_score or 0:.2f}",
                f"- Final Decision: {score.final_decision or 'N/A'}",
                f"- Rule Score: {score.rule_score:.2f}",
                f"- LLM Rubric Score: {llm_score if llm_score is not None else 'N/A'}",
                f"- LLM Confidence: {llm_confidence if llm_confidence is not None else 'N/A'}",
                f"- Embedding Score: {embedding.combined_similarity if embedding and embedding.combined_similarity is not None else 'N/A'}",
                f"- Human Feedback: {score.human_feedback_score if score.human_feedback_score is not None else 'N/A'}",
                "",
                "## Rubric Evaluation",
                "",
                "| Dimension | Score |",
                "|---|---:|",
                f"| Topic Relevance | {rubric.topic_relevance if rubric else ''} |",
                f"| Object Relevance | {rubric.object_relevance if rubric else ''} |",
                f"| Method Relevance | {rubric.method_relevance if rubric else ''} |",
                f"| Data Relevance | {rubric.data_relevance if rubric else ''} |",
                f"| Novelty | {rubric.novelty if rubric else ''} |",
                f"| Citation Value | {rubric.citation_value if rubric else ''} |",
                f"| Writing Value | {rubric.writing_value if rubric else ''} |",
                f"| Policy Relevance | {rubric.policy_relevance if rubric else ''} |",
                f"| Confidence | {rubric.confidence if rubric else ''} |",
                "",
                "## Evidence",
                *evidence_lines,
                "",
                "## Abstract",
                metadata.abstract or "No abstract available.",
                "",
                "## AI Evaluation",
                f"- Decision: {(rubric.decision if rubric else None) or score.final_decision or 'N/A'}",
                f"- Usable for: {', '.join(usable_for) if usable_for else 'N/A'}",
                f"- Labels: {', '.join(labels) if labels else 'N/A'}",
                f"- Reason: {reason}",
                "",
                "## How to Use This Paper in My Research",
                *use_lines,
                "",
                "## Notes",
                "",
                "(manual notes)",
                *cnki_section,
            ]
        )

    def _build_filename(self, paper: ScoredPaper, rank: int) -> str:
        first_author = extract_first_author(paper.metadata.authors)
        title_chunk = short_title(paper.metadata.title, max_words=6)
        base = clean_filename(f"{rank:03d}_{first_author}_{title_chunk}", max_len=120)
        return f"{base}.md"

    def _build_index(self, papers: list[ScoredPaper]) -> Path:
        index_path = self.card_dir / "index.md"
        lines = [
            "# Literature Cards Index",
            "",
            "| Rank | Final Score | Decision | Year | Title | Journal | DOI | Card |",
            "|---:|---:|---|---:|---|---|---|---|",
        ]
        for idx, paper in enumerate(papers, start=1):
            card_path = paper.metadata.raw.get("card_path")
            card_name = Path(card_path).name if card_path else ""
            lines.append(
                f"| {idx} | {paper.score.final_score or 0:.2f} | {paper.score.final_decision or ''} | {paper.metadata.year or ''} "
                f"| {paper.metadata.title.replace('|', ' ')} | {(paper.metadata.journal or '').replace('|', ' ')} "
                f"| {paper.metadata.doi or ''} | [{card_name}]({card_name}) |"
            )
        index_path.write_text("\n".join(lines), encoding="utf-8")
        self.last_index_path = str(index_path)
        return index_path
