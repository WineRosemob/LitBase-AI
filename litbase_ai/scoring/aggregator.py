from __future__ import annotations

from typing import Any

from litbase_ai.models import ScoredPaper


class FinalScoreAggregator:
    """Aggregate multi-source scores into final_score/final_decision."""

    def __init__(self, weights: dict | None = None, decision_thresholds: dict | None = None):
        self.weights = weights or {
            "rule_score": 0.40,
            "llm_rubric_score": 0.45,
            "embedding_score": 0.10,
            "human_feedback_score": 0.05,
        }
        self.decision_thresholds = decision_thresholds or {
            "core": 85,
            "important": 75,
            "background": 60,
            "peripheral": 40,
        }
        self.last_stats: dict[str, int] = {"final_score_aggregated": 0}

    def aggregate(self, paper: ScoredPaper) -> ScoredPaper:
        components: dict[str, float] = {"rule_score": float(paper.score.rule_score)}
        weights: dict[str, float] = {"rule_score": float(self.weights.get("rule_score", 0.40))}

        rubric = paper.score.rubric_score
        if rubric and rubric.final_llm_score is not None:
            llm_weight = float(self.weights.get("llm_rubric_score", 0.45))
            confidence = rubric.confidence if rubric.confidence is not None else 60.0
            if confidence < 50:
                llm_weight = llm_weight * 0.6
            components["llm_rubric_score"] = float(rubric.final_llm_score)
            weights["llm_rubric_score"] = llm_weight

        embedding = paper.score.embedding_score
        if embedding and embedding.combined_similarity is not None:
            components["embedding_score"] = float(embedding.combined_similarity)
            weights["embedding_score"] = float(self.weights.get("embedding_score", 0.10))

        if paper.score.human_feedback_score is not None:
            components["human_feedback_score"] = float(paper.score.human_feedback_score)
            weights["human_feedback_score"] = float(self.weights.get("human_feedback_score", 0.05))

        available_weight = sum(weights.values())
        if available_weight <= 0:
            final_score = float(paper.score.rule_score)
        else:
            final_score = 0.0
            for key, value in components.items():
                normalized_weight = weights[key] / available_weight
                final_score += normalized_weight * value
        final_score = round(max(0.0, min(100.0, final_score)), 2)

        paper.score.final_score = final_score
        base_decision = self._decision_from_score(final_score)

        rubric_decision = (paper.score.rubric_score.decision if paper.score.rubric_score else None) or None
        rubric_confidence = (paper.score.rubric_score.confidence if paper.score.rubric_score else None) or None
        final_decision = base_decision
        if rubric_decision and rubric_confidence is not None and rubric_confidence >= 70:
            final_decision = rubric_decision

        paper.score.final_decision = final_decision
        if rubric_confidence is not None:
            paper.score.final_confidence = rubric_confidence
        elif paper.score.final_confidence is None:
            paper.score.final_confidence = round(min(95.0, max(35.0, 40.0 + final_score * 0.6)), 2)
        self.last_stats["final_score_aggregated"] += 1
        return paper

    def aggregate_batch(self, papers: list[ScoredPaper]) -> list[ScoredPaper]:
        self.last_stats = {"final_score_aggregated": 0}
        return [self.aggregate(paper) for paper in papers]

    def _decision_from_score(self, score: float) -> str:
        core = float(self.decision_thresholds.get("core", 85))
        important = float(self.decision_thresholds.get("important", 75))
        background = float(self.decision_thresholds.get("background", 60))
        peripheral = float(self.decision_thresholds.get("peripheral", 40))
        if score >= core:
            return "core"
        if score >= important:
            return "important"
        if score >= background:
            return "background"
        if score >= peripheral:
            return "peripheral"
        return "irrelevant"
