from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PaperMetadata(BaseModel):
    """Unified metadata schema for a research paper."""

    id: str
    title: str
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    journal: str | None = None
    publisher: str | None = None
    citation_count: int | None = None
    source_database: str
    open_access_status: str | None = None
    pdf_url: str | None = None
    landing_page_url: str | None = None
    paper_type: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class RubricScore(BaseModel):
    """Structured multi-dimensional rubric score from LLM."""

    topic_relevance: float | None = None
    object_relevance: float | None = None
    method_relevance: float | None = None
    data_relevance: float | None = None
    novelty: float | None = None
    citation_value: float | None = None
    writing_value: float | None = None
    policy_relevance: float | None = None
    confidence: float | None = None
    final_llm_score: float | None = None
    decision: str | None = None
    reason: str | None = None
    evidence: list[str] = Field(default_factory=list)
    usable_for: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class EmbeddingScore(BaseModel):
    """Optional embedding-based similarity score."""

    model_config = ConfigDict(protected_namespaces=())

    title_similarity: float | None = None
    abstract_similarity: float | None = None
    keyword_similarity: float | None = None
    combined_similarity: float | None = None
    model_name: str | None = None
    enabled: bool = False


class EvidenceItem(BaseModel):
    """Extracted evidence snippet from metadata or LLM output."""

    source: str
    text: str
    relevance: str | None = None
    confidence: float | None = None


class PaperScore(BaseModel):
    """Score breakdown for one paper."""

    paper_id: str
    relevance_score: float
    year_score: float
    citation_score: float
    journal_score: float
    access_score: float
    type_score: float
    rule_score: float
    llm_score: float | None = None
    final_score: float | None = None
    llm_reason: str | None = None
    labels: list[str] = Field(default_factory=list)
    title_match_score: float | None = None
    abstract_match_score: float | None = None
    keyword_match_score: float | None = None
    phrase_match_score: float | None = None
    fuzzy_match_score: float | None = None
    embedding_score: EmbeddingScore | None = None
    rubric_score: RubricScore | None = None
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    human_label: str | None = None
    human_feedback_score: float | None = None
    final_decision: str | None = None
    final_confidence: float | None = None


class ScoredPaper(BaseModel):
    """Paired metadata and score."""

    metadata: PaperMetadata
    score: PaperScore


class ExpandedQuery(BaseModel):
    """Expanded search query representation for multi-source retrieval."""

    original_topic: str
    detected_language: str | None = None
    english_topic: str | None = None
    chinese_topic: str | None = None
    english_keywords: list[str] = Field(default_factory=list)
    chinese_keywords: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    related_terms: list[str] = Field(default_factory=list)
    boolean_queries: list[str] = Field(default_factory=list)
    loose_queries: list[str] = Field(default_factory=list)
    phrase_queries: list[str] = Field(default_factory=list)
