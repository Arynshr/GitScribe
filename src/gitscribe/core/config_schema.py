from typing import Literal

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "groq"
    model: str
    fallback_model: str
    base_url: str | None = None
    temperature: float = Field(ge=0.0, le=2.0, default=0.2)
    max_tokens: int = Field(gt=0, default=1000)


class RetrievalConfig(BaseModel):
    min_prs: int = Field(gt=0, default=3)
    max_prs: int = Field(gt=0, default=10)
    branch_prefix_match: bool = True


class RiskClassifierConfig(BaseModel):
    enabled: bool = True
    trivial_threshold: float = Field(ge=0.0, le=1.0, default=0.15)
    # spec §3.5 — weight given to structural signal (lint errors + blast
    # radius) vs the existing LLM risk score when blending. 0 preserves
    # today's LLM-only behavior exactly.
    structural_weight: float = Field(ge=0.0, le=1.0, default=0.3)


class FailureHandlingConfig(BaseModel):
    max_retries: int = Field(ge=0, default=2)
    retry_backoff_seconds: float = Field(ge=0, default=2)


class EmbeddingConfig(BaseModel):
    provider: str = "local"
    model: str = "all-MiniLM-L6-v2"


class MergePreviewConfig(BaseModel):
    escalate_below_confidence: Literal["high", "medium"] = "high"
    blast_radius_depth: int = Field(gt=0, default=2)
    worktree_cleanup: bool = True


class LintReviewConfig(BaseModel):
    enabled: bool = True


class AgenticReviewConfig(BaseModel):
    enabled: bool = True
    max_context_tokens: int = Field(gt=0, default=6000)
    hops: int = Field(gt=0, default=2)


class ReviewConfig(BaseModel):
    lint: LintReviewConfig = LintReviewConfig()
    agentic: AgenticReviewConfig = AgenticReviewConfig()


class GitScribeConfig(BaseModel):
    llm: LLMConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    risk_classifier: RiskClassifierConfig = RiskClassifierConfig()
    failure_handling: FailureHandlingConfig = FailureHandlingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    merge_preview: MergePreviewConfig = MergePreviewConfig()
    review: ReviewConfig = ReviewConfig()
    ignore_patterns: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict:
        return self.model_dump()
