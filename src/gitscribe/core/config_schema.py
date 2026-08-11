"""
Validates config.yaml on load. A malformed config now fails immediately
with a clear message instead of a raw KeyError three nodes deep.
"""
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


class FailureHandlingConfig(BaseModel):
    max_retries: int = Field(ge=0, default=2)
    retry_backoff_seconds: float = Field(ge=0, default=2)


class GitScribeConfig(BaseModel):
    llm: LLMConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    risk_classifier: RiskClassifierConfig = RiskClassifierConfig()
    failure_handling: FailureHandlingConfig = FailureHandlingConfig()
    ignore_patterns: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict:
        """Node functions currently index cfg["section"]["key"] - keep that working."""
        return self.model_dump()
    
class EmbeddingConfig(BaseModel):
    provider: str = "local" 
    model: str = "all-MiniLM-L6-v2"
    
