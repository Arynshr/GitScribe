"""
Shared state object passed between all LangGraph nodes.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RetrievedPR(BaseModel):
    id: int
    branch: str
    title: str
    body: str


class GitScribeState(BaseModel):
    # --- inputs ---
    branch_name: str = ""
    raw_diff: str = ""
    commit_messages: list[str] = Field(default_factory=list)

    # --- diff intelligence (deterministic) ---
    files_changed: list[str] = Field(default_factory=list)
    change_summary: list[str] = Field(default_factory=list)

    # --- risk classification (agentic node) ---
    risk_score: float = 1.0
    risk_reasoning: str = ""
    skip_generation: bool = False

    # --- retrieval (agentic node) ---
    retrieved_prs: list[RetrievedPR] = Field(default_factory=list)
    retrieval_depth_used: int = 0
    retrieval_stopped_reason: str = ""

    # --- generation (deterministic chain) ---
    prompt: str = ""
    pr_title: str = ""
    pr_body: str = ""

    # --- failure handling (agentic router) ---
    attempt_count: int = 0
    last_error: str | None = None
    failure_type: Literal["rate_limit", "bad_output", "timeout", "none"] | None = None
    fallback_used: bool = False

    # --- output ---
    final_output: str = ""
    status: Literal["pending", "success", "skipped", "failed"] = "pending"
