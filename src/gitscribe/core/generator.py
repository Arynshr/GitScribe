"""
Deterministic node: prompt construction + Groq call + structured parsing.
This is the README's core 'chain' — no agent behavior, single call.
"""
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from core.state import GitScribeState
from core.telemetry import timed_llm_call


class PRDescription(BaseModel):
    title: str = Field(description="Concise PR title")
    summary: str = Field(description="1-2 sentence summary")
    changes: str = Field(description="Bullet list of key changes")
    testing: str = Field(description="How this was/should be tested")
    impact: str = Field(description="Risk/impact notes")


PARSER = PydanticOutputParser(pydantic_object=PRDescription)

STYLE_INSTRUCTIONS = {
    "default": "Write in a clear, professional tone with complete sentences.",
    "concise": "Be terse. Use short bullet fragments, not full sentences. Omit any section with nothing notable to say.",
    "detailed": "Be thorough. Explain reasoning behind changes and call out edge cases considered.",
}

GEN_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior software engineer writing a Pull Request description.

Style: {style_instruction}

Diff Summary:
{change_summary}

Commit Messages:
{commit_messages}

Past Style Examples (for tone/format only, do not copy content):
{past_prs}

{format_instructions}"""
)


def generator_node(state: GitScribeState, cfg: dict) -> dict:
    """LangGraph node: returns partial update with pr_title, pr_body, status, etc."""
    model_name = cfg["llm"]["model"] if not state.fallback_used \
        else cfg["llm"]["fallback_model"]
    style_instruction = STYLE_INSTRUCTIONS.get(state.style, STYLE_INSTRUCTIONS["default"])

    llm = ChatGroq(model=model_name, temperature=cfg["llm"]["temperature"])
    prompt_value = GEN_PROMPT.invoke({
        "style_instruction": style_instruction,
        "change_summary": state.change_summary,
        "commit_messages": state.commit_messages,
        "past_prs": [p.title for p in state.retrieved_prs],
        "format_instructions": PARSER.get_format_instructions(),
    })

    with timed_llm_call("generator", model_name) as ctx:
        ai_msg = llm.invoke(prompt_value)
        ctx["usage"] = getattr(ai_msg, "usage_metadata", None)

    try:
        result: PRDescription = PARSER.invoke(ai_msg)
        body = (
            f" Summary\n{result.summary}\n\n"
            f" Changes\n{result.changes}\n\n"
            f" Testing\n{result.testing}\n\n"
            f" Impact\n{result.impact}"
        )
        return {
            "pr_title": result.title,
            "pr_body": body,
            "status": "success",
            "last_error": None,
        }
    except Exception as e:
        return {
            "last_error": str(e),
            "attempt_count": state.attempt_count + 1,
            "status": "failed",
        }
