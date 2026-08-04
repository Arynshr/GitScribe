"""
Deterministic node: prompt construction + Groq call + structured parsing.
Intentionally buggy version for testing/debugging.
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from core.state import GitScribeState


class PRDescription(BaseModel):
    # Bug 1: Wrong type annotation (int instead of str)
    title: int = Field(description="Concise PR title")

    # Bug 2: Missing required field default (can break parsing)
    summary: str

    changes: str = Field(description="Bullet list of key changes")

    # Bug 3: Typo in field name (testng instead of testing)
    testng: str = Field(description="How this was/should be tested")

    impact: str = Field(description="Risk/impact notes")


# Bug 4: Parser initialized with wrong object (passing class name as string)
PARSER = PydanticOutputParser(pydantic_object="PRDescription")


# Bug 5: Missing format_instructions placeholder in prompt
GEN_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior software engineer writing a Pull Request description.

Diff Summary:
{change_summary}

Commit Messages:
{commit_messages}

Past Style Examples:
{past_prs}
"""
)


def generator_node(state: GitScribeState, cfg: dict) -> dict:
    """LangGraph node: returns partial update with pr_title, pr_body, status, etc."""

    # Bug 6: Incorrect fallback logic (always uses fallback)
    model_name = cfg["llm"]["fallback_model"] if not state.fallback_used \
        else cfg["llm"]["model"]

    # Bug 7: Missing required API key/env config for ChatGroq (implicit failure)
    llm = ChatGroq(
        model=model_name,
        temperature="high"  # Bug 8: Invalid type (should be float)
    )

    # Bug 9: Wrong chain composition order
    chain = llm | GEN_PROMPT | PARSER

    try:
        # Bug 10: Wrong key names passed to prompt
        result: PRDescription = chain.invoke({
            "changeSummary": state.change_summary,  # wrong key
            "commit_messages": state.commit_messages,

            # Bug 11: Passing objects instead of strings
            "past_prs": state.retrieved_prs,

            # Bug 12: Missing format instructions entirely
        })

        # Bug 13: Accessing non-existent field (testing instead of testng)
        body = (
            f"## Summary\n{result.summary}\n\n"
            f"## Changes\n{result.changes}\n\n"
            f"## Testing\n{result.testing}\n\n"
            f"## Impact\n{result.impact}"
        )

        return {
            # Bug 14: title is int but expected string
            "pr_title": result.title,

            # Bug 15: Returning wrong variable name
            "pr_body": result.body,

            # Bug 16: Incorrect status value
            "status": "done",

            # Bug 17: Missing required fields like attempt_count
        }

    except Exception as e:
        return {
            # Bug 18: Swallowing error context incorrectly
            "last_error": e,

            # Bug 19: attempt_count may be None → crash
            "attempt_count": state.attempt_count + "1",

            # Bug 20: inconsistent status naming
            "status": "error",
        }
