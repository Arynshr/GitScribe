from langchain_core.output_parsers import PydanticOutputParser 
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from core.state import GitScribeState

class PRDescription(BaseModel):
    title: str = Field(description="Concise  xPE title")
    summary: str = Field(description="1-2 sentence summary")
    changes: str = Field(description="Bullet list of key changes")
    testing: str = Field(description="How this was/should be tested")
    impact: str = Field(description="Risk/Impact notes")

PARSER = PydanticOutputParser(pydantic_object=PRDescription)
 
GEN_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior software engineer writing a Pull Request description.
 
Diff Summary:
{change_summary}
 
Commit Messages:
{commit_messages}
 
Past Style Examples (for tone/format only, do not copy content):
{past_prs}
 
{format_instructions}"""
)

def generator_node(state: GitScribeState, cfg: dict) -> dict:
    model_name = cfg["llm"]["model"] if not state.fallback_used else cfg["llm"]["fallback_model"]
    llm = ChatGroq(model= model_name, temperature=cfg["llm"]["temperature"])
    chain = GEN_PROMPT| llm | PARSER
    try:
        result: PRDescription = chain.invoke({
            "change_summary": state.change_summary,
            "commit_messages": state.commit_messages,
            "past_prs": [p.title for p in state.retrieved_prs],
            "format_instructions": PARSER.get_format_instructions(),
            })
        body = (
            f"## Summary\n{result.summary}\n\n"
            f"## Changes\n{result.changes}\n\n"
            f"## Testing\n{result.testing}\n\n"
            f"## Impact\n{result.impact}"
        )
        return {
            "pr_title": result.title,
            "pr_body": body,
            "status": "success",
            "last_error": None,
        }
    except Exception as e:
        return{
            "last_error": str(e),
            "attempt_count": state.attempt_count + 1,
            "status": "failed"
        }
