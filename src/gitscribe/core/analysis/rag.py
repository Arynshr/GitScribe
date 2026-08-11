"""
core/analysis/rag.py
Query -> embed -> vector search -> graph-expand -> assembled
context. Replaces/augments today's branch-prefix PR retrieval with
code-aware context. 
Built as a public API from day one (not baked into
generator.py), per the platform decision — Stage 4's `gitscribe query`
CLI command and generator.py both call this module directly.
"""

from __future__ import annotations

from pydantic import BaseModel

from gitscribe.core.indexer.index_store import SearchResult, blast_radius, search


class ContextSnippet(BaseModel):
    symbol_id: int
    name: str
    file: str
    lineno: int
    relation: str  


class RAGContext(BaseModel):
    query: str
    snippets: list[ContextSnippet]

    def as_prompt_block(self) -> str:
        """Formats retrieved context for injection into generator.py's
        prompt template — plain text, no framework coupling.
        """
        lines = [f"# Relevant code context for: {self.query}"]
        for s in self.snippets:
            lines.append(f"- [{s.relation}] {s.name} ({s.file}:{s.lineno})")
        return "\n".join(lines)


def retrieve(query: str, cfg: dict, top_k: int = 5, expand_depth: int = 1) -> RAGContext:
    matches: list[SearchResult] = search(query, cfg, top_k=top_k)

    snippets: list[ContextSnippet] = []
    seen_ids: set[int] = set()

    for m in matches:
        if m.symbol_id in seen_ids:
            continue
        seen_ids.add(m.symbol_id)
        snippets.append(
            ContextSnippet(symbol_id=m.symbol_id, name=m.name, file=m.file, lineno=m.lineno, relation="match")
        )

        for related in blast_radius(m.symbol_id, max_depth=expand_depth):
            if related.symbol_id in seen_ids:
                continue
            seen_ids.add(related.symbol_id)
            snippets.append(
                ContextSnippet(
                    symbol_id=related.symbol_id,
                    name=related.name,
                    file=related.file,
                    lineno=0,  # blast_radius doesn't currently carry lineno; acceptable for context labeling
                    relation="caller" if related.depth > 0 else "callee",
                )
            )

    return RAGContext(query=query, snippets=snippets)
