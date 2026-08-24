# GitScribe

#### A local-first, BYOK code intelligence CLI — it writes your PR descriptions, indexes your codebase into a queryable symbol graph, and enforces commit/merge hygiene, all from git hooks running in your own repo, on your own LLM key.
---

## Features

### PR Generation
- Analyzes `git diff` + commit history, retrieves relevant past PRs from local memory, and generates a structured PR description (title, summary, changes, testing notes, impact) via your configured LLM
- **Risk-gated**: trivial diffs (formatting, version bumps) skip the LLM call entirely and use a template — saves API cost and noise
- **Adaptive retrieval**: pulls past PRs by branch prefix, widens the search (bounded) if the LLM judges the initial context insufficient
- **Failure-safe generation**: retry same model → retry fallback model → template fallback, so a provider outage never blocks a push
- Three output styles: `default`, `concise`, `detailed`
- `gitscribe create-pr` opens the PR directly via GitHub CLI (`gh`)

### Codebase Indexing & Analysis
- `gitscribe index` — parses your Python source (stdlib `ast`) into a symbol table, resolves call/import edges into a graph, and computes local embeddings (`sentence-transformers`, no API cost) — all in SQLite, full rebuild per run
- `gitscribe query "<question>"` — RAG: embeds your question, vector-searches the symbol graph, expands via blast-radius, and either returns raw grounded context (`--raw`/`--json`, no LLM call) or synthesizes a cited natural-language answer
- `gitscribe graph <symbol>` — blast-radius / dependency view: who calls this, what does it call, to what depth
- `gitscribe lint` — wraps `ruff`, normalizes findings into a shared schema, exits non-zero on error-severity findings (CI-friendly)

### Git Hygiene Automation
Installed via `gitscribe init`, these run as real hooks — no separate process to remember:
- **`pre-push`** — flags high-risk diffs (configurable hard block), auto-opens a PR if the branch has an upstream
- **`pre-merge-commit`** — risk-scores the incoming merge before it lands
- **`post-merge`** — records the merge to PR memory; can auto-tag using Conventional Commits-derived semver bumps
- **`commit-msg`** — enforces Conventional Commits format at commit time

---

## Installation

```bash
git clone https://github.com/Arynshr/GitScribe.git
cd GitScribe
pip install -e .
```

Requires **Python 3.13+**. Registers the `gitscribe` command via `[project.scripts]`.

---

## Quickstart

```bash
# from your project repo root
gitscribe init                        # installs git hooks, prompts for your LLM API key -> .env

# PR generation
gitscribe generate --style concise    # print a PR description, save to local memory
gitscribe create-pr                   # generate + open a PR via `gh`

# codebase intelligence
gitscribe index                       # build the symbol graph + embeddings
gitscribe query "how does risk scoring work?"
gitscribe graph risk_classifier_node  # blast radius for a symbol
gitscribe lint                        # ruff findings, normalized
```

`.env` holds a single `API_KEY` regardless of provider — set `llm.provider` in `config.yaml` and the same key is reused across `generate`, `query`, and every LLM-backed node. You can also just `export API_KEY=...`.

---

## How it works

### PR generation pipeline (LangGraph DAG)

```
diff_parser → summarizer → risk_classifier → retriever → generator → (failure_router on error)
```

| Node | Responsibility |
|---|---|
| `diff_parser` | `git diff origin/main...HEAD`, filtered by `.gitignore` / `ignore_patterns` |
| `summarizer` | Condenses the raw diff into a per-file change summary |
| `risk_classifier` | Scores the diff 0.0–1.0; below `trivial_threshold` → skip generation, use template |
| `retriever` | Pulls past PRs from local memory by branch prefix; widens (bounded, ≤2 iterations) if the LLM judges results insufficient |
| `generator` | Builds the prompt from diff summary + retrieved PRs + style, calls the LLM, parses a structured `PRDescription` |
| `failure_router` | On LLM failure: retry same model → retry fallback model → template fallback |

State flows through a single `GitScribeState` object across the graph. PR history persists to local SQLite (`Storage/gitscribe.db`, gitignored).

### Indexing & analysis pipeline

```
parser (ast) → graph_builder (call/import edges) → embedder (local) → index_store (public API)
```

`index_store.py` exposes `search()` and `blast_radius()` as the stable public API — both `gitscribe query`/`graph` and internal callers (e.g. `analysis/rag.py`) go through it rather than touching SQLite directly. `analysis/rag.py` composes `search()` + `blast_radius()` into ranked, deduplicated context blocks for the `query` command. `analysis/linter.py` wraps `ruff` and `analysis/semantic_checks.py` runs graph queries (cycle detection, dead-code reachability, fan-in/out) over the same index.

---

## CLI reference

| Command | Purpose |
|---|---|
| `gitscribe init` | Installs `pre-push`, `pre-merge-commit`, `post-merge`, `commit-msg` hooks; prompts for `API_KEY` |
| `gitscribe generate [--style] [--dry-run]` | Generate a PR description for the current branch's diff |
| `gitscribe create-pr [--style] [--push/--no-push]` | Generate + open a PR via `gh` |
| `gitscribe index [--repo-root] [--json]` | Build/refresh the symbol graph + embeddings (full rebuild) |
| `gitscribe query <text> [--top-k] [--raw] [--json]` | RAG-grounded question answering over the indexed codebase |
| `gitscribe graph <symbol> [--depth] [--json]` | Blast-radius / dependency view for a symbol (exact or fuzzy match) |
| `gitscribe lint [--repo-root] [--json] [--fails-on-error]` | Ruff findings, normalized, with a severity score |

Hook-only subcommands (`pre-push`, `merge-check`, `post-merge`, `commit-msg`, `merge-conflict`) are invoked by the installed git hooks and generally aren't run manually.

---

## Project layout

```
src/gitscribe/
├── cli.py                        # typer CLI: all commands above
├── console.py                    # shared output formatting
├── hooks/                        # installable shell hooks (pre-push.sh, post-merge.sh, ...)
└── core/
    ├── llm_client.py             # provider-agnostic chat-model factory (BYOK)
    ├── diff_parser.py            # git diff extraction + .gitignore-aware filtering
    ├── summarizer.py             # diff -> per-file change summary
    ├── risk_classifier.py        # 0-1 risk scoring, trivial-diff gating
    ├── retriever.py              # adaptive past-PR retrieval from memory
    ├── generator.py              # structured PRDescription generation
    ├── failure_router.py         # retry -> fallback model -> template chain
    ├── graph.py                  # LangGraph DAG wiring
    ├── state.py                  # GitScribeState
    ├── memory.py                 # SQLite PR history, self-initializing schema
    ├── telemetry.py              # LLM call logging
    ├── hooks.py                  # hook-command business logic (risk caching, semver bump, conflict detection)
    ├── config_schema.py          # pydantic-validated config.yaml
    ├── indexer/
    │   ├── parser.py             # stdlib `ast` symbol extraction
    │   ├── graph_builder.py      # call/import edge resolution
    │   ├── embedder.py           # local sentence-transformers embeddings
    │   ├── index_store.py        # public API: search(), blast_radius()
    │   └── schema.sql            # symbols / edges / embeddings tables
    └── analysis/
        ├── rag.py                # query -> embed -> search -> graph-expand -> context
        ├── linter.py             # ruff wrapper, normalized findings
        └── semantic_checks.py    # cycles, dead code, fan-in/out over the graph
```

---

## Testing

```bash
pytest
```

Test suite covers pipeline nodes (`test_generator`, `test_retriever`, `test_risk_classifier`, `test_failure_router`, `test_graph`), config validation (`test_config_schema`), and the indexing/analysis layer (`test_parser`, `test_graph_builder`, `test_embedder`, `test_index_store`, `test_rag`, `test_linter`, `test_semantic_checks`, `test_memory`).

---

## License

MIT — see [LICENSE](LICENSE).
