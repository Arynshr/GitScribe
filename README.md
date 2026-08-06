# GitScribe

Stateful, LangGraph-powered PR description generator. Analyzes your git diff and commit history, pulls relevant past PRs from local memory, and generates a structured PR description via an LLM (provider-agnostic: groq/openai/anthropic/etc.).

## How it works

```
diff_parser → summarizer → risk_classifier → retriever → generator → (failure_router on error)
```

- **diff_parser** — `git diff origin/main...HEAD`, filtered by `.gitignore` / `ignore_patterns`
- **risk_classifier** — scores the diff 0–1; skips generation below `risk_classifier.trivial_threshold` and falls back to a template
- **retriever** — pulls past PRs by branch prefix from local memory, widens the search if the LLM judges them insufficiently relevant (bounded to 2 iterations)
- **generator** — builds the prompt from the diff summary + retrieved PRs + style, calls the LLM, parses a structured `PRDescription`
- **failure_router** — on LLM failure: retry same model → retry fallback model → template fallback

State flows through a single `GitScribeState` object across the graph. History persists to local SQLite (`Storage/gitscribe.db`, gitignored — see note below).

## Install

```bash
pip install -e .
```

This registers the `gitscribe` command via `[project.scripts]`.

## Setup

```bash
gitscribe init          # installs pre-push hook, prompts for your LLM API key -> writes .env
```

`.env` holds a single `API_KEY` regardless of provider — set `llm.provider` in `config.yaml` (groq/openai/anthropic/...) and the same key works. You can also just `export API_KEY=...` instead of using `.env`.

## Usage

```bash
gitscribe generate --style concise           # print description, save to memory
gitscribe generate --dry-run                 # print description, skip saving
gitscribe create-pr --style concise          # generate + open PR via `gh` (needs `gh auth login`)
```

`--style`: `default` | `concise` | `detailed`

## Config (`config.yaml`)

| Key | Default | Notes |
|---|---|---|
| `llm.provider` | `groq` | any provider `langchain.chat_models.init_chat_model` supports |
| `llm.model` / `llm.fallback_model` | — | primary / retry-fallback model |
| `llm.base_url` | `null` | optional, for self-hosted or custom-endpoint providers |
| `retrieval.min_prs` / `max_prs` | 3 / 10 | retrieval depth bounds |
| `risk_classifier.trivial_threshold` | 0.15 | below this, generation is skipped |
| `failure_handling.max_retries` | 2 | before falling back to template |

## Project layout

```
src/gitscribe/
├── cli.py                 # typer CLI: init, generate, create-pr
└── core/
    ├── diff_parser.py
    ├── summarizer.py
    ├── risk_classifier.py
    ├── retriever.py
    ├── generator.py
    ├── llm_client.py       # provider-agnostic chat-model factory
    ├── failure_router.py
    ├── graph.py             # LangGraph wiring
    ├── state.py             # GitScribeState
    ├── memory.py            # SQLite persistence, self-initializing schema
    ├── telemetry.py
    └── config_schema.py
```

## Known limitations

- Diff base is hardcoded to `origin/main...HEAD` (no `--base` flag yet)
- `Storage/gitscribe.db` is local, per-machine memory — gitignore it, don't commit it

## Status: v0.1 baseline

Diff extraction, risk-gated generation, adaptive retrieval, failure fallback chain, dry-run mode, and self-initializing local memory all working end to end.
