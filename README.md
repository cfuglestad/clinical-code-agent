# Clinical Code Intelligence Agent

A tool-calling AI agent that makes ICD-10, MS-DRG, CPT, and HCPCS codes human-accessible. Ask a clinical question or enter a code — the agent resolves, explains, and connects it to the broader classification hierarchy.

## What it does

1. **Code → English:** Input `R65.20`, get a full explanation with DRG impact and clinical context
2. **English → Codes:** Input "heart valve replacement", get ranked relevant procedure codes
3. **DRG Decomposition:** Input `MS-DRG 470`, get the diagnosis + procedure codes that trigger it
4. **Hierarchy Navigation:** Input "what's under R65?", get subcategories with descriptions
5. **Coding Pattern Insight:** Ask "why would sepsis coding rates be high?" — get the clinical AND financial context

## Architecture

ReAct-style tool-calling agent built on LangGraph:

```
[User Query]
    → classify_node (determine intent: code lookup / semantic / hierarchy / DRG)
    → execute_tool_node (call appropriate tool)
        Tools:
          - code_lookup (exact match via DuckDB)
          - semantic_search (vector similarity via ChromaDB)
          - hierarchy_traversal (parent/child code relationships)
          - drg_explainer (DRG → component codes)
    → synthesize_node (LLM generates human-readable explanation)
    → END
```

## Tech stack

| Component | Tool | Cost |
| --- | --- | --- |
| Agent framework | LangGraph | Free |
| LLM | OpenAI gpt-4o-mini | ~$0.15/1M input tokens |
| Vector store | ChromaDB (local) | Free |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free |
| Structured store | DuckDB | Free |
| Code reference data | CMS public files | Free (public domain) |
| Experiment tracking | MLflow (local) | Free |
| UI | Streamlit Cloud | Free tier |

## Data sources

All publicly available from CMS.gov, no PHI:
- ICD-10-CM (diagnosis codes, ~72K codes)
- ICD-10-PCS (procedure codes, ~78K codes)
- MS-DRG definitions and grouper logic
- HCPCS Level II codes
- GEM crosswalks (ICD-9 ↔ ICD-10)

## Quick start

```bash
git clone https://github.com/cfuglestad/clinical-code-agent.git
cd clinical-code-agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Set your OpenAI API key:
```bash
export CODE_AGENT_OPENAI_API_KEY="sk-..."
```

Ingest CMS data and start:
```bash
python scripts/download_cms_data.py
python scripts/ingest_codes.py
python main.py --interactive
```

## Project structure

```
src/clinical_code_agent/
    agent/          # LangGraph state machine + routing
    tools/          # code_lookup, semantic_search, hierarchy, drg_explainer
    ingestion/      # CMS file parsers
    store/          # DuckDB (structured) + ChromaDB (vector)
    evaluation/     # Agent correctness metrics
app/                # Streamlit demo
scripts/            # Data download + ingestion
data/
    raw/            # CMS source files (gitignored)
    processed/      # Parsed + indexed (gitignored)
evaluation/         # Gold-standard Q&A pairs
tests/              # pytest suite
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

## License

MIT
