# Clinical Code Intelligence Agent — Handoff Document

**Last updated:** 2026-08-25
**Repo:** `/Workspace/Users/connor.fuglestad@ssmhealth.com/clinical-code-agent`
**GitHub:** `cfuglestad/clinical-code-agent`
**Branch:** `main`
**Git credential:** `1062364054167757` (gitHubOAuth, username `cfuglestad`)

---

## Project Purpose

A **tool-calling AI agent** that makes clinical codes (ICD-10-CM, ICD-10-PCS, MS-DRG, HCPCS) human-accessible. Part of a portfolio of healthcare AI projects aligned to **Vizient's GenAI strategy** ("making complex healthcare codes and terminology human-accessible and actionable using AI on Databricks").

The agent uses the **ReAct pattern**: classify user intent → call the right tool (exact lookup, semantic search, hierarchy traversal, DRG decomposition) → synthesize a human-readable explanation. Not a memorization system — the LLM routes and explains, while verified databases provide the knowledge.

---

## Architecture

```
[User Input: code or clinical question]
    → classify_node (determine intent: code_lookup / semantic_search / hierarchy / drg_explain)
    → execute_tool_node (call appropriate tool based on classification)
        Tools:
          - code_lookup (exact match via DuckDB)
          - semantic_search (vector similarity via ChromaDB)
          - hierarchy_traversal (parent/child code relationships)
          - drg_explainer (DRG → component diagnoses + procedures)
    → synthesize_node (LLM generates human-readable explanation from tool results)
    → END
```

---

## Tech Stack (all free or near-free)

| Component | Tool | Cost |
| --- | --- | --- |
| Agent framework | LangGraph | Free |
| LLM (routing + synthesis) | OpenAI gpt-4o-mini | ~$0.15/$0.60 per 1M tokens |
| Vector store | ChromaDB (local) | Free |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free |
| Structured store | DuckDB | Free |
| Code reference data | CMS public files (ICD-10-CM/PCS, MS-DRG) | Free (public domain) |
| Experiment tracking | MLflow (local) | Free |
| UI | Streamlit Cloud | Free tier |
| CI | GitHub Actions | Free tier |

---

## Completed Phases

### Phase 1: Project Scaffold ✅ (pushed)

**Package structure:** `src/clinical_code_agent/`

| File | Purpose |
| --- | --- |
| `__init__.py` | Version 0.1.0, package docstring |
| `config.py` | Pydantic `Settings` with `CODE_AGENT_` env prefix. Keys: `openai_api_key`, `openai_model` (gpt-4o-mini), `temperature` (0.0), `duckdb_path`, `chroma_persist_dir`, `semantic_search_top_k` (10), `similarity_threshold` (0.4), `max_tool_calls` (3), `abstention_threshold` (0.3), `mlflow_experiment_name` |
| `state.py` | `AgentState` TypedDict (total=False): `query`, `query_type`, `tool_results: list[dict]`, `synthesis`, `citations: list[str]`, `confidence: float`, `messages: list[str]`, `error` |
| `agent/__init__.py` | Exports `build_graph` |
| `agent/graph.py` | LangGraph `StateGraph(AgentState)` with 3 nodes: `_classify_query` (heuristic regex for now), `_execute_tool` (stub with RuntimeWarning), `_synthesize` (stub). Linear flow: classify → execute → synthesize → END. `_should_continue()` routing logic defined but not wired yet (for multi-tool loops in Phase 5). `_classify_with_heuristic()` detects ICD-10 patterns, DRG patterns, CPT/HCPCS, hierarchy keywords, defaults to semantic_search. |
| `tools/__init__.py` | Placeholder |
| `ingestion/__init__.py` | Placeholder |
| `store/__init__.py` | Placeholder |
| `evaluation/__init__.py` | Placeholder |

**Root files:**
- `main.py` — CLI with argparse: `python main.py "R65.20"`, `python main.py --interactive` (REPL mode). `run_query()`, `print_result()`, `interactive_mode()`, `cli()`.
- `pyproject.toml` — setuptools build, Python >=3.11, all deps listed, `[dev]` and `[app]` optional extras, ruff/pytest/mypy config.
- `requirements.txt` — flat deps for Streamlit Cloud compatibility.
- `.gitignore` — data/raw/, data/processed/, chroma_db/, *.duckdb, mlruns/, .env, __pycache__/.
- `README.md` — full project docs with architecture diagram, tech stack table, quick start.
- `tests/test_graph.py` — parametrized classification tests (14 cases covering ICD-10, DRG, CPT, hierarchy, semantic), graph build/invoke tests.

### Phase 2: CMS Data Ingestion ✅ (pushed)

| File | Purpose |
| --- | --- |
| `scripts/download_cms_data.py` | Downloads FY2025 ICD-10-CM, ICD-10-PCS, MS-DRG ZIP files from CMS.gov. `CMS_SOURCES` dict with URLs + extract patterns. Progress reporting, graceful failure with manual download instructions. |
| `src/clinical_code_agent/ingestion/parsers.py` | `CodeRecord` dataclass (code, description, code_system, chapter, category, is_header). Parsers: `parse_icd10cm()` (fixed-width: positions 6-12=code, 14=header flag, 16+=description), `parse_icd10pcs()` (7-char codes, section/body system extraction), `parse_msdrg_descriptions()` (regex + tab-delimited fallback). Helpers: `_format_icd10_code()` (dot after 3rd char), `_get_icd10cm_chapter()` (first-letter→chapter map), `_estimate_mdc()` (DRG number ranges→MDC). |
| `src/clinical_code_agent/store/duckdb_store.py` | `CodeStore` class with lazy DuckDB connection. Schema: `codes` (code, description, code_system, chapter, category, is_header — PK: code+code_system), `code_hierarchy` (parent_code, child_code, code_system, relationship_type), `drg_components` (drg_code, component_code, component_system, component_role). Methods: `load_records()`, `build_hierarchy()` (derives parent-child from ICD-10 structure via SQL), `lookup_code()` (normalized: strips dots, uppercases), `get_children()`, `search_description()` (ILIKE fallback), `stats` property. |
| `scripts/ingest_codes.py` | Orchestrator: finds raw files → parses → loads into DuckDB → builds hierarchy → reports stats. CLI with `--raw-dir`, `--db-path`, `--only` flags. |
| `tests/test_ingestion.py` | 15+ tests: code formatting, chapter mapping, parser output with synthetic CMS-format files, store CRUD (load, exact lookup, normalized lookup, case-insensitive, DRG lookup, not-found, description search, hierarchy build, stats). Uses `tmp_path` fixtures. |

**Data pipeline (run locally):**
```bash
python scripts/download_cms_data.py     # Downloads ~30MB from CMS.gov
python scripts/ingest_codes.py          # Produces data/processed/codes.duckdb (~150K+ codes)
```

---

## Remaining Phases

### Phase 3: Vector Index for Semantic Search (NEXT)

Build `scripts/build_vector_index.py` that:
1. Reads all code descriptions from DuckDB
2. Embeds them using sentence-transformers (all-MiniLM-L6-v2, runs locally)
3. Stores in ChromaDB with metadata (code, code_system, chapter)
4. Creates `src/clinical_code_agent/store/vector_store.py` — `VectorStore` class with `search(query, top_k)` method

This enables the "plain English → code" lookup (e.g., "heart valve replacement" → finds relevant ICD-10-PCS codes).

### Phase 4: Tool Definitions

Build 4 standalone tools in `src/clinical_code_agent/tools/`:
- `code_lookup.py` — wraps `CodeStore.lookup_code()`
- `semantic_search.py` — wraps `VectorStore.search()`
- `hierarchy.py` — wraps `CodeStore.get_children()` + parent traversal
- `drg_explainer.py` — DRG → component codes (uses `drg_components` table)

Each tool has a consistent interface: takes query string + optional params, returns `list[dict]` with `code`, `description`, `source`, `score` keys.

### Phase 5: LangGraph Agent with LLM Routing

Replace stubs with real LLM calls:
- `_classify_query` → OpenAI function calling for intent classification (structured output)
- `_execute_tool` → dispatch to real tools based on classification
- `_synthesize` → OpenAI gpt-4o-mini generates contextual explanation from tool results
- Wire conditional routing: if confidence < threshold, try a different tool (multi-hop)

### Phase 6: Evaluation Framework

- Gold-standard Q&A pairs in `evaluation/` (JSON)
- Metrics: tool selection accuracy, answer factual grounding, end-to-end correctness
- MLflow local tracking of evaluation runs

### Phase 7: Streamlit UI

- `app/streamlit_app.py` — interactive demo
- Input: code or natural language question
- Output: explanation + hierarchy visualization + citations
- Deploy on Streamlit Cloud (free tier)

---

## Key Design Decisions

1. **Hybrid retrieval (DuckDB + ChromaDB):** Exact codes use SQL (deterministic, fast). Natural language uses vectors (fuzzy, semantic). The agent decides which to use.
2. **Stub-first development:** All nodes emit `RuntimeWarning` when using stubs. Tests pass in stub mode. Real implementations replace stubs phase by phase.
3. **Normalized lookup:** `lookup_code()` strips dots and uppercases before matching — so `r65.20`, `R6520`, and `R65.20` all find the same record.
4. **Hierarchy derivation:** Parent-child relationships are COMPUTED from code structure (not stored in CMS files). 3-char ICD-10-CM categories → their 4-7 char children.
5. **Free-tier constraint:** Everything runs locally or on free tiers. OpenAI gpt-4o-mini for LLM calls (~$0.0002 per query). No Databricks endpoints needed.

---

## Running Tests

```bash
cd /Workspace/Users/connor.fuglestad@ssmhealth.com/clinical-code-agent
pip install -e ".[dev]"
pytest tests/ -p no:cacheprovider
```

Note: Same as resume-optimization — copy to /tmp first if `__pycache__` write issues occur on workspace filesystem.

---

## CMS Data Source URLs (FY2025)

- ICD-10-CM: `https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip`
- ICD-10-PCS: `https://www.cms.gov/files/zip/2025-icd-10-pcs-codes-file.zip`
- MS-DRG: `https://www.cms.gov/files/zip/icd-10-ms-drg-definitions-manual-files-v42.zip`

All public domain, no license required.

---

## User Preferences (from .assistant_instructions.md)

- **Study mode:** Use `/deepdive <topic>` to teach concepts at interview depth as they arise in code
- **Stakeholder mode:** Use `/explain <topic>` for non-technical-but-intelligent audience translation
- **Decomposition:** Break multi-step tasks into explicit sub-tasks with `manageTodoList`
- **Self-reflection:** Review outputs for correctness/completeness before presenting
- **Free tools only:** All projects use free tiers + cheap OpenAI API for personal portfolio
- **Healthcare-specific:** Projects aligned to Vizient's data & digital strategy
- **Frontier mode:** Requested but MCP endpoint unavailable — use direct build with explanations

---

## Related Portfolio Projects

| Project | Skills Demonstrated | Status |
| --- | --- | --- |
| `resume-optimization` | QLoRA fine-tuning, LangGraph, DSPy, MLflow, model serving | Complete (deployed on Streamlit Cloud) |
| `clinical-guideline-qanda` | RAG, ChromaDB, citation grounding, abstention, LangGraph | Complete |
| `doc-drift-analyzer` | Semantic similarity, protocol-based DI, document parsing, evaluation | Complete |
| `clinical-code-agent` | Tool-calling agents, hybrid retrieval, ontology traversal | **In progress (Phase 2 complete)** |

---

## Resume Instructions

To continue from where this left off:
1. Load the repo at `/Workspace/Users/connor.fuglestad@ssmhealth.com/clinical-code-agent`
2. Next action: Build Phase 3 (vector index for semantic search)
3. The user has NOT yet run the download/ingest scripts locally — the DuckDB is not populated. Phase 3 code should be written with self-contained tests (using fixtures) so it can be built and pushed without needing real CMS data present.
4. After Phase 3, move to Phase 4 (tool definitions) — this is where the stubs in `agent/graph.py` get replaced with real tool calls.
5. Use `/deepdive` to explain new concepts as they arise (embedding-based retrieval for Phase 3, function calling schemas for Phase 5).
