"""Data stores — DuckDB for structured lookups, ChromaDB for vectors.

The dual-store pattern enables hybrid retrieval: exact code matches
come from DuckDB (fast, deterministic), while fuzzy natural-language
queries use ChromaDB's vector similarity search.
"""
