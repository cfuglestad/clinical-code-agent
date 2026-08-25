"""ChromaDB vector store for semantic code search.

This enables the "plain English → clinical code" retrieval path.
When a user asks "what's the code for heart valve replacement?", the
agent embeds that question and finds the closest matching code
descriptions via cosine similarity.

Design choices:
- Uses sentence-transformers all-MiniLM-L6-v2 (384-dim, runs locally, free)
- ChromaDB handles persistence and ANN indexing
- Metadata stored with each vector: code, code_system, chapter
- Returns results in the same dict format as CodeStore for tool uniformity
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from clinical_code_agent.config import settings

logger = logging.getLogger(__name__)

# Collection name used across the project
COLLECTION_NAME = "clinical_codes"

# Embedding model — small, fast, runs on CPU
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorStore:
    """ChromaDB-backed vector store for semantic clinical code search.

    Usage:
        store = VectorStore()            # uses config defaults
        store = VectorStore(persist_dir) # explicit path (for tests)

        # Indexing (done once via build_vector_index.py)
        store.add_records(records)  # list of dicts with code, description, etc.

        # Querying (done by the agent at runtime)
        results = store.search("heart failure", top_k=10)
    """

    def __init__(self, persist_dir: Path | str | None = None) -> None:
        self._persist_dir = str(persist_dir or settings.chroma_persist_dir)
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Lazy ChromaDB client — creates persistence directory on first access."""
        if self._client is None:
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the clinical codes collection.

        Uses ChromaDB's default embedding function (all-MiniLM-L6-v2)
        which matches what we use in build_vector_index.py.
        """
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_records(
        self,
        records: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> int:
        """Add code records to the vector store.

        Each record must have: code, description, code_system.
        Optional: chapter, category.

        The description is what gets embedded. Metadata (code, code_system,
        chapter) is stored alongside for filtering and retrieval.

        Returns number of records added.
        """
        if not records:
            return 0

        total_added = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            ids = []
            documents = []
            metadatas = []

            for rec in batch:
                code = rec["code"]
                code_system = rec["code_system"]
                # Unique ID: code_system:code (handles cross-system duplicates)
                doc_id = f"{code_system}:{code}"

                ids.append(doc_id)
                documents.append(rec["description"])
                metadatas.append({
                    "code": code,
                    "code_system": code_system,
                    "chapter": rec.get("chapter", ""),
                    "category": rec.get("category", ""),
                })

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total_added += len(batch)

        logger.info(f"Added {total_added} records to vector store")
        return total_added

    def search(
        self,
        query: str,
        top_k: int | None = None,
        code_system: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search for clinical codes by natural language query.

        Args:
            query: Natural language description (e.g., "heart valve replacement")
            top_k: Number of results to return (default: settings.semantic_search_top_k)
            code_system: Optional filter — restrict to ICD-10-CM, ICD-10-PCS, etc.

        Returns:
            List of dicts with keys: code, description, code_system, chapter,
            category, score (cosine similarity, 0-1 where 1 is perfect match).
            Sorted by score descending.
        """
        if top_k is None:
            top_k = settings.semantic_search_top_k

        # Build optional where filter
        where_filter = None
        if code_system:
            where_filter = {"code_system": code_system}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

        # ChromaDB returns nested lists (one per query)
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0] if results["documents"] else ["" for _ in ids]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{} for _ in ids]
        distances = results["distances"][0] if results["distances"] else [1.0 for _ in ids]

        output = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            score = 1.0 - (dist / 2.0)

            output.append({
                "code": meta.get("code", ""),
                "description": doc,
                "code_system": meta.get("code_system", ""),
                "chapter": meta.get("chapter", ""),
                "category": meta.get("category", ""),
                "score": round(score, 4),
                "source": "vector_search",
            })

        # Filter below similarity threshold
        threshold = settings.similarity_threshold
        output = [r for r in output if r["score"] >= threshold]

        return output

    @property
    def stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        try:
            count = self.collection.count()
            return {
                "total_records": count,
                "persist_dir": self._persist_dir,
                "collection_name": COLLECTION_NAME,
            }
        except Exception:
            return {"total_records": 0, "persist_dir": self._persist_dir}

    def reset(self) -> None:
        """Delete and recreate the collection. Use for re-indexing."""
        self.client.delete_collection(COLLECTION_NAME)
        self._collection = None
        logger.info("Vector store reset — collection deleted")
