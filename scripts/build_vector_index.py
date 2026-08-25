#!/usr/bin/env python3
"""Build the ChromaDB vector index from DuckDB code records.

This script reads all non-header code descriptions from the structured
DuckDB store and embeds them into ChromaDB for semantic search.

Prerequisites:
    - DuckDB must be populated (run ingest_codes.py first)
    - sentence-transformers will be downloaded on first run (~90MB)

Usage:
    python scripts/build_vector_index.py
    python scripts/build_vector_index.py --db-path data/processed/codes.duckdb
    python scripts/build_vector_index.py --chroma-dir chroma_db --reset
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from clinical_code_agent.store.vector_store import VectorStore  # noqa: E402


def fetch_codes_from_duckdb(db_path: str) -> list[dict]:
    """Read all non-header code records from DuckDB.

    Returns list of dicts ready for VectorStore.add_records().
    Only includes actual codes (not category headers) to keep
    the index focused on searchable clinical terms.
    """
    conn = duckdb.connect(db_path, read_only=True)

    results = conn.execute("""
        SELECT code, description, code_system, chapter, category
        FROM codes
        WHERE is_header = FALSE
            AND description != ''
        ORDER BY code_system, code
    """).fetchall()

    conn.close()

    return [
        {
            "code": row[0],
            "description": row[1],
            "code_system": row[2],
            "chapter": row[3] or "",
            "category": row[4] or "",
        }
        for row in results
    ]


def build_index(
    db_path: str = "data/processed/codes.duckdb",
    chroma_dir: str = "chroma_db",
    batch_size: int = 500,
    reset: bool = False,
) -> dict:
    """Build the full vector index.

    Args:
        db_path: Path to the populated DuckDB file.
        chroma_dir: Where to persist the ChromaDB files.
        batch_size: Records per embedding batch (larger = faster but more RAM).
        reset: If True, delete existing index before rebuilding.

    Returns:
        Dict with stats about the build process.
    """
    # Validate DuckDB exists
    if not Path(db_path).exists():
        print(f"\n\u274c DuckDB not found at: {db_path}")
        print("  Run 'python scripts/ingest_codes.py' first to populate the structured store.")
        sys.exit(1)

    print(f"\n\U0001f4e6 Building vector index...")
    print(f"   DuckDB source: {db_path}")
    print(f"   ChromaDB target: {chroma_dir}")
    print()

    # Step 1: Fetch records from DuckDB
    print("[1/3] Reading codes from DuckDB...")
    t0 = time.time()
    records = fetch_codes_from_duckdb(db_path)
    fetch_time = time.time() - t0
    print(f"       Found {len(records):,} code records ({fetch_time:.1f}s)")

    if not records:
        print("\n\u274c No records found in DuckDB. Is it populated?")
        sys.exit(1)

    # Summarize by code system
    system_counts: dict[str, int] = {}
    for r in records:
        system_counts[r["code_system"]] = system_counts.get(r["code_system"], 0) + 1
    for system, count in sorted(system_counts.items()):
        print(f"       - {system}: {count:,}")

    # Step 2: Initialize vector store
    print("\n[2/3] Initializing ChromaDB...")
    store = VectorStore(persist_dir=chroma_dir)

    if reset:
        print("       Resetting existing index...")
        try:
            store.reset()
        except Exception:
            pass  # Collection may not exist yet

    # Step 3: Embed and index
    print(f"\n[3/3] Embedding and indexing ({batch_size} records/batch)...")
    print("       (First run downloads the embedding model ~90MB)")
    t0 = time.time()

    total_batches = (len(records) + batch_size - 1) // batch_size
    total_added = 0

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        store.add_records(batch, batch_size=batch_size)
        total_added += len(batch)

        # Progress reporting
        pct = (total_added / len(records)) * 100
        print(f"       Batch {batch_num}/{total_batches} "
              f"({total_added:,}/{len(records):,} records, {pct:.0f}%)", end="\r")

    embed_time = time.time() - t0
    print(f"\n       Done! {total_added:,} records indexed in {embed_time:.1f}s")

    # Final stats
    stats = store.stats
    print(f"\n\u2705 Vector index built successfully!")
    print(f"   Total vectors: {stats['total_records']:,}")
    print(f"   Persist dir: {stats['persist_dir']}")
    print(f"   Total time: {fetch_time + embed_time:.1f}s")

    return {
        "records_fetched": len(records),
        "records_indexed": total_added,
        "systems": system_counts,
        "fetch_time_s": round(fetch_time, 2),
        "embed_time_s": round(embed_time, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector index from DuckDB code records."
    )
    parser.add_argument(
        "--db-path",
        default="data/processed/codes.duckdb",
        help="Path to populated DuckDB file (default: data/processed/codes.duckdb)",
    )
    parser.add_argument(
        "--chroma-dir",
        default="chroma_db",
        help="ChromaDB persistence directory (default: chroma_db)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Records per embedding batch (default: 500)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing index before rebuilding",
    )
    args = parser.parse_args()

    build_index(
        db_path=args.db_path,
        chroma_dir=args.chroma_dir,
        batch_size=args.batch_size,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
