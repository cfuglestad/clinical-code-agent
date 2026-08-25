"""Ingest parsed CMS data into DuckDB structured store.

Reads files from data/raw/ (produced by download_cms_data.py),
parses them, and loads into the DuckDB database at data/processed/codes.duckdb.

Usage:
    python scripts/ingest_codes.py
    python scripts/ingest_codes.py --raw-dir data/raw --db-path data/processed/codes.duckdb
    python scripts/ingest_codes.py --only icd10cm
"""

import argparse
import sys
from pathlib import Path

# Add src/ to path for imports (when not pip-installed)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_code_agent.ingestion.parsers import (
    CodeRecord,
    parse_icd10cm,
    parse_icd10pcs,
    parse_msdrg_descriptions,
)
from clinical_code_agent.store.duckdb_store import CodeStore

DEFAULT_RAW = Path("data/raw")
DEFAULT_DB = Path("data/processed/codes.duckdb")


def find_file(directory: Path, pattern: str) -> Path | None:
    """Find a file matching a pattern in directory (case-insensitive)."""
    for f in directory.rglob("*"):
        if f.is_file() and pattern.lower() in f.name.lower():
            return f
    return None


def ingest_icd10cm(raw_dir: Path, store: CodeStore) -> int:
    """Find and ingest ICD-10-CM codes."""
    print("\n--- ICD-10-CM ---")
    # Prefer order file (has header flags); fall back to codes file
    code_file = find_file(raw_dir / "icd10cm", "icd10cm_order")

    if not code_file:
        code_file = find_file(raw_dir / "icd10cm", "icd10cm_codes")

    if not code_file:
        code_file = find_file(raw_dir / "icd10cm", "codes")

    if not code_file:
        print("  WARNING: ICD-10-CM code file not found.")
        print(f"  Expected in: {raw_dir / 'icd10cm'}")
        print("  Run: python scripts/download_cms_data.py --only icd10cm")
        return 0

    print(f"  Parsing: {code_file.name}")
    records = parse_icd10cm(code_file)
    print(f"  Parsed: {len(records):,} codes")

    billable = sum(1 for r in records if not r.is_header)
    headers = sum(1 for r in records if r.is_header)
    print(f"  Billable codes: {billable:,} | Category headers: {headers:,}")

    count = store.load_records(records)
    print(f"  Loaded: {count:,} records into DuckDB")
    return count


def ingest_icd10pcs(raw_dir: Path, store: CodeStore) -> int:
    """Find and ingest ICD-10-PCS codes."""
    print("\n--- ICD-10-PCS ---")
    code_file = find_file(raw_dir / "icd10pcs", "icd10pcs_codes")

    if not code_file:
        code_file = find_file(raw_dir / "icd10pcs", "codes")

    if not code_file:
        print("  WARNING: ICD-10-PCS code file not found.")
        print(f"  Expected in: {raw_dir / 'icd10pcs'}")
        print("  Run: python scripts/download_cms_data.py --only icd10pcs")
        return 0

    print(f"  Parsing: {code_file.name}")
    records = parse_icd10pcs(code_file)
    print(f"  Parsed: {len(records):,} codes")

    count = store.load_records(records)
    print(f"  Loaded: {count:,} records into DuckDB")
    return count


def ingest_msdrg(raw_dir: Path, store: CodeStore) -> int:
    """Find and ingest MS-DRG definitions."""
    print("\n--- MS-DRG ---")
    drg_dir = raw_dir / "msdrg"

    # MS-DRG files can have various names depending on CMS release
    drg_file = find_file(drg_dir, "drg") or find_file(drg_dir, "appendix")

    if not drg_file:
        # Fall back: look for any .txt file in the msdrg directory
        txt_files = list(drg_dir.rglob("*.txt")) if drg_dir.exists() else []
        if txt_files:
            drg_file = txt_files[0]

    if not drg_file:
        print("  WARNING: MS-DRG definitions file not found.")
        print(f"  Expected in: {drg_dir}")
        print("  Run: python scripts/download_cms_data.py --only msdrg")
        return 0

    print(f"  Parsing: {drg_file.name}")
    records = parse_msdrg_descriptions(drg_file)
    print(f"  Parsed: {len(records):,} DRG definitions")

    count = store.load_records(records)
    print(f"  Loaded: {count:,} records into DuckDB")
    return count


def main() -> None:
    """Run the full ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest CMS data into DuckDB")
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW,
        help=f"Directory with raw CMS files (default: {DEFAULT_RAW})",
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB,
        help=f"DuckDB file path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--only", choices=["icd10cm", "icd10pcs", "msdrg"],
        help="Ingest only a specific source",
    )
    args = parser.parse_args()

    print("Clinical Code Agent — Data Ingestion")
    print(f"Raw data: {args.raw_dir.resolve()}")
    print(f"Database: {args.db_path.resolve()}")

    # Ensure DB directory exists
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    store = CodeStore(db_path=args.db_path)
    total = 0

    if args.only is None or args.only == "icd10cm":
        total += ingest_icd10cm(args.raw_dir, store)

    if args.only is None or args.only == "icd10pcs":
        total += ingest_icd10pcs(args.raw_dir, store)

    if args.only is None or args.only == "msdrg":
        total += ingest_msdrg(args.raw_dir, store)

    # Build hierarchy from loaded codes
    print("\n--- Building Hierarchy ---")
    hierarchy_count = store.build_hierarchy()
    print(f"  Relationships created: {hierarchy_count:,}")

    # Print summary
    stats = store.stats
    print(f"\n{'=' * 60}")
    print("INGESTION COMPLETE")
    print(f"{'=' * 60}")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    print(f"\nDatabase: {args.db_path.resolve()}")
    print("\nNext step: python scripts/build_vector_index.py")

    # DuckDB connections are closed automatically on garbage collection
    # but we can explicitly delete if needed
    del store


if __name__ == "__main__":
    main()
