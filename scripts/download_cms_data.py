"""Download public CMS reference files for clinical code data.

This script fetches freely available ICD-10-CM, ICD-10-PCS, and MS-DRG
data files from CMS.gov. All data is public domain (no license required).

Usage:
    python scripts/download_cms_data.py
    python scripts/download_cms_data.py --output-dir data/raw
    python scripts/download_cms_data.py --only icd10cm

CMS updates these files annually (usually October for the new fiscal year).
Current files: FY2025 (effective Oct 1, 2024).
"""

import argparse
import io
import zipfile
from pathlib import Path

import requests

# CMS public download URLs (FY2025 release)
# These are stable annual release URLs from CMS.gov
CMS_SOURCES = {
    "icd10cm": {
        "url": "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip",
        "description": "ICD-10-CM diagnosis codes and descriptions (FY2025)",
        "extract_pattern": "icd10cm_order_2025.txt",
    },
    "icd10pcs": {
        "url": "https://www.cms.gov/files/zip/2025-icd-10-pcs-codes-file.zip",
        "description": "ICD-10-PCS procedure codes and descriptions (FY2025)",
        "extract_pattern": "icd10pcs_codes_2025.txt",
    },
    "msdrg": {
        "url": "https://www.cms.gov/files/zip/icd-10-ms-drg-definitions-manual-files-v42.zip",
        "description": "MS-DRG v42 definitions and logic (FY2025)",
        "extract_pattern": None,  # Extract all files
    },
}

DEFAULT_OUTPUT = Path("data/raw")


def download_file(url: str, description: str) -> bytes:
    """Download a file from URL with progress indication."""
    print(f"  Downloading: {description}")
    print(f"  URL: {url}")

    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    data = io.BytesIO()
    downloaded = 0

    for chunk in response.iter_content(chunk_size=8192):
        data.write(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"\r  Progress: {pct:.0f}% ({downloaded:,} / {total:,} bytes)", end="")

    print(f"\n  Downloaded: {downloaded:,} bytes")
    return data.getvalue()


def extract_zip(zip_bytes: bytes, output_dir: Path, pattern: str | None = None) -> list[Path]:
    """Extract ZIP contents, optionally filtering by filename pattern."""
    extracted = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            # Skip directories and macOS resource forks
            if name.endswith("/") or "__MACOSX" in name:
                continue

            if pattern and pattern.lower() not in name.lower():
                continue

            output_path = output_dir / Path(name).name
            output_path.write_bytes(zf.read(name))
            extracted.append(output_path)
            print(f"  Extracted: {output_path.name} ({output_path.stat().st_size:,} bytes)")

    return extracted


def download_source(source_key: str, output_dir: Path) -> list[Path]:
    """Download and extract a single CMS data source."""
    source = CMS_SOURCES[source_key]
    print(f"\n{'=' * 60}")
    print(f"Source: {source_key.upper()}")
    print(f"{'=' * 60}")

    source_dir = output_dir / source_key
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        zip_bytes = download_file(source["url"], source["description"])
        extracted = extract_zip(zip_bytes, source_dir, source["extract_pattern"])
        print(f"  Total files extracted: {len(extracted)}")
        return extracted
    except requests.RequestException as e:
        print(f"  ERROR: Download failed: {e}")
        print(f"  You can manually download from: {source['url']}")
        print(f"  Place extracted files in: {source_dir}")
        return []


def main() -> None:
    """Download all CMS reference data."""
    parser = argparse.ArgumentParser(description="Download CMS clinical code reference data")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory for raw files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--only", choices=list(CMS_SOURCES.keys()),
        help="Download only a specific source",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("CMS Clinical Code Data Downloader")
    print(f"Output: {args.output_dir.resolve()}")

    sources = [args.only] if args.only else list(CMS_SOURCES.keys())
    all_files: list[Path] = []

    for source_key in sources:
        files = download_source(source_key, args.output_dir)
        all_files.extend(files)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(all_files)} files downloaded to {args.output_dir}")
    print(f"{'=' * 60}")
    print("\nNext step: python scripts/ingest_codes.py")


if __name__ == "__main__":
    main()
