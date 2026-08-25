"""Parsers for CMS public reference file formats.

Each parser takes a file path and returns a list of dicts ready for
DuckDB insertion. The CMS file formats are idiosyncratic:
- ICD-10-CM: fixed-width text (code at col 6, description follows)
- ICD-10-PCS: fixed-width text (7-char code, then description)
- MS-DRG: varies by file (some CSV, some fixed-width)

All parsers normalize output to a consistent schema per code system.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeRecord:
    """A single clinical code with its metadata."""

    code: str
    description: str
    code_system: str
    chapter: str = ""
    category: str = ""
    is_header: bool = False  # True for category headers (not billable)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "description": self.description,
            "code_system": self.code_system,
            "chapter": self.chapter,
            "category": self.category,
            "is_header": self.is_header,
        }


def parse_icd10cm(file_path: Path) -> list[CodeRecord]:
    """Parse ICD-10-CM code file (tabular order format).

    CMS format: fixed-width, each line has:
    - Positions 0-4: order number (ignored)
    - Position 5: blank
    - Positions 6-12: code (no dot in file, e.g., 'R6520')
    - Position 13: blank
    - Position 14: header indicator (0=valid billable code, 1=header/category)
    - Position 15: blank
    - Positions 16+: short description

    We format the code with a dot (R6520 -> R65.20) for readability.
    """
    records: list[CodeRecord] = []

    with open(file_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if len(line) < 17:
                continue

            raw_code = line[6:13].strip()
            is_header = line[14:15] == "1"
            description = line[16:].strip()

            if not raw_code:
                continue

            # Format with dot: R6520 -> R65.20 (dot after 3rd char for CM codes)
            formatted_code = _format_icd10_code(raw_code)

            # Derive chapter and category from code structure
            chapter = _get_icd10cm_chapter(raw_code)
            category = raw_code[:3] if len(raw_code) >= 3 else raw_code

            records.append(CodeRecord(
                code=formatted_code,
                description=description,
                code_system="ICD-10-CM",
                chapter=chapter,
                category=category,
                is_header=is_header,
            ))

    return records


def parse_icd10pcs(file_path: Path) -> list[CodeRecord]:
    """Parse ICD-10-PCS code file.

    CMS format: fixed-width, each line has:
    - Positions 0-4: order number (ignored)
    - Position 5: blank
    - Positions 6-12: 7-character PCS code (no dots)
    - Position 13: blank
    - Position 14: header indicator (0=valid, 1=header)
    - Position 15: blank
    - Positions 16+: short description

    PCS codes are always 7 characters and have no dot.
    """
    records: list[CodeRecord] = []

    with open(file_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if len(line) < 17:
                continue

            raw_code = line[6:13].strip()
            is_header = line[14:15] == "1"
            description = line[16:].strip()

            if not raw_code or len(raw_code) != 7:
                continue

            # PCS section is first character
            section = raw_code[0]
            # Body system is second character
            category = raw_code[:2]

            records.append(CodeRecord(
                code=raw_code,
                description=description,
                code_system="ICD-10-PCS",
                chapter=f"Section {section}",
                category=category,
                is_header=is_header,
            ))

    return records


def parse_msdrg_descriptions(file_path: Path) -> list[CodeRecord]:
    """Parse MS-DRG descriptions from the CMS definitions file.

    The MS-DRG list is typically a text or CSV file with:
    - DRG number (3 digits, zero-padded)
    - Description
    - MDC (Major Diagnostic Category) number
    - Type (Medical/Surgical/Other)

    Handles multiple formats: tries tab-delimited, then fixed-width.
    """
    records: list[CodeRecord] = []
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.strip().split("\n")

    for line in lines:
        # Try to extract DRG number and description
        # Pattern: DRG number (1-3 digits) followed by description
        match = re.match(r"^\s*(\d{1,3})\s+(.+)$", line.strip())
        if not match:
            # Try tab-delimited
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0].strip().isdigit():
                drg_num = parts[0].strip()
                description = parts[1].strip()
            else:
                continue
        else:
            drg_num = match.group(1)
            description = match.group(2).strip()

        # Pad DRG to 3 digits
        drg_code = f"{int(drg_num):03d}"

        # Derive MDC from DRG range (approximate — refined in later phases)
        mdc = _estimate_mdc(int(drg_num))

        records.append(CodeRecord(
            code=drg_code,
            description=description,
            code_system="MS-DRG",
            chapter=f"MDC {mdc:02d}" if mdc > 0 else "Pre-MDC",
            category=f"DRG {drg_code}",
            is_header=False,
        ))

    return records


def _format_icd10_code(raw: str) -> str:
    """Format a raw ICD-10 code with dot notation.

    ICD-10-CM: dot after 3rd character (e.g., R6520 -> R65.20)
    Only codes longer than 3 characters get a dot.
    """
    if len(raw) <= 3:
        return raw
    return f"{raw[:3]}.{raw[3:]}"


def _get_icd10cm_chapter(code: str) -> str:
    """Map ICD-10-CM code to chapter by first-letter range."""
    first = code[0].upper() if code else ""
    chapter_map = {
        "A": "I: Infectious", "B": "I: Infectious",
        "C": "II: Neoplasms", "D": "III: Blood/Immune",
        "E": "IV: Endocrine", "F": "V: Mental",
        "G": "VI: Nervous", "H": "VII: Eye/Ear",
        "I": "IX: Circulatory", "J": "X: Respiratory",
        "K": "XI: Digestive", "L": "XII: Skin",
        "M": "XIII: Musculoskeletal", "N": "XIV: Genitourinary",
        "O": "XV: Pregnancy", "P": "XVI: Perinatal",
        "Q": "XVII: Congenital", "R": "XVIII: Symptoms",
        "S": "XIX: Injury", "T": "XIX: Injury",
        "V": "XX: External Causes", "W": "XX: External Causes",
        "X": "XX: External Causes", "Y": "XX: External Causes",
        "Z": "XXI: Factors",
    }
    return chapter_map.get(first, "Unknown")


def _estimate_mdc(drg_num: int) -> int:
    """Rough MDC estimation from DRG number.

    This is approximate — the real mapping comes from the CMS grouper.
    Used only for initial categorization; refined when full DRG table is loaded.
    """
    if drg_num <= 17:
        return 0  # Pre-MDC (transplants, tracheostomies, ECMO)
    if drg_num <= 103:
        return 1  # Nervous System
    if drg_num <= 125:
        return 2  # Eye
    if drg_num <= 159:
        return 3  # ENT
    if drg_num <= 168:
        return 4  # Respiratory
    if drg_num <= 316:
        return 5  # Circulatory
    if drg_num <= 395:
        return 6  # Digestive
    if drg_num <= 446:
        return 7  # Hepatobiliary
    if drg_num <= 521:
        return 8  # Musculoskeletal
    if drg_num <= 566:
        return 9  # Skin
    if drg_num <= 607:
        return 10  # Endocrine
    if drg_num <= 645:
        return 11  # Kidney
    if drg_num <= 675:
        return 12  # Male Reproductive
    if drg_num <= 700:
        return 13  # Female Reproductive
    if drg_num <= 782:
        return 14  # Pregnancy
    if drg_num <= 795:
        return 15  # Newborn
    if drg_num <= 816:
        return 16  # Blood
    if drg_num <= 849:
        return 17  # Myeloproliferative
    if drg_num <= 872:
        return 18  # Infectious
    if drg_num <= 897:
        return 19  # Mental
    if drg_num <= 923:
        return 20  # Alcohol/Drug
    if drg_num <= 935:
        return 21  # Injury
    if drg_num <= 945:
        return 22  # Burns
    if drg_num <= 951:
        return 23  # Other Factors
    return 25  # Unrelated OR / HIV
