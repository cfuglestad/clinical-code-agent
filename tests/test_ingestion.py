"""Tests for CMS data parsing and DuckDB store."""

import tempfile
from pathlib import Path

import pytest

from clinical_code_agent.ingestion.parsers import (
    CodeRecord,
    _format_icd10_code,
    _get_icd10cm_chapter,
    parse_icd10cm,
    parse_icd10pcs,
)
from clinical_code_agent.store.duckdb_store import CodeStore


# --- Parser Tests ---


class TestICD10Formatting:
    """Test code formatting helpers."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("R65", "R65"),          # 3-char category: no dot
            ("R6520", "R65.20"),     # 5-char: dot after 3rd
            ("I2101", "I21.01"),     # STEMI
            ("Z96641", "Z96.641"),   # 6-char extension
            ("A00", "A00"),          # 3-char boundary
            ("A000", "A00.0"),       # 4-char
        ],
    )
    def test_format_code(self, raw: str, expected: str) -> None:
        assert _format_icd10_code(raw) == expected

    @pytest.mark.parametrize(
        "code,expected_chapter",
        [
            ("R65", "XVIII: Symptoms"),
            ("I21", "IX: Circulatory"),
            ("A41", "I: Infectious"),
            ("Z96", "XXI: Factors"),
            ("C34", "II: Neoplasms"),
            ("J18", "X: Respiratory"),
        ],
    )
    def test_chapter_mapping(self, code: str, expected_chapter: str) -> None:
        assert _get_icd10cm_chapter(code) == expected_chapter


class TestICD10CMParser:
    """Test parsing of ICD-10-CM fixed-width format."""

    def test_parses_sample_lines(self, tmp_path: Path) -> None:
        """Parse a synthetic CMS-format file."""
        # CMS format: order(5) space(1) code(7) space(1) header(1) space(1) description
        sample = (
            "00001 R65    1 Conditions involving the immune mechanism\n"
            "00002 R6510  0 Systemic inflammatory response syndrome (SIRS) of non-infectious origin without acute organ dysfunction\n"
            "00003 R6520  0 Severe sepsis without septic shock\n"
            "00004 R6521  0 Severe sepsis with septic shock\n"
        )
        code_file = tmp_path / "icd10cm_codes_2025.txt"
        code_file.write_text(sample)

        records = parse_icd10cm(code_file)

        assert len(records) == 4
        # Header record
        assert records[0].code == "R65"
        assert records[0].is_header is True
        # Billable code
        assert records[2].code == "R65.20"
        assert records[2].is_header is False
        assert "sepsis" in records[2].description.lower()
        assert records[2].code_system == "ICD-10-CM"

    def test_skips_short_lines(self, tmp_path: Path) -> None:
        """Lines too short to parse are skipped."""
        sample = "short\n\n00001 R6520  0 Valid line\n"
        code_file = tmp_path / "test.txt"
        code_file.write_text(sample)

        records = parse_icd10cm(code_file)
        assert len(records) == 1


class TestICD10PCSParser:
    """Test parsing of ICD-10-PCS fixed-width format."""

    def test_parses_pcs_codes(self, tmp_path: Path) -> None:
        sample = (
            "00001 0SR9019 0 Replacement of Right Hip Joint with Metal Synthetic Substitute, Cemented, Open Approach\n"
            "00002 0SRB019 0 Replacement of Right Knee Joint with Metal Synthetic Substitute, Cemented, Open Approach\n"
        )
        code_file = tmp_path / "icd10pcs_codes_2025.txt"
        code_file.write_text(sample)

        records = parse_icd10pcs(code_file)

        assert len(records) == 2
        assert records[0].code == "0SR9019"
        assert records[0].code_system == "ICD-10-PCS"
        assert "hip" in records[0].description.lower()
        assert records[0].chapter == "Section 0"  # Medical and Surgical


# --- DuckDB Store Tests ---


class TestCodeStore:
    """Test DuckDB store operations."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> CodeStore:
        """Create a temporary DuckDB store."""
        db_path = tmp_path / "test.duckdb"
        s = CodeStore(db_path=db_path)
        # Force schema creation
        _ = s.conn
        return s

    @pytest.fixture
    def loaded_store(self, store: CodeStore) -> CodeStore:
        """Store preloaded with sample records."""
        records = [
            CodeRecord("R65", "Conditions involving immune mechanism", "ICD-10-CM",
                       "XVIII: Symptoms", "R65", True),
            CodeRecord("R65.20", "Severe sepsis without septic shock", "ICD-10-CM",
                       "XVIII: Symptoms", "R65", False),
            CodeRecord("R65.21", "Severe sepsis with septic shock", "ICD-10-CM",
                       "XVIII: Symptoms", "R65", False),
            CodeRecord("I21.0", "ST elevation myocardial infarction involving left main coronary artery", "ICD-10-CM",
                       "IX: Circulatory", "I21", False),
            CodeRecord("470", "Major Hip and Knee Joint Replacement without MCC", "MS-DRG",
                       "MDC 08", "DRG 470", False),
        ]
        store.load_records(records)
        return store

    def test_load_records(self, store: CodeStore) -> None:
        """Records load into the database."""
        records = [
            CodeRecord("R65.20", "Severe sepsis", "ICD-10-CM", "XVIII", "R65", False),
        ]
        count = store.load_records(records)
        assert count == 1
        assert store.stats["total_codes"] == 1

    def test_lookup_exact(self, loaded_store: CodeStore) -> None:
        """Exact code lookup works."""
        results = loaded_store.lookup_code("R65.20")
        assert len(results) == 1
        assert results[0]["code"] == "R65.20"
        assert results[0]["score"] == 1.0
        assert "sepsis" in results[0]["description"].lower()

    def test_lookup_without_dot(self, loaded_store: CodeStore) -> None:
        """Lookup normalizes dots away."""
        results = loaded_store.lookup_code("R6520")
        assert len(results) == 1
        assert results[0]["code"] == "R65.20"

    def test_lookup_case_insensitive(self, loaded_store: CodeStore) -> None:
        """Lookup is case-insensitive."""
        results = loaded_store.lookup_code("r65.20")
        assert len(results) == 1

    def test_lookup_drg(self, loaded_store: CodeStore) -> None:
        """DRG codes are found."""
        results = loaded_store.lookup_code("470", code_system="MS-DRG")
        assert len(results) == 1
        assert "hip" in results[0]["description"].lower()

    def test_lookup_not_found(self, loaded_store: CodeStore) -> None:
        """Unknown codes return empty list."""
        results = loaded_store.lookup_code("Z99.99")
        assert results == []

    def test_description_search(self, loaded_store: CodeStore) -> None:
        """Text search on descriptions."""
        results = loaded_store.search_description("sepsis")
        assert len(results) == 2  # R65.20 and R65.21
        assert all("sepsis" in r["description"].lower() for r in results)

    def test_hierarchy_build(self, loaded_store: CodeStore) -> None:
        """Hierarchy is derived from code structure."""
        count = loaded_store.build_hierarchy()
        # R65 (header) should be parent of R65.20, R65.21
        assert count >= 2

        children = loaded_store.get_children("R65", code_system="ICD-10-CM")
        codes = [c["code"] for c in children]
        assert "R65.20" in codes
        assert "R65.21" in codes

    def test_stats(self, loaded_store: CodeStore) -> None:
        """Stats reflect loaded data."""
        stats = loaded_store.stats
        assert stats["total_codes"] == 5
        assert stats["codes_icd_10_cm"] == 4
        assert stats["codes_ms_drg"] == 1
