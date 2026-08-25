"""Tests for the ChromaDB vector store.

All tests are self-contained: they use temporary directories and
synthetic data so they run without needing real CMS downloads.
The sentence-transformers model is used for embedding (downloaded
automatically on first test run, ~90MB).
"""

import pytest

from clinical_code_agent.store.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path):
    """VectorStore backed by a temporary directory."""
    return VectorStore(persist_dir=tmp_path / "chroma_test")


@pytest.fixture
def sample_records():
    """Synthetic clinical code records for testing."""
    return [
        {
            "code": "I21.0",
            "description": "ST elevation myocardial infarction involving left main coronary artery",
            "code_system": "ICD-10-CM",
            "chapter": "Diseases of the circulatory system",
            "category": "Ischemic heart diseases",
        },
        {
            "code": "I21.1",
            "description": "ST elevation myocardial infarction involving left anterior descending coronary artery",
            "code_system": "ICD-10-CM",
            "chapter": "Diseases of the circulatory system",
            "category": "Ischemic heart diseases",
        },
        {
            "code": "I50.9",
            "description": "Heart failure unspecified",
            "code_system": "ICD-10-CM",
            "chapter": "Diseases of the circulatory system",
            "category": "Heart failure",
        },
        {
            "code": "J18.9",
            "description": "Pneumonia unspecified organism",
            "code_system": "ICD-10-CM",
            "chapter": "Diseases of the respiratory system",
            "category": "Pneumonia",
        },
        {
            "code": "R65.20",
            "description": "Severe sepsis without septic shock",
            "code_system": "ICD-10-CM",
            "chapter": "Symptoms signs and abnormal clinical and laboratory findings",
            "category": "SIRS",
        },
        {
            "code": "R65.21",
            "description": "Severe sepsis with septic shock",
            "code_system": "ICD-10-CM",
            "chapter": "Symptoms signs and abnormal clinical and laboratory findings",
            "category": "SIRS",
        },
        {
            "code": "02RF0JZ",
            "description": "Replacement of aortic valve with synthetic substitute open approach",
            "code_system": "ICD-10-PCS",
            "chapter": "",
            "category": "Heart and Great Vessels",
        },
        {
            "code": "02RG0JZ",
            "description": "Replacement of mitral valve with synthetic substitute open approach",
            "code_system": "ICD-10-PCS",
            "chapter": "",
            "category": "Heart and Great Vessels",
        },
        {
            "code": "0016070",
            "description": "Bypass cerebral ventricle to nasopharynx with autologous tissue substitute open approach",
            "code_system": "ICD-10-PCS",
            "chapter": "",
            "category": "Central Nervous System",
        },
        {
            "code": "470",
            "description": "Major hip and knee joint replacement or reattachment of lower extremity without MCC",
            "code_system": "MS-DRG",
            "chapter": "",
            "category": "MDC 08",
        },
    ]


@pytest.fixture
def populated_store(tmp_store, sample_records):
    """VectorStore with sample records already indexed."""
    tmp_store.add_records(sample_records)
    return tmp_store


# ---------------------------------------------------------------------------
# Tests: Indexing
# ---------------------------------------------------------------------------


class TestAddRecords:
    """Tests for VectorStore.add_records()."""

    def test_add_records_returns_count(self, tmp_store, sample_records):
        """add_records returns the number of records added."""
        count = tmp_store.add_records(sample_records)
        assert count == len(sample_records)

    def test_add_empty_list_returns_zero(self, tmp_store):
        """Adding empty list is a no-op."""
        assert tmp_store.add_records([]) == 0

    def test_stats_reflects_indexed_count(self, populated_store, sample_records):
        """stats.total_records matches what was indexed."""
        stats = populated_store.stats
        assert stats["total_records"] == len(sample_records)

    def test_upsert_is_idempotent(self, tmp_store, sample_records):
        """Adding the same records twice doesn't duplicate them."""
        tmp_store.add_records(sample_records)
        tmp_store.add_records(sample_records)
        assert tmp_store.stats["total_records"] == len(sample_records)


# ---------------------------------------------------------------------------
# Tests: Search
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for VectorStore.search()."""

    def test_search_returns_results(self, populated_store):
        """Searching for a known concept returns results."""
        results = populated_store.search("heart attack")
        assert len(results) > 0

    def test_search_returns_dicts_with_expected_keys(self, populated_store):
        """Each result has the standard tool interface keys."""
        results = populated_store.search("sepsis")
        assert len(results) > 0
        for r in results:
            assert "code" in r
            assert "description" in r
            assert "code_system" in r
            assert "score" in r
            assert "source" in r
            assert r["source"] == "vector_search"

    def test_search_scores_are_between_0_and_1(self, populated_store):
        """Scores are normalized to [0, 1]."""
        results = populated_store.search("pneumonia")
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_search_respects_top_k(self, populated_store):
        """top_k limits the number of results."""
        results = populated_store.search("heart", top_k=2)
        assert len(results) <= 2

    def test_search_empty_store_returns_empty(self, tmp_store):
        """Searching an empty store returns no results (not an error)."""
        results = tmp_store.search("anything")
        assert results == []

    def test_search_relevance_ordering(self, populated_store):
        """More relevant results have higher scores."""
        results = populated_store.search("severe sepsis with shock")
        if len(results) >= 2:
            # Scores should be descending (chromadb returns sorted by distance)
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_finds_correct_code_system(self, populated_store):
        """Searching for a procedure finds ICD-10-PCS codes."""
        results = populated_store.search("valve replacement surgery")
        pcs_results = [r for r in results if r["code_system"] == "ICD-10-PCS"]
        # At least one PCS result should appear for a procedure query
        assert len(pcs_results) > 0

    def test_search_code_system_filter(self, populated_store):
        """code_system filter restricts results to one system."""
        results = populated_store.search(
            "heart", code_system="ICD-10-PCS"
        )
        for r in results:
            assert r["code_system"] == "ICD-10-PCS"


# ---------------------------------------------------------------------------
# Tests: Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for VectorStore.reset()."""

    def test_reset_clears_collection(self, populated_store):
        """After reset, collection is empty."""
        assert populated_store.stats["total_records"] > 0
        populated_store.reset()
        # Need to re-access collection after reset
        assert populated_store.stats["total_records"] == 0
