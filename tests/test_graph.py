"""Tests for the agent graph structure and classification logic."""

import pytest

from clinical_code_agent.agent.graph import _classify_with_heuristic, build_graph
from clinical_code_agent.state import AgentState


class TestQueryClassification:
    """Test the heuristic query classifier."""

    @pytest.mark.parametrize(
        "query,expected",
        [
            # ICD-10-CM diagnosis codes
            ("R65.20", "code_lookup"),
            ("I21.0", "code_lookup"),
            ("Z96.641", "code_lookup"),
            ("A41.9", "code_lookup"),
            # DRG patterns
            ("DRG 470", "drg_explain"),
            ("MS-DRG 871", "drg_explain"),
            ("drg 291", "drg_explain"),
            # CPT/HCPCS codes
            ("99213", "code_lookup"),
            ("27447", "code_lookup"),
            ("J0129", "code_lookup"),
            # Hierarchy questions
            ("what's under R65?", "hierarchy"),
            ("subcategories of sepsis", "hierarchy"),
            ("children of I21", "hierarchy"),
            # Semantic / natural language
            ("heart valve replacement", "semantic_search"),
            ("what codes relate to sepsis?", "semantic_search"),
            ("hip replacement surgery", "semantic_search"),
        ],
    )
    def test_classification(self, query: str, expected: str) -> None:
        assert _classify_with_heuristic(query) == expected


class TestGraphExecution:
    """Test that the graph runs end-to-end in stub mode."""

    def test_graph_builds(self) -> None:
        """Graph compiles without error."""
        app = build_graph()
        assert app is not None

    def test_graph_invokes(self) -> None:
        """Graph runs end-to-end with stub tools."""
        app = build_graph()
        initial_state: AgentState = {
            "query": "R65.20",
            "query_type": "",
            "tool_results": [],
            "synthesis": "",
            "citations": [],
            "confidence": 0.0,
            "messages": [],
            "error": "",
        }

        with pytest.warns(RuntimeWarning):
            result = app.invoke(initial_state)

        assert result["query_type"] == "code_lookup"
        assert result["synthesis"]  # non-empty
        assert len(result["messages"]) >= 2  # classify + execute + synthesize

    def test_semantic_query(self) -> None:
        """Natural language queries route to semantic_search."""
        app = build_graph()
        initial_state: AgentState = {
            "query": "what codes relate to heart failure?",
            "query_type": "",
            "tool_results": [],
            "synthesis": "",
            "citations": [],
            "confidence": 0.0,
            "messages": [],
            "error": "",
        }

        with pytest.warns(RuntimeWarning):
            result = app.invoke(initial_state)

        assert result["query_type"] == "semantic_search"
