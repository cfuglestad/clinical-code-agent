"""LangGraph state machine for the Clinical Code Intelligence Agent.

Architecture (ReAct-style tool-calling agent):

    [User Query]
        → classify_node: determine intent (code lookup vs semantic vs hierarchy vs DRG)
        → execute_tool_node: call the appropriate tool based on classification
        → synthesize_node: LLM generates human-readable explanation from tool results
        → END

The agent can optionally loop (call additional tools) if the first result
is insufficient, up to settings.max_tool_calls times.
"""

from langgraph.graph import END, StateGraph

from clinical_code_agent.config import settings
from clinical_code_agent.state import AgentState


def _classify_query(state: AgentState) -> dict[str, object]:
    """Classify the user's query into an intent category.

    Intent categories:
    - 'code_lookup': user provided a specific code (e.g., 'R65.20', 'DRG 470')
    - 'semantic_search': user asked in plain English (e.g., 'heart valve surgery')
    - 'hierarchy': user wants parent/child relationships (e.g., 'what's under R65?')
    - 'drg_explain': user wants DRG decomposition (e.g., 'what triggers DRG 871?')
    - 'general': unclear — try semantic search as default
    """
    # TODO: Replace with LLM-based classification in Phase 5
    # For now, use heuristic pattern matching as a stub
    query = state.get("query", "").strip()
    messages = list(state.get("messages", []))

    query_type = _classify_with_heuristic(query)
    messages.append(f"Classified query as: {query_type}")

    return {"query_type": query_type, "messages": messages}


def _classify_with_heuristic(query: str) -> str:
    """Rule-based classification stub (replaced by LLM in Phase 5)."""
    import re

    upper = query.upper().strip()

    # DRG pattern: 'DRG 470', 'MS-DRG 871', 'drg470'
    if re.search(r"\bM?S?-?DRG\s*\d", upper):
        return "drg_explain"

    # ICD-10 pattern: letter + digits + optional dot + digits
    if re.match(r"^[A-TV-Z]\d{2}\.?\d{0,4}$", upper):
        return "code_lookup"

    # CPT/HCPCS pattern: 5 digits or letter + 4 digits
    if re.match(r"^\d{5}$", upper) or re.match(r"^[A-V]\d{4}$", upper):
        return "code_lookup"

    # Hierarchy keywords
    hierarchy_signals = ["under", "children", "parent", "subcategories", "category"]
    if any(signal in query.lower() for signal in hierarchy_signals):
        return "hierarchy"

    # Default: semantic search
    return "semantic_search"


def _execute_tool(state: AgentState) -> dict[str, object]:
    """Execute the appropriate tool based on classified intent.

    Each tool returns a list of result dicts with consistent keys:
    - 'code': the clinical code
    - 'description': human-readable description
    - 'source': which reference system (ICD-10-CM, MS-DRG, etc.)
    - 'score': relevance/confidence (1.0 for exact matches)
    """
    # TODO: Wire real tools in Phase 4. For now, return stub results.
    import warnings

    query = state.get("query", "")
    query_type = state.get("query_type", "general")
    messages = list(state.get("messages", []))

    warnings.warn(
        f"Using stub tool execution for '{query_type}'. "
        "Wire real tools in Phase 4.",
        RuntimeWarning,
        stacklevel=1,
    )

    tool_results = [{
        "code": "STUB",
        "description": f"Stub result for query: {query}",
        "source": "stub",
        "score": 0.0,
    }]
    messages.append(f"Executed tool: {query_type} (stub mode)")

    return {"tool_results": tool_results, "messages": messages}


def _synthesize(state: AgentState) -> dict[str, object]:
    """Generate a human-readable explanation from tool results.

    Uses the LLM to transform structured tool output into natural language
    that a clinician, coder, or analyst would find immediately useful.
    """
    # TODO: Replace with LLM call in Phase 5
    import warnings

    tool_results = state.get("tool_results", [])
    messages = list(state.get("messages", []))

    warnings.warn(
        "Using stub synthesis. Replace with OpenAI call in Phase 5.",
        RuntimeWarning,
        stacklevel=1,
    )

    if not tool_results:
        synthesis = "No results found for your query."
        confidence = 0.0
    else:
        # Stub: just format the raw results
        lines = []
        for r in tool_results:
            lines.append(f"{r.get('code', '?')}: {r.get('description', '?')}")
        synthesis = "\n".join(lines)
        confidence = max((r.get("score", 0.0) for r in tool_results), default=0.0)

    citations = [
        f"{r.get('source', 'unknown')}: {r.get('code', '?')}"
        for r in tool_results
    ]
    messages.append("Synthesized explanation (stub mode)")

    return {
        "synthesis": synthesis,
        "citations": citations,
        "confidence": confidence,
        "messages": messages,
    }


def _should_continue(state: AgentState) -> str:
    """Decide whether to call another tool or proceed to synthesis.

    Routes to 'execute_tool' if confidence is low and we haven't
    exceeded max_tool_calls. Otherwise routes to 'synthesize'.
    """
    confidence = state.get("confidence", 0.0)
    messages = state.get("messages", [])
    tool_call_count = sum(1 for m in messages if "Executed tool" in m)

    if confidence < settings.abstention_threshold and tool_call_count < settings.max_tool_calls:
        return "execute_tool"
    return "synthesize"


def build_graph() -> StateGraph:
    """Construct and compile the agent state graph.

    Returns a compiled LangGraph Runnable ready for .invoke() or .stream().
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", _classify_query)
    graph.add_node("execute_tool", _execute_tool)
    graph.add_node("synthesize", _synthesize)

    # Define edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "execute_tool")
    graph.add_edge("execute_tool", "synthesize")
    graph.add_edge("synthesize", END)

    # NOTE: The conditional routing (for multi-tool loops) will be added
    # in Phase 5 when we have real confidence scores from the LLM.
    # For now, it's a simple linear flow: classify → execute → synthesize.

    return graph.compile()
