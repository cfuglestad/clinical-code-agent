"""LangGraph agent — the ReAct-style reasoning engine.

The agent classifies user intent, selects and calls tools,
then synthesizes results into human-readable explanations.
"""

from clinical_code_agent.agent.graph import build_graph

__all__ = ["build_graph"]
