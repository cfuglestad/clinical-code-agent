"""Agent tools for clinical code intelligence.

Each tool is a standalone callable that the agent can invoke.
Tools follow a consistent interface: take a query string,
return a list of result dicts with 'text', 'source', and 'score' keys.
"""
