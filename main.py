"""CLI entry point for the Clinical Code Intelligence Agent.

Usage:
    python main.py "R65.20"
    python main.py "what codes relate to sepsis?"
    python main.py --interactive
"""

import argparse
import sys

from clinical_code_agent.agent import build_graph
from clinical_code_agent.config import settings
from clinical_code_agent.state import AgentState


def run_query(query: str) -> AgentState:
    """Run a single query through the agent pipeline."""
    app = build_graph()

    initial_state: AgentState = {
        "query": query,
        "query_type": "",
        "tool_results": [],
        "synthesis": "",
        "citations": [],
        "confidence": 0.0,
        "messages": [],
        "error": "",
    }

    result = app.invoke(initial_state)
    return result  # type: ignore[return-value]


def print_result(result: AgentState) -> None:
    """Pretty-print agent results to terminal."""
    print("\n" + "=" * 60)
    print(f"Query type: {result.get('query_type', 'unknown')}")
    print(f"Confidence: {result.get('confidence', 0.0):.0%}")
    print("-" * 60)
    print(result.get("synthesis", "No result."))

    citations = result.get("citations", [])
    if citations:
        print("\nSources:")
        for cite in citations:
            print(f"  - {cite}")

    messages = result.get("messages", [])
    if messages:
        print(f"\nTrace ({len(messages)} steps):")
        for msg in messages:
            print(f"  [{messages.index(msg) + 1}] {msg}")
    print("=" * 60)


def interactive_mode() -> None:
    """Run the agent in interactive REPL mode."""
    print("Clinical Code Intelligence Agent (interactive mode)")
    print(f"Model: {settings.openai_model} | Mode: {'LIVE' if settings.openai_api_key else 'STUB'}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not query:
            continue

        result = run_query(query)
        print_result(result)


def cli() -> None:
    """Parse CLI arguments and dispatch."""
    parser = argparse.ArgumentParser(
        description="Clinical Code Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "R65.20"
  python main.py "what codes relate to sepsis?"
  python main.py "DRG 470"
  python main.py --interactive
""",
    )
    parser.add_argument("query", nargs="?", help="Code or clinical question")
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive REPL mode",
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.query:
        result = run_query(args.query)
        print_result(result)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
