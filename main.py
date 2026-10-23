"""Command-line entry point for the MetagenomicKG agent."""

import argparse
import sys

from metagenomic_kg_agent.agent import build_agent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask questions about MetagenomicKG using GraphDB and Ollama."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. Omit it to start an interactive session.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show agent tool reasoning/logs.")
    args = parser.parse_args()

    agent = build_agent(verbose=args.verbose)
    question = " ".join(args.question).strip()

    if question:
        result = agent.invoke({"input": question})
        print(result.get("output", result))
        return 0

    print("MetagenomicKG assistant. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0
        if not question:
            continue
        try:
            result = agent.invoke({"input": question})
            print(f"\nAssistant: {result.get('output', result)}")
        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            print("Check that GraphDB and Ollama are running and configured correctly.")


if __name__ == "__main__":
    raise SystemExit(main())
