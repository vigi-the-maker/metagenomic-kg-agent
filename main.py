import argparse
import uuid

from langchain_core.messages import HumanMessage
from metagenomic_kg_agent.workflow.graph import build_workflow


def main():

    parser = argparse.ArgumentParser(
        description="MetagenomicKG Multi-Agent Framework"
    )

    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask the multi-agent framework"
    )

    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    parser.add_argument(
        "--thread-id",
        default=None,
        help="Conversation thread ID"
    )

    args = parser.parse_args()

    app = build_workflow()

    thread_id = args.thread_id or str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    question = " ".join(args.question).strip()

    print(f"\nConversation ID: {thread_id}")

    if question:

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config=config
        )

        print("\nAssistant:")
        print(result.get("final_response", "No response generated."))

        return 0

    print("\nMetagenomicKG Multi-Agent Framework")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config=config
        )

        print("\nAssistant:")
        print(result.get("final_response", "No response generated."))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())