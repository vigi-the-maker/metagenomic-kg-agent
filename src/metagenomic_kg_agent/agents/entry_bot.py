import json
import re

from langchain_core.messages import SystemMessage, HumanMessage


ENTRY_BOT_PROMPT = """
You are the Entry Bot of a multi-agent microbiome data analysis framework.

Your responsibility is to determine the relationship between the user's
latest message and the previous conversation.

Classify the latest message into exactly one of these categories:

1. new_task:
   The user is starting an entirely different task.

2. continuation:
   The user is continuing an existing task or asking for its next step.

3. clarification:
   The user is clarifying, correcting, or modifying an earlier request.

IMPORTANT RULES:

- Examine the previous conversation before deciding.
- Use the context summary when available.
- A short message such as "continue", "next step", or "explain this"
  usually depends on previous context.
- A new biological dataset or unrelated research question may represent
  a new task.
- Do not assume that every new message represents a new task.
- Do not invent missing conversation history.
- If there is no previous conversation, classify it as new_task.

Return ONLY valid JSON in this format:

{
    "task_type": "new_task",
    "is_new_task": true,
    "requires_previous_context": false,
    "entry_reason": "Explanation of the classification",
    "context_summary": "Brief summary of relevant context"
}
"""


def _extract_json(response):

    content = response.content

    if isinstance(content, list):
        content = " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    content = str(content).strip()

    # Remove optional Markdown code fences.
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE
    )

    match = re.search(r"\{.*\}", content, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group())


def _message_to_text(message):

    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    return str(content)


def build_entry_bot(model_registry):

    llm = model_registry.get_model("entry_bot")

    def entry_bot(state):

        messages = state.get("messages", [])

        context_summary = state.get("context_summary", "")

        # Keep the recent conversation available for classification.
        previous_messages = messages[:-1]

        recent_history = previous_messages[-8:]

        latest_message = (
            _message_to_text(messages[-1])
            if messages
            else ""
        )

        # No previous context means this must be a new task.
        if not recent_history and not context_summary.strip():

            return {
                "task_type": "new_task",
                "is_new_task": True,
                "requires_previous_context": False,
                "entry_reason": "No previous conversation context exists.",
                "context_summary": ""
            }

        history_text = "\n".join(
            f"{message.type}: {_message_to_text(message)}"
            for message in recent_history
        )

        user_prompt = f"""
Previous context summary:

{context_summary}

Recent conversation:

{history_text}

Latest user message:

{latest_message}

Classify the latest message.
"""

        response = llm.invoke([
            SystemMessage(content=ENTRY_BOT_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        try:

            result = _extract_json(response)

            task_type = result.get("task_type", "new_task")

            if task_type not in {
                "new_task",
                "continuation",
                "clarification"
            }:
                raise ValueError("Invalid task classification.")

            is_new_task = task_type == "new_task"

            requires_previous_context = not is_new_task

            return {
                "task_type": task_type,
                "is_new_task": is_new_task,
                "requires_previous_context": requires_previous_context,
                "entry_reason": result.get("entry_reason", ""),
                "context_summary": result.get(
                    "context_summary",
                    context_summary
                )
            }

        except (ValueError, json.JSONDecodeError) as error:

            return {
                "task_type": "new_task",
                "is_new_task": True,
                "requires_previous_context": False,
                "entry_reason": f"Classification failed: {error}",
                "errors": [str(error)]
            }

    return entry_bot