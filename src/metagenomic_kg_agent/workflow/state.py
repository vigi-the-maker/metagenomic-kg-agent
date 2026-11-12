from typing import TypedDict, Annotated, Any, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):

    # =================================
    # CONVERSATION
    # =================================

    messages: Annotated[list[BaseMessage], add_messages]

    context_summary: str

    original_question: str

    conversation_id: str

    # =================================
    # ENTRY BOT
    # =================================

    task_type: str

    converstation_id: str

    is_new_task: bool

    entry_reason: str

    requires_previous_context: bool

    # =================================
    # VALIDATOR BOT
    # =================================

    validation_status: str

    missing_information: list[str]

    available_capabilities: list[str]

    required_capabilities: list[str]

    validation_reason: str

    unsupported_tasks: list[Any]

    identified_entities: list[Any]

    # =================================
    # ORCHESTRATOR
    # =================================

    selected_agents: list[str]

    execution_plan: list[dict[str, Any]]

    aggregated_results: dict[str, Any]

    execution_results: dict[Any, Any]

    completed_tasks: list[Any]

    failed_tasks: list[Any]

    skipped_tasks: list[Any]

    no_results_tasks: list[Any]

    orchestration_reason: str

    output_result: Any

    # =================================
    # INPUT DATA
    # =================================

    uploaded_files: list[str]

    input_type: str

    dataset_metadata: dict[str, Any]

    # =================================
    # KNOWLEDGE GRAPH BOT
    # =================================

    kg_query: str

    kg_results: Any

    # =================================
    # OPERATOR BOT
    # =================================

    analysis_type: list[str]

    analysis_results: Any

    # =================================
    # OUTPUT BOT
    # =================================

    final_response: str

    output_status: str

    knowledge_graph_sources: list[str]

    execution_summary: dict[str, Any]

    # =================================
    # EXECUTION MANAGEMENT
    # =================================

    execution_status: str

    errors: list[str]