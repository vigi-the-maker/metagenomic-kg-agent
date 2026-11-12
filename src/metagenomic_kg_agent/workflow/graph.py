from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from metagenomic_kg_agent.workflow.state import AgentState

from metagenomic_kg_agent.models.model_config import MODEL_CONFIG
from metagenomic_kg_agent.models.model_registry import ModelRegistry

from metagenomic_kg_agent.agents.entry_bot import build_entry_bot
from metagenomic_kg_agent.agents.validator_bot import build_validator_bot
from metagenomic_kg_agent.agents.kg_bot import build_kg_bot
from metagenomic_kg_agent.agents.orchestrator_bot import build_orchestrator_bot
from metagenomic_kg_agent.agents.output_bot import build_output_bot


def route_after_validation(state: AgentState):
    """
    Decide whether the request can proceed to orchestration.
    """

    status = state.get("validation_status")

    if status in ["needs_input", "unsupported"]:
        return "output_bot"

    return "orchestrator_bot"


def build_workflow():

    # =====================================
    # 1. INITIALISE MODEL REGISTRY
    # =====================================

    model_registry = ModelRegistry(MODEL_CONFIG)

    # =====================================
    # 2. INITIALISE AGENTS
    # =====================================

    entry_bot = build_entry_bot(model_registry)

    validator_bot = build_validator_bot(model_registry)

    kg_bot = build_kg_bot(model_registry)

    output_bot = build_output_bot(model_registry)

    # The Orchestrator receives the KG Bot and Output Bot.
    # It executes the KG tasks and then passes aggregated
    # results to the Output Bot.

    orchestrator_bot = build_orchestrator_bot(
        model_registry=model_registry,
        kg_bot=kg_bot,
        output_bot=output_bot
    )

    # =====================================
    # 3. CREATE LANGGRAPH
    # =====================================

    workflow = StateGraph(AgentState)

    # =====================================
    # 4. REGISTER NODES
    # =====================================

    workflow.add_node("entry_bot", entry_bot)

    workflow.add_node("validator_bot", validator_bot)

    workflow.add_node("orchestrator_bot", orchestrator_bot)

    workflow.add_node("output_bot", output_bot)

    # =====================================
    # 5. DEFINE EDGES
    # =====================================

    workflow.add_edge(START, "entry_bot")

    workflow.add_edge("entry_bot", "validator_bot")

    workflow.add_conditional_edges(
        "validator_bot",
        route_after_validation,
        {
            "orchestrator_bot": "orchestrator_bot",
            "output_bot": "output_bot"
        }
    )

    # The Orchestrator already invokes the KG Bot and Output Bot.
    # Therefore, its execution ends the graph workflow.
    workflow.add_edge("orchestrator_bot", END)

    workflow.add_edge("output_bot", END)

    # =====================================
    # 6. ADD CONVERSATION MEMORY
    # =====================================

    memory = MemorySaver()

    app = workflow.compile(
        checkpointer=memory
    )

    return app