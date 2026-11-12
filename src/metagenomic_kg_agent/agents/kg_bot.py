"""
KG Bot for the MetagenomicKG multi-agent framework.

Responsibilities:
1. Receive structured tasks from the Orchestrator.
2. Identify the required knowledge graph capability.
3. Select the corresponding registered tool.
4. Extract and validate input entities.
5. Execute the selected tool.
6. Return standardised results to the Orchestrator.

The KG Bot does not decompose questions or create execution plans.
"""

import json

from metagenomic_kg_agent.tools import (
    search_entities_tool,
    get_taxonomy_relationships_tool,
    get_biological_associations_tool,
    get_molecular_interactions_tool,
    get_disease_phenotype_treatment_tool,
    get_process_composition_location_tool,
    get_metadata_provenance_tool,
    get_metabolites_for_microbe_tool,
    get_microbes_for_metabolite_tool,
)


# ============================================================
# 1. CAPABILITY-TO-TOOL MAPPING
# ============================================================

CAPABILITY_TO_TOOL = {

    # MetagenomicKG

    "knowledge_graph_entity_search":
        search_entities_tool,

    "knowledge_graph_taxonomy_relationships":
        get_taxonomy_relationships_tool,

    "knowledge_graph_biological_associations":
        get_biological_associations_tool,

    "knowledge_graph_molecular_interactions":
        get_molecular_interactions_tool,

    "knowledge_graph_disease_phenotype_treatment":
        get_disease_phenotype_treatment_tool,

    "knowledge_graph_process_composition_location":
        get_process_composition_location_tool,

    "knowledge_graph_metadata_provenance":
        get_metadata_provenance_tool,

    # MGMLink

    "knowledge_graph_metabolites_for_microbe":
        get_metabolites_for_microbe_tool,

    "knowledge_graph_microbes_for_metabolite":
        get_microbes_for_metabolite_tool
}


# ============================================================
# 2. EXPECTED INPUT PARAMETERS
# ============================================================

CAPABILITY_INPUTS = {

    "knowledge_graph_entity_search": "name",

    "knowledge_graph_taxonomy_relationships": "entity_uri",

    "knowledge_graph_biological_associations": "entity_uri",

    "knowledge_graph_molecular_interactions": "entity_uri",

    "knowledge_graph_disease_phenotype_treatment": "entity_uri",

    "knowledge_graph_process_composition_location": "entity_uri",

    "knowledge_graph_metadata_provenance": "entity_uri",

    "knowledge_graph_metabolites_for_microbe": "microbe",

    "knowledge_graph_microbes_for_metabolite": "metabolite"
}


# ============================================================
# 3. HELPER: PARSE TOOL RESPONSE
# ============================================================

def _parse_tool_response(response):
    """
    Convert tool responses into a consistent Python structure.
    """

    if isinstance(response, dict):
        return response

    if isinstance(response, str):

        try:
            return json.loads(response)

        except json.JSONDecodeError:

            # MetagenomicKG tools may return explanatory text
            # or error messages instead of JSON.
            if response.startswith("Error:") or response.startswith(
                "Error "
            ):
                return {
                    "status": "error",
                    "message": response,
                    "results": []
                }

            if response.startswith("No entities found"):
                return {
                    "status": "no_results",
                    "message": response,
                    "results": []
                }

            if response.startswith("No relationships found"):
                return {
                    "status": "no_results",
                    "message": response,
                    "results": []
                }

            return {
                "status": "success",
                "results": response
            }

    return {
        "status": "success",
        "results": response
    }


# ============================================================
# 4. HELPER: NORMALISE RESPONSE
# ============================================================

def _normalise_response(response):
    """
    Standardise the response from different knowledge graphs.

    Expected statuses:
    success
    no_results
    error
    """

    parsed = _parse_tool_response(response)

    # MGMLink returns JSON containing a results field.
    if isinstance(parsed, dict):

        results = parsed.get("results")

        if results is None:

            if parsed.get("status") in {
                "error",
                "no_results"
            }:
                return parsed

            return {
                "status": "no_results",
                "results": [],
                "message": parsed.get(
                    "message",
                    "No results returned."
                )
            }

        if isinstance(results, list) and len(results) == 0:

            return {
                "status": "no_results",
                "results": [],
                "message": parsed.get(
                    "message",
                    "The query returned no records."
                )
            }

        # Preserve metadata returned by the underlying tool.
        return {
            **parsed,
            "status": "success"
        }

    return {
        "status": "success",
        "results": parsed
    }


# ============================================================
# 5. HELPER: EXTRACT TASK INPUT
# ============================================================

def _extract_task_input(task):
    """
    Extract the actual input parameter for the selected tool.
    """

    task_input = task.get("input", {})

    if not isinstance(task_input, dict):
        raise ValueError(
            "Task input must be a dictionary."
        )

    # Directly supplied tool parameter.
    capability = task.get("required_capability")

    expected_parameter = CAPABILITY_INPUTS.get(capability)

    if expected_parameter is None:
        raise ValueError(
            f"No input mapping exists for capability: {capability}"
        )

    if expected_parameter in task_input:

        return {
            expected_parameter: task_input[expected_parameter]
        }

    # The Orchestrator can pass entities from earlier tasks.
    input_entities = task_input.get(
        "input_entities",
        []
    )

    if isinstance(input_entities, str):
        input_entities = [input_entities]

    if input_entities:

        return {
            expected_parameter: input_entities[0]
        }

    raise ValueError(
        f"Missing required input '{expected_parameter}' "
        f"for capability '{capability}'."
    )


# ============================================================
# 6. EXECUTE A SINGLE TASK
# ============================================================

def _execute_task(task):
    """
    Execute one structured knowledge graph task.
    """

    task_id = task.get("task_id")

    capability = task.get(
        "required_capability"
    )

    knowledge_graph = task.get(
        "knowledge_graph"
    )

    # --------------------------------------------------------
    # Check capability availability
    # --------------------------------------------------------

    if capability not in CAPABILITY_TO_TOOL:

        return {
            "task_id": task_id,
            "status": "invalid",
            "knowledge_graph": knowledge_graph,
            "operation": capability,
            "results": [],
            "error": (
                f"Unsupported capability: {capability}"
            )
        }

    # --------------------------------------------------------
    # Check graph selection
    # --------------------------------------------------------

    mgmlink_capabilities = {
        "knowledge_graph_metabolites_for_microbe",
        "knowledge_graph_microbes_for_metabolite"
    }

    expected_graph = (
        "MGMLink"
        if capability in mgmlink_capabilities
        else "MetagenomicKG"
    )

    if knowledge_graph != expected_graph:

        return {
            "task_id": task_id,
            "status": "invalid",
            "knowledge_graph": knowledge_graph,
            "operation": capability,
            "results": [],
            "error": (
                f"Capability '{capability}' requires "
                f"the {expected_graph} knowledge graph."
            )
        }

    # --------------------------------------------------------
    # Retrieve tool
    # --------------------------------------------------------

    selected_tool = CAPABILITY_TO_TOOL[capability]

    # --------------------------------------------------------
    # Extract tool input
    # --------------------------------------------------------

    try:

        tool_input = _extract_task_input(task)

    except ValueError as error:

        return {
            "task_id": task_id,
            "status": "invalid",
            "knowledge_graph": knowledge_graph,
            "operation": capability,
            "results": [],
            "error": str(error)
        }

    # --------------------------------------------------------
    # Execute tool
    # --------------------------------------------------------

    try:

        print(
            f"\n[KG Bot] Executing task: {task_id}"
        )

        print(
            f"[KG Bot] Knowledge graph: {knowledge_graph}"
        )

        print(
            f"[KG Bot] Capability: {capability}"
        )

        print(
            f"[KG Bot] Input: {tool_input}"
        )

        response = selected_tool.invoke(tool_input)

        parsed_response = _normalise_response(response)

        return {
            "task_id": task_id,
            "status": parsed_response["status"],
            "knowledge_graph": knowledge_graph,
            "operation": capability,
            "results": parsed_response.get(
                "results",
                []
            ),
            "message": parsed_response.get(
                "message"
            ),
            "error": parsed_response.get(
                "error"
            )
        }

    except Exception as error:

        print(
            f"[KG Bot] Task {task_id} failed: {error}"
        )

        return {
            "task_id": task_id,
            "status": "error",
            "knowledge_graph": knowledge_graph,
            "operation": capability,
            "results": [],
            "error": str(error)
        }


# ============================================================
# 7. BUILD KG BOT
# ============================================================

def build_kg_bot(model_registry=None, verbose=False):
    """
    Build the KG Bot.

    The model_registry parameter is retained for compatibility
    with the existing framework.

    KG Bot now uses deterministic tool dispatch and does not
    require an LLM to execute structured tasks.
    """

    def kg_bot(state):

        # ----------------------------------------------------
        # Step 1: Retrieve structured task
        # ----------------------------------------------------

        task = state.get("structured_task")

        if not task:

            return {
                "kg_results": None,
                "execution_status": "kg_failed",
                "errors": [
                    "No structured task received from Orchestrator."
                ]
            }

        # ----------------------------------------------------
        # Step 2: Execute task
        # ----------------------------------------------------

        result = _execute_task(task)

        # ----------------------------------------------------
        # Step 3: Return standardised response
        # ----------------------------------------------------

        if verbose:

            print(
                f"\n[KG Bot] Task status: {result['status']}"
            )

        return {

            "kg_results": result,

            "execution_status": (
                "kg_completed"
                if result["status"] in {
                    "success",
                    "no_results"
                }
                else "kg_failed"
            ),

            "task_id": result["task_id"]
        }

    return kg_bot