import json


# ============================================================
# 1. AVAILABLE AGENTS
# ============================================================

AVAILABLE_AGENTS = {
    "kg_bot": True,
    "operator_bot": False,
    "output_bot": True
}


# ============================================================
# 2. CAPABILITY REGISTRY
# ============================================================

CAPABILITY_REGISTRY = {

    # ---------------- MetagenomicKG ----------------

    "knowledge_graph_entity_search": {
        "knowledge_graph": "MetagenomicKG",
        "agent": "kg_bot",
        "input_key": "name"
    },

    "knowledge_graph_taxonomy_relationships": {
        "knowledge_graph": "MetagenomicKG",
        "agent": "kg_bot",
        "input_key": "entity_uri"
    },

    "knowledge_graph_biological_associations": {
        "knowledge_graph": "MetagenomicKG",
        "agent": "kg_bot",
        "input_key": "entity_uri"
    },

    "knowledge_graph_molecular_interactions": {
        "knowledge_graph": "MetagenomicKG",
        "agent": "kg_bot",
        "input_key": "entity_uri"
    },

    "knowledge_graph_disease_phenotype_treatment": {
        "knowledge_graph": "MetagenomicKG",
        "agent": "kg_bot",
        "input_key": "entity_uri"
    },

    "knowledge_graph_process_composition_location": {
        "knowledge_graph": "MetagenomicKG",
        "agent": "kg_bot",
        "input_key": "entity_uri"
    },

    "knowledge_graph_metadata_provenance": {
        "knowledge_graph": "MetagenomicKG",
        "agent": "kg_bot",
        "input_key": "entity_uri"
    },

    # ---------------- MGMLink ----------------

    "knowledge_graph_metabolites_for_microbe": {
        "knowledge_graph": "MGMLink",
        "agent": "kg_bot",
        "input_key": "microbe"
    },

    "knowledge_graph_microbes_for_metabolite": {
        "knowledge_graph": "MGMLink",
        "agent": "kg_bot",
        "input_key": "metabolite"
    }
}


# ============================================================
# 3. RESULT NORMALISATION
# ============================================================

def _normalise_result(result):

    if isinstance(result, dict):
        return result

    if isinstance(result, str):

        try:
            parsed = json.loads(result)

            if isinstance(parsed, dict):
                return parsed

            return {
                "status": "success",
                "results": parsed
            }

        except json.JSONDecodeError:

            if result.startswith(("Error:", "Error ")):
                return {
                    "status": "error",
                    "error": result,
                    "results": []
                }

            return {
                "status": "success",
                "results": result
            }

    if result is None:

        return {
            "status": "no_results",
            "results": None
        }

    return {
        "status": "success",
        "results": result
    }


def _get_result_status(result):

    status = result.get("status")

    if status in {
        "success",
        "no_results",
        "error",
        "invalid"
    }:
        return status

    if result.get("error"):
        return "error"

    # Check common result containers.
    for key in ("results", "relationships"):

        if key in result:

            data = result[key]

            if data is None:
                return "no_results"

            if isinstance(data, (list, dict)) and not data:
                return "no_results"

    return "success"


# ============================================================
# 4. ENTITY EXTRACTION
# ============================================================

def _extract_entities_from_result(result):

    data = result.get("results")

    if data is None:
        data = result.get("relationships")

    if data is None:
        return []

    if isinstance(data, str):

        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return []

    entities = []

    for record in data:

        if not isinstance(record, dict):
            continue

        entity = (
            record.get("metabolite")
            or record.get("entity")
            or record.get("name")
        )

        if entity:
            entities.append(entity)

    return list(dict.fromkeys(entities))


# ============================================================
# 5. TASK INPUT PREPARATION
# ============================================================

def _prepare_task_input(task, previous_results):

    task_input = dict(task.get("input", {}))

    input_entities = task.get("input_entities", [])

    if not isinstance(input_entities, list):
        input_entities = [input_entities]

    task_input["input_entities"] = input_entities

    dependencies = task.get("depends_on", [])

    if not dependencies:
        return task_input

    dependency_results = {
        dependency_id: previous_results.get(dependency_id)
        for dependency_id in dependencies
    }

    task_input["previous_results"] = dependency_results

    input_from = task.get("input_from")

    # If the Validator omits input_from but there is
    # exactly one dependency, infer the source task.
    if input_from is None and len(dependencies) == 1:
        input_from = dependencies[0]

    if input_from is not None:

        source_result = previous_results.get(input_from)

        if source_result:

            entities = _extract_entities_from_result(
                source_result
            )

            if entities:
                task_input["input_entities"] = entities

    return task_input


# ============================================================
# 6. CAPABILITY AND KNOWLEDGE GRAPH RESOLUTION
# ============================================================

def _resolve_task_routing(task):

    capability = task.get("required_capability")

    registry_entry = CAPABILITY_REGISTRY.get(capability)

    if registry_entry is None:

        return None, {
            "status": "invalid",
            "reason": f"Unsupported capability: {capability!r}"
        }

    agent = registry_entry["agent"]

    if not AVAILABLE_AGENTS.get(agent, False):

        return None, {
            "status": "error",
            "reason": f"Required agent '{agent}' is unavailable."
        }

    return {
        "required_capability": capability,
        "knowledge_graph": registry_entry["knowledge_graph"],
        "agent": agent,
        "input_key": registry_entry["input_key"]
    }, None


def _apply_capability_input(task_input, routing):

    input_key = routing["input_key"]

    # Use an explicitly supplied input if available.
    if task_input.get(input_key) not in (None, ""):
        return task_input

    # Otherwise use the first entity supplied by the
    # Validator or extracted from a dependency.
    entities = task_input.get("input_entities", [])

    if not isinstance(entities, list):
        entities = [entities]

    if entities:
        task_input[input_key] = entities[0]

    return task_input


# ============================================================
# 7. ORCHESTRATOR BUILDER
# ============================================================

def build_orchestrator_bot(
    model_registry,
    kg_bot,
    output_bot=None
):

    def orchestrator_bot(state):

        # ----------------------------------------------------
        # STEP 1: VALIDATION CHECK
        # ----------------------------------------------------

        validation_status = state.get(
            "validation_status",
            "needs_input"
        )

        if validation_status in {
            "needs_input",
            "unsupported"
        }:

            return {
                "execution_status": validation_status,
                "execution_results": {},
                "completed_tasks": [],
                "failed_tasks": [],
                "skipped_tasks": [],
                "final_response": None,
                "orchestration_reason": (
                    "Execution cannot proceed because "
                    "validation did not pass."
                )
            }

        # ----------------------------------------------------
        # STEP 2: RETRIEVE EXECUTION PLAN
        # ----------------------------------------------------

        execution_plan = state.get("execution_plan", [])

        if not isinstance(execution_plan, list):

            return {
                "execution_status": "orchestration_failed",
                "errors": [
                    "Validator execution_plan must be a list."
                ]
            }

        if not execution_plan:

            return {
                "execution_status": "orchestration_failed",
                "errors": [
                    "No executable tasks were supplied."
                ]
            }

        original_question = state.get(
            "original_question",
            ""
        )

        # ----------------------------------------------------
        # STEP 3: INITIALISE EXECUTION STATE
        # ----------------------------------------------------

        execution_results = {}

        completed_tasks = []
        no_results_tasks = []
        failed_tasks = []
        skipped_tasks = []

        # ----------------------------------------------------
        # STEP 4: EXECUTE TASKS
        # ----------------------------------------------------

        for task in execution_plan:

            task_id = task.get("task_id")

            dependencies = task.get("depends_on", [])

            # -----------------------------------------------
            # 4A. CHECK DEPENDENCIES
            # -----------------------------------------------

            failed_dependencies = [
                dependency
                for dependency in dependencies
                if execution_results.get(
                    dependency,
                    {}
                ).get("status") != "success"
            ]

            if failed_dependencies:

                execution_results[task_id] = {
                    "task_id": task_id,
                    "status": "skipped",
                    "reason": (
                        "Prerequisite task(s) did not "
                        "produce usable results."
                    ),
                    "failed_dependencies": failed_dependencies
                }

                skipped_tasks.append(task_id)

                continue

            # -----------------------------------------------
            # 4B. RESOLVE CAPABILITY AND KG
            # -----------------------------------------------

            routing, routing_error = _resolve_task_routing(task)

            if routing_error is not None:

                execution_results[task_id] = {
                    "task_id": task_id,
                    "status": routing_error["status"],
                    "required_capability": task.get(
                        "required_capability"
                    ),
                    "reason": routing_error["reason"]
                }

                failed_tasks.append(task_id)

                continue

            # -----------------------------------------------
            # 4C. PREPARE INPUT
            # -----------------------------------------------

            task_input = _prepare_task_input(
                task,
                execution_results
            )

            task_input = _apply_capability_input(
                task_input,
                routing
            )

            required_input = routing["input_key"]

            if task_input.get(required_input) in (None, ""):

                execution_results[task_id] = {
                    "task_id": task_id,
                    "status": "invalid",
                    "required_capability": routing[
                        "required_capability"
                    ],
                    "knowledge_graph": routing[
                        "knowledge_graph"
                    ],
                    "reason": (
                        f"Missing required input '{required_input}' "
                        f"for capability '{routing['required_capability']}'."
                    )
                }

                failed_tasks.append(task_id)

                continue

            # -----------------------------------------------
            # 4D. BUILD STRUCTURED TASK
            # -----------------------------------------------

            structured_task = {

                "task_id": task_id,

                "description": task.get(
                    "description",
                    ""
                ),

                "required_capability": routing[
                    "required_capability"
                ],

                "knowledge_graph": routing[
                    "knowledge_graph"
                ],

                "agent": routing["agent"],

                "input": task_input,

                "expected_output": task.get(
                    "expected_output",
                    ""
                ),

                "original_question": original_question
            }

            # -----------------------------------------------
            # 4E. EXECUTE THROUGH KG BOT
            # -----------------------------------------------

            try:

                kg_response = kg_bot(
                    {
                        "structured_task": structured_task,
                        "previous_results": execution_results,
                        "selected_capability": routing[
                            "required_capability"
                        ],
                        "selected_knowledge_graph": routing[
                            "knowledge_graph"
                        ]
                    }
                )

                if isinstance(kg_response, dict):

                    if "kg_results" in kg_response:
                        kg_response = kg_response["kg_results"]

                result = _normalise_result(kg_response)

                task_status = _get_result_status(result)

                result["task_id"] = task_id
                result["status"] = task_status
                result["knowledge_graph"] = routing[
                    "knowledge_graph"
                ]
                result["required_capability"] = routing[
                    "required_capability"
                ]

                execution_results[task_id] = result

                if task_status == "success":

                    completed_tasks.append(task_id)

                elif task_status == "no_results":

                    no_results_tasks.append(task_id)

                else:

                    failed_tasks.append(task_id)

            except Exception as error:

                execution_results[task_id] = {
                    "task_id": task_id,
                    "status": "error",
                    "reason": str(error)
                }

                failed_tasks.append(task_id)

        # ----------------------------------------------------
        # STEP 5: DETERMINE EXECUTION STATUS
        # ----------------------------------------------------

        if failed_tasks or skipped_tasks:

            if completed_tasks or no_results_tasks:
                overall_status = "partially_completed"
            else:
                overall_status = "failed"

        elif no_results_tasks:

            overall_status = "completed_with_no_results"

        else:

            overall_status = "completed"

        # ----------------------------------------------------
        # STEP 6: AGGREGATE RESULTS
        # ----------------------------------------------------

        aggregated_results = {

            "original_question": original_question,

            "execution_plan": execution_plan,

            "execution_results": execution_results,

            "completed_tasks": completed_tasks,

            "no_results_tasks": no_results_tasks,

            "failed_tasks": failed_tasks,

            "skipped_tasks": skipped_tasks,

            "execution_status": overall_status

        }

        # ----------------------------------------------------
        # STEP 7: CALL OUTPUT BOT
        # ----------------------------------------------------

        output_result = None

        if output_bot is not None:

            try:

                output_result = output_bot(
                    {
                        "messages": state.get("messages", []),
                        "execution_data": aggregated_results
                    }
                )

            except Exception as error:

                output_result = {
                    "output_status": "failed",
                    "errors": [str(error)]
                }

        # ----------------------------------------------------
        # STEP 8: RETURN FINAL STATE
        # ----------------------------------------------------

        final_response = None

        if isinstance(output_result, dict):
            final_response = output_result.get("final_response")

        elif output_result is not None:
            final_response = str(output_result)

        return {

            "execution_status": overall_status,

            "execution_results": execution_results,

            "completed_tasks": completed_tasks,

            "no_results_tasks": no_results_tasks,

            "failed_tasks": failed_tasks,

            "skipped_tasks": skipped_tasks,

            "aggregated_results": aggregated_results,

            "output_result": output_result,

            "final_response": final_response,

            "orchestration_reason": (
                "All executable tasks have been processed."
            )

        }

    return orchestrator_bot