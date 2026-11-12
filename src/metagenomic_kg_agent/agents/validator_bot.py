import json
import re

from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# 1. VALIDATOR SYSTEM PROMPT
# ============================================================

VALIDATOR_PROMPT = """
You are the Validator Bot in a multi-agent biological
data analysis framework.

Your responsibilities are LIMITED to four tasks:

1. Understand the user's request.
2. Identify the required capabilities and agents.
3. Check capability availability and identify missing inputs.
4. Decompose complex questions into smaller logical tasks.

You do NOT execute queries or analyses.

You do NOT generate SPARQL or Python code.

You do NOT call knowledge graph tools.

The Orchestrator is responsible for executing the tasks
you generate.

============================================================
1. UNDERSTAND THE USER REQUEST
============================================================

Identify:

- The biological or computational objective.
- Biological entities mentioned in the question.
- The type of analysis requested.
- The expected information or result.

Preserve scientific names, strain names and identifiers.

============================================================
2. IDENTIFY REQUIRED CAPABILITIES
============================================================

Determine which available capabilities are required.

Select only capabilities explicitly provided to you.

Identify the agents required to fulfil the request.

============================================================
3. VALIDATE CAPABILITIES AND INPUTS
============================================================

Validation statuses:

ready:
All required capabilities and inputs are available.

needs_input:
The task is supported, but additional information or
files are required.

unsupported:
The requested task cannot currently be performed.

partially_supported:
Some parts can be performed, but others cannot.

Important rules:

- Never invent capabilities.
- Never invent uploaded files.
- Distinguish missing inputs from unsupported capabilities.
- Do not assume KG retrieval can perform raw sequencing
  analysis, statistical analysis or machine learning.
- Distinguish taxonomic profiling from functional profiling.
- For FASTQ analysis, consider sequencing type and
  required input information.

============================================================
4. QUESTION DECOMPOSITION
============================================================

Decompose the user's request into smaller logical tasks
that can be executed independently or sequentially.

For simple questions, create one task.

For complex questions, create multiple tasks.

Each task must contain:

- task_id: Unique integer.
- description: What the task needs to accomplish.
- required_capability: Capability needed.
- knowledge_graph: Relevant knowledge graph.
- input_entities: Biological entities or inputs required.
- depends_on: IDs of prerequisite tasks.
- expected_output: Information expected from the task.

Dependency rules:

1. Independent tasks must have an empty depends_on list.

2. If a task requires the output of an earlier task,
   specify that task in depends_on.

3. Do not invent results from earlier tasks.

4. If a task requires multiple entities retrieved from
   an earlier task, indicate that the Orchestrator must
   execute the relevant operation for each entity.

5. Arrange tasks in logical execution order.

6. Do not create circular dependencies.

7. Only decompose tasks that can be supported by the
   available capabilities.

8. If a task is unsupported, identify it separately
   in unsupported_tasks.

9. Do not generate SPARQL, Python or executable code.

10. Do not decide whether a task succeeded. The
    Orchestrator checks execution results.

============================================================
AVAILABLE AGENTS
============================================================

entry_agent:
Receives the original user request.

validator_bot:
Understands, validates and decomposes the request.

kg_bot:
Executes supported knowledge graph operations.

orchestrator_bot:
Controls task execution, dependencies and execution state.

output_agent:
Combines the results into a final response.

Do not select an Operator Agent because it has not
yet been implemented.

============================================================
BIOLOGICAL RULES
============================================================

1. Distinguish produces from metabolizes.

2. Do not infer causation from associations.

3. Do not assume different knowledge graph IRIs
   identify the same biological entity.

4. Preserve the distinction between microbial species,
   strains and subspecies.

5. Do not assume that absence of KG results means
   that a biological relationship does not exist.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Use this structure:

{
    "validation_status": "ready",
    "original_question": "",
    "identified_entities": [],
    "analysis_type": [],
    "available_capabilities": [],
    "required_capabilities": [],
    "selected_agents": [],
    "execution_plan": [
        {
            "task_id": 1,
            "description": "",
            "required_capability": "",
            "knowledge_graph": "",
            "input_entities": [],
            "depends_on": [],
            "expected_output": ""
        }
    ],
    "unsupported_tasks": [],
    "missing_information": [],
    "validation_reason": ""
}

Do not include Markdown fences or explanatory text
outside the JSON object.
"""


# ============================================================
# 2. AVAILABLE CAPABILITIES
# ============================================================

AVAILABLE_CAPABILITIES = [
    # MetagenomicKG
    "knowledge_graph_entity_search",
    "knowledge_graph_taxonomy_relationships",
    "knowledge_graph_biological_associations",
    "knowledge_graph_molecular_interactions",
    "knowledge_graph_disease_phenotype_treatment",
    "knowledge_graph_process_composition_location",
    "knowledge_graph_metadata_provenance",

    # MGMLink
    "knowledge_graph_metabolites_for_microbe",
    "knowledge_graph_microbes_for_metabolite"
]


# ============================================================
# 3. AVAILABLE AGENTS
# ============================================================

AVAILABLE_AGENTS = [
    "entry_agent",
    "validator_bot",
    "kg_bot",
    "orchestrator_bot",
    "output_agent"
]


# ============================================================
# 4. JSON EXTRACTION
# ============================================================

def _extract_json(response):

    content = getattr(response, "content", response)

    if isinstance(content, list):
        content = " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    content = str(content).strip()

    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE
    )

    decoder = json.JSONDecoder()

    for index, character in enumerate(content):

        if character == "{":

            try:
                result, _ = decoder.raw_decode(content[index:])

                if isinstance(result, dict):
                    return result

            except json.JSONDecodeError:
                continue

    raise ValueError(
        "No valid JSON object found in model response."
    )


# ============================================================
# 5. RESPONSE VALIDATION
# ============================================================

def _validate_response(result):

    valid_statuses = {
        "ready",
        "needs_input",
        "unsupported",
        "partially_supported"
    }

    status = result.get("validation_status")

    if status not in valid_statuses:
        raise ValueError(
            f"Invalid validation status: {status}"
        )

    required_fields = [
        "original_question",
        "identified_entities",
        "analysis_type",
        "available_capabilities",
        "required_capabilities",
        "selected_agents",
        "execution_plan",
        "unsupported_tasks",
        "missing_information",
        "validation_reason"
    ]

    for field in required_fields:

        if field not in result:
            raise ValueError(
                f"Missing required field: {field}"
            )

    # --------------------------------------------------------
    # Validate capabilities
    # --------------------------------------------------------

    for capability in result["available_capabilities"]:

        if capability not in AVAILABLE_CAPABILITIES:
            raise ValueError(
                f"Unknown capability: {capability}"
            )

    for capability in result["required_capabilities"]:

        if capability not in AVAILABLE_CAPABILITIES:
            raise ValueError(
                f"Unknown required capability: {capability}"
            )

    # --------------------------------------------------------
    # Validate agents
    # --------------------------------------------------------

    for agent in result["selected_agents"]:

        if agent not in AVAILABLE_AGENTS:
            raise ValueError(
                f"Unknown agent: {agent}"
            )

    # --------------------------------------------------------
    # Validate execution plan
    # --------------------------------------------------------

    execution_plan = result["execution_plan"]

    if not isinstance(execution_plan, list):
        raise ValueError(
            "execution_plan must be a list."
        )

    seen_task_ids = set()

    for task in execution_plan:

        required_task_fields = [
            "task_id",
            "description",
            "required_capability",
            "knowledge_graph",
            "input_entities",
            "depends_on",
            "expected_output"
        ]

        for field in required_task_fields:

            if field not in task:
                raise ValueError(
                    f"Task missing field: {field}"
                )

        task_id = task["task_id"]

        if not isinstance(task_id, int) or task_id <= 0:
            raise ValueError(
                "task_id must be a positive integer."
            )

        if task_id in seen_task_ids:
            raise ValueError(
                f"Duplicate task_id: {task_id}"
            )

        capability = task["required_capability"]

        if capability not in AVAILABLE_CAPABILITIES:
            raise ValueError(
                f"Unsupported task capability: {capability}"
            )

        dependencies = task["depends_on"]

        if not isinstance(dependencies, list):
            raise ValueError(
                "depends_on must be a list."
            )

        # Dependencies must refer to earlier tasks.
        for dependency in dependencies:

            if dependency not in seen_task_ids:
                raise ValueError(
                    f"Invalid dependency {dependency} "
                    f"in task {task_id}."
                )

        seen_task_ids.add(task_id)

    return result


# ============================================================
# 6. BUILD VALIDATOR BOT
# ============================================================

def build_validator_bot(model_registry):

    llm = model_registry.get_model("validator_bot")

    def validator_bot(state):

        # ----------------------------------------------------
        # Step 1: Retrieve user request
        # ----------------------------------------------------

        messages = state.get("messages", [])

        if not messages:

            return {
                "validation_status": "needs_input",
                "original_question": "",
                "identified_entities": [],
                "analysis_type": [],
                "available_capabilities": [],
                "required_capabilities": [],
                "selected_agents": [],
                "execution_plan": [],
                "unsupported_tasks": [],
                "missing_information": [
                    "User request"
                ],
                "validation_reason": (
                    "No user request was provided."
                )
            }

        latest_message = messages[-1].content

        if isinstance(latest_message, list):

            latest_message = " ".join(
                item.get("text", "")
                for item in latest_message
                if isinstance(item, dict)
            )

        latest_message = str(latest_message).strip()

        if not latest_message:

            return {
                "validation_status": "needs_input",
                "original_question": "",
                "identified_entities": [],
                "analysis_type": [],
                "available_capabilities": [],
                "required_capabilities": [],
                "selected_agents": [],
                "execution_plan": [],
                "unsupported_tasks": [],
                "missing_information": [
                    "User request"
                ],
                "validation_reason": (
                    "The user request is empty."
                )
            }

        # ----------------------------------------------------
        # Step 2: Retrieve context
        # ----------------------------------------------------

        context_summary = state.get(
            "context_summary",
            ""
        )

        task_type = state.get(
            "task_type",
            "new_task"
        )

        uploaded_files = state.get(
            "uploaded_files",
            []
        )

        dataset_metadata = state.get(
            "dataset_metadata",
            {}
        )

        # ----------------------------------------------------
        # Step 3: Prepare user prompt
        # ----------------------------------------------------

        user_prompt = f"""
USER REQUEST:
{latest_message}

TASK CLASSIFICATION:
{task_type}

PREVIOUS CONTEXT:
{context_summary}

UPLOADED FILES:
{json.dumps(uploaded_files, default=str)}

DATASET METADATA:
{json.dumps(dataset_metadata, default=str)}

AVAILABLE CAPABILITIES:
{json.dumps(AVAILABLE_CAPABILITIES, indent=2)}

AVAILABLE AGENTS:
{json.dumps(AVAILABLE_AGENTS, indent=2)}

Understand, validate and decompose this request.

Return the required JSON structure.
"""

        # ----------------------------------------------------
        # Step 4: Invoke LLM
        # ----------------------------------------------------

        response = llm.invoke([
            SystemMessage(
                content=VALIDATOR_PROMPT
            ),
            HumanMessage(
                content=user_prompt
            )
        ])

        # ----------------------------------------------------
        # Step 5: Parse and validate response
        # ----------------------------------------------------

        try:

            result = _extract_json(response)

            result = _validate_response(result)

            # ------------------------------------------------
            # Step 6: Return validation result
            # ------------------------------------------------

            return {
                "validation_status": result[
                    "validation_status"
                ],

                "original_question": result[
                    "original_question"
                ],

                "identified_entities": result[
                    "identified_entities"
                ],

                "analysis_type": result[
                    "analysis_type"
                ],

                "available_capabilities": result[
                    "available_capabilities"
                ],

                "required_capabilities": result[
                    "required_capabilities"
                ],

                "selected_agents": result[
                    "selected_agents"
                ],

                "execution_plan": result[
                    "execution_plan"
                ],

                "unsupported_tasks": result[
                    "unsupported_tasks"
                ],

                "missing_information": result[
                    "missing_information"
                ],

                "validation_reason": result[
                    "validation_reason"
                ]
            }

        except (ValueError, json.JSONDecodeError) as error:

            return {
                "validation_status": "needs_input",
                "original_question": latest_message,
                "identified_entities": [],
                "analysis_type": [],
                "available_capabilities": [],
                "required_capabilities": [],
                "selected_agents": [],
                "execution_plan": [],
                "unsupported_tasks": [],
                "missing_information": [
                    "Validation or task decomposition "
                    "could not be completed reliably."
                ],
                "validation_reason": str(error),
                "errors": [str(error)]
            }

    return validator_bot