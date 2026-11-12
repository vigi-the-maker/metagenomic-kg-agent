"""
Controlled multi-knowledge-graph agent using LangChain and Ollama.

Supported knowledge graphs:
1. MetagenomicKG
2. MGMLink

The agent performs:
- Question classification
- Entity extraction
- Knowledge graph routing
- Tool execution
- Result validation
- Evidence-based summarisation
"""

import json
import re

from langchain_ollama import ChatOllama

from .config import OLLAMA_MODEL

from .tools import (
    # MetagenomicKG tools
    search_entities_tool,
    get_taxonomy_relationships_tool,
    get_biological_associations_tool,
    get_molecular_interactions_tool,
    get_disease_phenotype_treatment_tool,
    get_process_composition_location_tool,
    get_metadata_provenance_tool,

    # MGMLink tools
    get_metabolites_for_microbe_tool,
    get_microbes_for_metabolite_tool,
)


# ============================================================
# 1. SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are a biological knowledge graph assistant.

You can retrieve information from:
1. MetagenomicKG
2. MGMLink

IMPORTANT RULES:

1. Only use information retrieved from the supplied knowledge graphs.
2. Never invent biological entities, URIs or relationships.
3. Clearly distinguish between missing evidence and evidence
   that a biological relationship does not exist.
4. Preserve organism strain and subspecies information.
5. Distinguish between produces and metabolizes.
6. Do not infer causation from correlation or association.
7. Do not invent experimental validation.
8. Identify the knowledge graph used as the data source.
9. If no usable results are retrieved, explicitly report this.
10. Keep answers scientifically accurate and concise.
"""


# ============================================================
# 2. METAGENOMIC KG RELATIONSHIP TOOLS
# ============================================================

RELATIONSHIP_TOOLS = {

    "taxonomy": get_taxonomy_relationships_tool,

    "association": get_biological_associations_tool,

    "molecular": get_molecular_interactions_tool,

    "disease": get_disease_phenotype_treatment_tool,

    "process": get_process_composition_location_tool,

    "metadata": get_metadata_provenance_tool,

}


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def _clean_entity_name(name: str) -> str:
    """Clean an entity name extracted by the LLM."""

    if not isinstance(name, str):
        return ""

    name = name.strip()

    if name.lower().startswith("name:"):
        name = name[5:].strip()

    return name.strip("\"' ").strip()


def _extract_json(text: str) -> dict:
    """Extract a JSON object from an LLM response."""

    if not isinstance(text, str):
        raise ValueError("Expected a string containing JSON.")

    text = text.strip()

    # Remove Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Locate the first valid JSON object.
    decoder = json.JSONDecoder()

    for index, character in enumerate(text):

        if character == "{":

            try:
                result, _ = decoder.raw_decode(text[index:])

                if isinstance(result, dict):
                    return result

            except json.JSONDecodeError:
                continue

    raise ValueError(
        "The model did not return a valid JSON object."
    )


def _response_text(response) -> str:
    """Extract text from an Ollama/LangChain response."""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


def _parse_tool_results(result):
    """
    Convert tool output into a Python object.

    Supports JSON strings, dictionaries and lists.
    """

    if isinstance(result, (list, dict)):
        return result

    if not isinstance(result, str):
        return []

    result = result.strip()

    if not result:
        return []

    try:
        return json.loads(result)

    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": "Tool returned invalid JSON.",
            "raw_result": result,
        }


# ============================================================
# 4. MAIN AGENT
# ============================================================

class MetagenomicKGAgent:
    """
    Multi-knowledge-graph agent.

    Responsible for:
    - Classifying biological questions
    - Extracting biological entities
    - Selecting the appropriate knowledge graph
    - Executing graph queries
    - Summarising retrieved evidence
    """

    def __init__(self, verbose=False, llm=None):

        self.verbose = verbose

        self.llm = llm or ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
        )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    def _log(self, message: str):

        if self.verbose:
            print(message)

    # ========================================================
    # 5. QUESTION CLASSIFICATION
    # ========================================================

    def _classify_question(self, question: str) -> dict:
        """
        Extract the biological entity and classify the query.

        The LLM determines the entity and relationship category.
        The final knowledge graph routing is handled separately.
        """

        prompt = f"""
You are a biological knowledge graph query classifier.

Analyse the user's question.

Return ONLY a JSON object containing:

{{
    "entity": "biological entity mentioned in the question",
    "search_term": "clean entity name for database searching",
    "category": "taxonomy"
}}

The category MUST be one of:

- taxonomy
- association
- molecular
- disease
- process
- metadata

CATEGORY GUIDANCE:

taxonomy:
Taxonomic classification, ancestors, descendants,
species hierarchy and organism classification.

association:
Microbe associations, genetic associations,
biological relationships and correlations.

molecular:
Metabolites, genes, proteins, biochemical interactions,
molecular functions and metabolic relationships.

disease:
Diseases, phenotypes, biomarkers and treatments.

process:
Biological processes, participants, composition and location.

metadata:
Descriptions, labels, sources and entity types.

RULES:

1. Extract the actual biological entity.
2. Remove unnecessary question wording.
3. Do not include explanations.
4. Do not invent entities.
5. Return valid JSON only.

USER QUESTION:
{question}
"""

        response = self.llm.invoke(prompt)

        result = _extract_json(_response_text(response))

        entity = _clean_entity_name(
            result.get("entity", "")
        )

        search_term = _clean_entity_name(
            result.get("search_term", entity)
        )

        category = str(
            result.get("category", "association")
        ).strip().lower()

        if category not in RELATIONSHIP_TOOLS:
            category = "association"

        if not entity or not search_term:

            raise ValueError(
                "Could not extract a valid biological entity."
            )

        return {
            "entity": entity,
            "search_term": search_term,
            "category": category,
        }

    # ========================================================
    # 6. KNOWLEDGE GRAPH ROUTER
    # ========================================================

    def _route_question(self, question: str) -> str:
        """
        Determine which knowledge graph should handle the query.

        Returns:
        - MGMLINK_MICROBE
        - MGMLINK_METABOLITE
        - METAGENOMIC_KG
        """

        question_lower = question.lower()

        metabolite_terms = [
            "metabolite",
            "metabolites",
            "metabolic product",
            "metabolic products",
            "metabolomics",
        ]

        is_metabolite_query = any(
            term in question_lower
            for term in metabolite_terms
        )

        if not is_metabolite_query:
            return "METAGENOMIC_KG"

        # Reverse lookup:
        # Metabolite -> Microbes

        reverse_patterns = [
            r"\bwhich microbes\b",
            r"\bwhich microorganisms\b",
            r"\bwhich bacteria\b",
            r"\bfind microbes\b",
            r"\bfind microorganisms\b",
            r"\bfind bacteria\b",
            r"\bmicrobes associated with\b",
            r"\bmicroorganisms associated with\b",
            r"\bbacteria associated with\b",
            r"\bmicrobes producing\b",
            r"\bwhich microbes produce\b",
        ]

        for pattern in reverse_patterns:

            if re.search(pattern, question_lower):
                return "MGMLINK_METABOLITE"

        # Default direction:
        # Microbe -> Metabolites

        return "MGMLINK_MICROBE"

    # ========================================================
    # 7. MGMLINK EXECUTION
    # ========================================================

    def _execute_mgmlink(self, question: str, plan: dict,
                         route: str) -> dict:
        """
        Execute MGMLink queries.

        Supports:
        1. Microbe -> Metabolites
        2. Metabolite -> Microbes
        """

        search_term = plan["search_term"]

        if route == "MGMLINK_MICROBE":

            self._log(
                "\n[KG Agent] Selected knowledge graph: MGMLink"
            )

            self._log(
                "[KG Agent] Operation: Microbe -> Metabolites"
            )

            result = get_metabolites_for_microbe_tool.invoke(
                {
                    "microbe": search_term
                }
            )

        elif route == "MGMLINK_METABOLITE":

            self._log(
                "\n[KG Agent] Selected knowledge graph: MGMLink"
            )

            self._log(
                "[KG Agent] Operation: Metabolite -> Microbes"
            )

            result = get_microbes_for_metabolite_tool.invoke(
                {
                    "metabolite": search_term
                }
            )

        else:

            raise ValueError(
                f"Unsupported MGMLink route: {route}"
            )

        self._log(
            "[KG Agent] MGMLink execution completed."
        )

        records = _parse_tool_results(result)

        # Handle error responses from the tool.
        if isinstance(records, dict):

            if records.get("status") == "error":

                return {
                    "status": "error",
                    "graph": "MGMLink",
                    "message": records.get(
                        "message",
                        "MGMLink query failed."
                    ),
                    "data": [],
                }

            # Support wrappers that return {"results": [...]}.
            if isinstance(records.get("results"), list):

                records = records["results"]

            else:

                return {
                    "status": "error",
                    "graph": "MGMLink",
                    "message": records.get(
                        "message",
                        "Unexpected MGMLink response."
                    ),
                    "data": [],
                }

        if not isinstance(records, list):

            return {
                "status": "error",
                "graph": "MGMLink",
                "message": "Unexpected MGMLink result format.",
                "data": [],
            }

        self._log(
            f"[KG Agent] Retrieved {len(records)} MGMLink records."
        )

        return {
            "status": "success",
            "graph": "MGMLink",
            "operation": route,
            "query_entity": search_term,
            "records_retrieved": len(records),
            "data": records,
        }

    # ========================================================
    # 8. METAGENOMIC KG EXECUTION
    # ========================================================

    def _execute_metagenomic_kg(self, question: str,
                                plan: dict) -> dict:
        """
        Execute an entity search followed by a relationship query
        in MetagenomicKG.
        """

        search_term = plan["search_term"]

        category = plan["category"]

        self._log(
            "\n[KG Agent] Selected knowledge graph: MetagenomicKG"
        )

        self._log(
            f"[KG Agent] Relationship category: {category}"
        )

        # ----------------------------------------------------
        # Entity search
        # ----------------------------------------------------

        search_result = search_entities_tool.invoke(
            {
                "name": search_term
            }
        )

        search_data = _parse_tool_results(search_result)

        if isinstance(search_data, dict):

            return {
                "status": "error",
                "graph": "MetagenomicKG",
                "message": search_data.get(
                    "message",
                    "Entity search returned an unexpected response."
                ),
                "data": [],
            }

        if not isinstance(search_data, list) or not search_data:

            return {
                "status": "success",
                "graph": "MetagenomicKG",
                "message": (
                    f"No entities found for '{search_term}'."
                ),
                "data": [],
            }

        # ----------------------------------------------------
        # Resolve the entity URI
        # ----------------------------------------------------

        selected_entity = next(
            (
                item for item in search_data
                if isinstance(item, dict)
                and isinstance(item.get("entity"), str)
                and item["entity"].startswith(
                    ("http://", "https://")
                )
            ),
            None,
        )

        if selected_entity is None:

            return {
                "status": "error",
                "graph": "MetagenomicKG",
                "message": (
                    "Entity search did not return a valid URI."
                ),
                "data": [],
            }

        entity_uri = selected_entity["entity"]

        self._log(
            f"[KG Agent] Resolved entity URI: {entity_uri}"
        )

        # ----------------------------------------------------
        # Relationship retrieval
        # ----------------------------------------------------

        relationship_tool = RELATIONSHIP_TOOLS[category]

        relationship_result = relationship_tool.invoke(
            {
                "entity_uri": entity_uri
            }
        )

        relationship_data = _parse_tool_results(
            relationship_result
        )

        if isinstance(relationship_data, dict):

            if relationship_data.get("status") == "error":

                return {
                    "status": "error",
                    "graph": "MetagenomicKG",
                    "message": relationship_data.get(
                        "message",
                        "Relationship query failed."
                    ),
                    "data": [],
                }

        return {
            "status": "success",
            "graph": "MetagenomicKG",
            "entity": plan["entity"],
            "entity_uri": entity_uri,
            "category": category,
            "data": relationship_data,
        }

    # ========================================================
    # 9. RESULT SUMMARISATION
    # ========================================================

    def _summarise_results(self, question: str,
                           results: dict) -> str:
        """
        Generate an evidence-based biological answer.
        """

        if results.get("status") == "error":

            return (
                f"The query against {results.get('graph')} "
                f"could not be completed.\n\n"
                f"Details: {results.get('message')}"
            )

        data = results.get("data", [])

        if not data:

            return (
                f"No usable results were retrieved from "
                f"{results.get('graph')}.\n\n"
                "This indicates missing evidence in the queried "
                "knowledge graph, not necessarily the absence "
                "of the biological relationship."
            )

        evidence = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        prompt = f"""
{SYSTEM_INSTRUCTIONS}

USER QUESTION:
{question}

KNOWLEDGE GRAPH:
{results.get('graph')}

QUERY DETAILS:
{json.dumps({
    key: value
    for key, value in results.items()
    if key != "data"
}, indent=2, default=str)}

RETRIEVED BIOLOGICAL EVIDENCE:
{evidence}

YOUR TASK:

Answer the user's question using the retrieved evidence.

For MGMLink metabolite queries:

1. Group metabolites by relationship type.
2. Distinguish produces from metabolizes.
3. Preserve species, strain and subspecies names.
4. Preserve host information where available.
5. Avoid duplicate metabolite names where appropriate.
6. Report the number of retrieved records.

For MetagenomicKG queries:

1. Explain the retrieved relationships.
2. Identify the relevant biological entities.
3. Preserve the meaning of the predicates.
4. Avoid unsupported biological interpretations.

For all queries:

- Never invent evidence.
- Never invent missing results.
- Do not infer causation.
- Mention the knowledge graph used.
- If evidence is ambiguous, explicitly state that.
- Return a structured but concise scientific answer.

Return only the final answer.
"""

        response = self.llm.invoke(prompt)

        answer = _response_text(response)

        if not answer:

            return (
                "The knowledge graph returned results, "
                "but a summary could not be generated."
            )

        return answer

    # ========================================================
    # 10. MAIN INVOKE FUNCTION
    # ========================================================

    def invoke(self, input_data):
        """
        Main entry point.

        Pipeline:
        Question
            |
        Classification
            |
        Knowledge graph routing
            |
        Tool execution
            |
        Result validation
            |
        LLM summarisation
            |
        Final answer
        """

        # ----------------------------------------------------
        # Extract question
        # ----------------------------------------------------

        if isinstance(input_data, str):

            question = input_data

        elif isinstance(input_data, dict):

            question = (
                input_data.get("input")
                or input_data.get("question")
                or ""
            )

        else:

            raise TypeError(
                "Input must be a string or a dictionary "
                "containing 'input' or 'question'."
            )

        question = question.strip()

        if not question:

            return {
                "output": "Please provide a question."
            }

        try:

            # ------------------------------------------------
            # Step 1: Classify question
            # ------------------------------------------------

            plan = self._classify_question(question)

            self._log(
                f"\n[KG Agent] Entity: {plan['entity']}"
            )

            self._log(
                f"[KG Agent] Search term: {plan['search_term']}"
            )

            self._log(
                f"[KG Agent] Category: {plan['category']}"
            )

            # ------------------------------------------------
            # Step 2: Select knowledge graph
            # ------------------------------------------------

            route = self._route_question(question)

            self._log(
                f"[KG Agent] Selected route: {route}"
            )

            # ------------------------------------------------
            # Step 3: Execute appropriate graph query
            # ------------------------------------------------

            if route.startswith("MGMLINK"):

                results = self._execute_mgmlink(
                    question=question,
                    plan=plan,
                    route=route,
                )

            else:

                results = self._execute_metagenomic_kg(
                    question=question,
                    plan=plan,
                )

            # ------------------------------------------------
            # Step 4: Summarise results
            # ------------------------------------------------

            answer = self._summarise_results(
                question=question,
                results=results,
            )

            # ------------------------------------------------
            # Step 5: Return final response
            # ------------------------------------------------

            return {
                "output": answer,
                "execution_status": results.get(
                    "status",
                    "unknown"
                ),
                "knowledge_graph": results.get(
                    "graph",
                    "unknown"
                ),
                "records_retrieved": results.get(
                    "records_retrieved",
                    None
                ),
            }

        except Exception as exc:

            self._log(
                f"\n[KG Agent] Execution error: {exc}"
            )

            return {
                "output": (
                    "The knowledge-graph query could not "
                    f"be completed. Error: {exc}"
                ),
                "execution_status": "failed",
                "errors": [str(exc)],
            }


# ============================================================
# 11. AGENT BUILDER
# ============================================================

def build_agent(
    verbose: bool = False,
    llm=None
) -> MetagenomicKGAgent:
    """
    Construct the multi-knowledge-graph agent.
    """

    return MetagenomicKGAgent(
        verbose=verbose,
        llm=llm,
    )