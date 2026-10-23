"""Controlled LangChain + Ollama agent for MetagenomicKG."""

import json
import re

from langchain_ollama import ChatOllama

from .config import OLLAMA_MODEL
from .tools import (
    search_entities_tool,
    get_taxonomy_relationships_tool,
    get_biological_associations_tool,
    get_molecular_interactions_tool,
    get_disease_phenotype_treatment_tool,
    get_process_composition_location_tool,
    get_metadata_provenance_tool,
)


SYSTEM_INSTRUCTIONS = """
You are a biological knowledge graph assistant for MetagenomicKG.

You must answer questions using the supplied knowledge-graph results.

Rules:
1. Use only the supplied graph results for claims about graph contents.
2. Never invent entities, URIs, relationships, or evidence.
3. If no results are supplied, clearly state that the graph returned no results.
4. Distinguish missing graph evidence from evidence that a relationship does not exist.
5. Do not infer that a disease is caused by a microorganism merely because
   an association exists.
6. Keep the final answer concise.
"""


RELATIONSHIP_TOOLS = {
    "taxonomy": get_taxonomy_relationships_tool,
    "association": get_biological_associations_tool,
    "molecular": get_molecular_interactions_tool,
    "disease": get_disease_phenotype_treatment_tool,
    "process": get_process_composition_location_tool,
    "metadata": get_metadata_provenance_tool,
}


def _clean_entity_name(name: str) -> str:
    """Clean common formatting mistakes in extracted entity names."""

    name = name.strip()

    if name.lower().startswith("name:"):
        name = name[5:].strip()

    return name.strip("\"' ").strip()


def _extract_json(text: str) -> dict:
    """Extract a JSON object from the model's response."""

    text = text.strip()

    # Handle JSON enclosed in Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: find the first JSON object in the response.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError("The model did not return valid JSON.")


class MetagenomicKGAgent:
    """A controlled, non-looping agent for MetagenomicKG."""

    def __init__(self, verbose: bool = False):
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
        )
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _classify_question(self, question: str) -> dict:
        """Extract the entity name and choose a relationship category."""

        prompt = f"""
Identify the biological entity and the knowledge-graph relationship
category needed to answer the user's question.

Return ONLY a JSON object with these keys:
- entity: the biological entity name
- category: one of taxonomy, association, molecular, disease,
  process, metadata
- search_term: the entity name to search for

Category guidance:
- taxonomy: taxonomy, classification, ancestors, descendants
- association: genetic or general biological associations
- molecular: metabolites, molecular interactions, production,
  genes, proteins, biochemical functions
- disease: diseases, phenotypes, biomarkers, treatments
- process: biological processes, participants, composition, location
- metadata: descriptions, labels, sources, entity type

Do not answer the biological question.
Do not invent an entity URI.
Do not include Markdown fences.

Question:
{question}
"""

        response = self.llm.invoke(prompt)
        result = _extract_json(response.content)

        entity = result.get("entity", "")
        search_term = result.get("search_term", entity)
        category = result.get("category", "association")

        entity = _clean_entity_name(str(entity))
        search_term = _clean_entity_name(str(search_term))
        category = str(category).strip().lower()

        if category not in RELATIONSHIP_TOOLS:
            category = "association"

        if not entity or not search_term:
            raise ValueError(
                "The model could not identify a biological entity "
                "in the question."
            )

        return {
            "entity": entity,
            "search_term": search_term,
            "category": category,
        }

    def invoke(self, input_data):
        """Run entity resolution, relationship retrieval, and summarization."""

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
                "Input must be a question string or a dictionary "
                "containing 'input' or 'question'."
            )

        question = question.strip()

        if not question:
            return {"output": "Please provide a question."}

        try:
            # Step 1: Identify the entity and relevant relationship category.
            plan = self._classify_question(question)

            self._log(f"Entity: {plan['entity']}")
            self._log(f"Category: {plan['category']}")

            # Step 2: Search the graph for the entity.
            search_result = search_entities_tool.invoke(
                {"name": plan["search_term"]}
            )

            self._log(f"Entity search result: {search_result}")

            if not isinstance(search_result, str):
                search_result = str(search_result)

            if search_result.startswith("Error:"):
                return {"output": search_result}

            if search_result.startswith("No entities found"):
                return {
                    "output": (
                        f"I could not find '{plan['search_term']}' "
                        "in MetagenomicKG. No relationship query was run. "
                        "Try another name or inspect the graph's labels."
                    )
                }

            # Step 3: Parse the entity results and obtain a real URI.
            try:
                entities = json.loads(search_result)
            except json.JSONDecodeError:
                return {
                    "output": (
                        "The entity search returned an unexpected result. "
                        "I stopped rather than guessing an entity URI."
                    )
                }

            if not isinstance(entities, list) or not entities:
                return {
                    "output": (
                        "No usable entity results were returned. "
                        "I did not query relationships."
                    )
                }

            # The SPARQL search query uses ?entity, which run_sparql()
            # returns as the dictionary key "entity".
            selected_entity = next(
                (
                    item for item in entities
                    if isinstance(item, dict)
                    and isinstance(item.get("entity"), str)
                    and item["entity"].startswith(("http://", "https://"))
                ),
                None,
            )

            if selected_entity is None:
                return {
                    "output": (
                        "Search results did not contain a valid entity URI. "
                        "I stopped rather than inventing one."
                    )
                }

            entity_uri = selected_entity["entity"]

            self._log(f"Selected entity URI: {entity_uri}")

            # Step 4: Call exactly one relationship tool.
            relationship_tool = RELATIONSHIP_TOOLS[plan["category"]]

            relationship_result = relationship_tool.invoke(
                {"entity_uri": entity_uri}
            )

            self._log(f"Relationship results: {relationship_result}")

            # Step 5: Summarize the retrieved graph evidence.
            summary_prompt = f"""
{SYSTEM_INSTRUCTIONS}

User question:
{question}

Resolved entity:
{plan['entity']}

Resolved entity URI:
{entity_uri}

Relationship category:
{plan['category']}

Raw relationship results from MetagenomicKG:
{relationship_result}

Provide a concise answer to the user's question.

Requirements:
- Use only relationships explicitly present in the retrieved results.
- Do not invent missing relationship details.
- Do not claim that an association proves causation.
- If the relationship results contain no usable evidence, say so.
- Do not call tools or generate Action/Action Input instructions.
- Return only the final answer.
"""

            final_response = self.llm.invoke(summary_prompt)
            answer = final_response.content.strip()

            if not answer:
                answer = (
                    "The knowledge graph was queried, but the model "
                    "did not produce a final answer."
                )

            return {"output": answer}

        except Exception as exc:
            self._log(f"Agent error: {exc}")

            return {
                "output": (
                    "I could not complete the knowledge-graph query. "
                    f"Error: {exc}"
                )
            }


def build_agent(verbose: bool = False):
    """Build the controlled MetagenomicKG agent."""

    return MetagenomicKGAgent(verbose=verbose)