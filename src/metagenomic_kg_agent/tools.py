"""LangChain tools for querying MetagenomicKG.

Provides:
1. Entity search.
2. Taxonomy hierarchy relationships.
3. Genetic and biological associations.
4. Molecular interactions and functions.
5. Disease, phenotype, and treatment relationships.
6. Processes, composition, and location relationships.
7. Metadata and provenance relationships.
"""

import json

from langchain_core.tools import tool

from .queries import (
    search_entities_detailed,
    get_relationships_by_predicates,
    search_entities_flexible,
)


# ============================================================
# PREDICATE CATEGORIES
# ============================================================

PREDICATE_CATEGORIES = {
    "taxonomy_hierarchy": [
        "subclass_of",
        "superclass_of",
        "same_as",
        "close_match",
    ],

    "genetic_biological_associations": [
        "genetically_associated_with",
        "associated_with",
        "related_to",
        "correlated_with",
        "causes",
        "affects",
        "contributes_to",
        "regulates",
        "temporally_related_to",
    ],

    "molecular_interactions_functions": [
        "interacts_with",
        "physically_interacts_with",
        "chemically_similar_to",
        "catalyzes",
        "gene_product_of",
        "is_sequence_variant_of",
        "has_input",
        "produces",
        "has_metabolite",
        "disrupts",
    ],

    "disease_phenotype_treatment": [
        "has_phenotype",
        "biomarker_for",
        "manifestation_of",
        "treats",
        "applied_to_treat",
        "contraindicated_in",
        "disease_has_basis_in",
        "drug_regulatory_status_world_wide",
    ],

    "processes_composition_location": [
        "participates_in",
        "has_participant",
        "has_part",
        "has_member",
        "located_in",
    ],

    "metadata_provenance": [
        "label",
        "comment",
        "altLabel",
        "link",
        "source",
        "type",
        "is_pathogen",
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _format_results(results) -> str:
    """Format query results as readable JSON."""

    if not results:
        return "No relationships found for this entity and predicate category."

    return json.dumps(
        results,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def _get_category_relationships(
    entity_uri: str,
    category: str,
    limit: int = 30,
) -> str:
    """Retrieve relationships for an entity within a predicate category."""

    if not entity_uri or not entity_uri.strip():
        return "Error: entity_uri cannot be empty."

    if category not in PREDICATE_CATEGORIES:
        return f"Error: Unknown predicate category '{category}'."

    predicates = PREDICATE_CATEGORIES[category]

    try:
        results = get_relationships_by_predicates(
            entity_uri=entity_uri.strip(),
            predicates=predicates,
            limit=limit,
        )

        if not results:
            return (
                f"No relationships found for URI: {entity_uri}\n"
                f"Predicate category: {category}\n"
                f"Predicates checked: {', '.join(predicates)}"
            )

        return _format_results(results)

    except Exception as exc:
        return (
            f"Error retrieving relationships for URI '{entity_uri}' "
            f"in category '{category}': {exc}"
        )


# ============================================================
# TOOL 1: SEARCH ENTITIES
# ============================================================

@tool("search_entities")
def search_entities_tool(name: str) -> str:
    """Search MetagenomicKG for a biological entity by name.

    Input should be a plain entity name, such as Escherichia coli.
    Use this tool before querying relationships for an entity.
    Never invent an entity URI if the search returns no results.
    """

    if not isinstance(name, str) or not name.strip():
        return "Error: Please provide a non-empty entity name."

    # Remove common formatting added by an LLM agent.
    name = name.strip()

    if name.startswith("name:"):
        name = name[len("name:"):].strip()

    # Remove surrounding single or double quotes.
    name = name.strip().strip('"').strip("'").strip()

    if not name:
        return "Error: The entity name is empty after input cleaning."

    try:
        results = search_entities_detailed(name=name, limit=10)

        if not results:
            return (
                f"No entities found for '{name}'. "
                "Try a partial name or inspect the graph's labels."
            )

        return json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    except Exception as exc:
        return f"Error searching for '{name}': {exc}"

# ============================================================
# TOOL 2: TAXONOMY HIERARCHY
# ============================================================

@tool("get_taxonomy_relationships")
def get_taxonomy_relationships_tool(entity_uri: str) -> str:
    """Retrieve taxonomy and hierarchy relationships for an entity.

    Includes:
    - subclass_of
    - superclass_of
    - same_as
    - close_match

    Use the full entity URI returned by search_entities.
    Do not guess the URI.

    Args:
        entity_uri: Full URI of the entity.
    """

    return _get_category_relationships(
        entity_uri=entity_uri,
        category="taxonomy_hierarchy",
    )


# ============================================================
# TOOL 3: GENETIC AND BIOLOGICAL ASSOCIATIONS
# ============================================================

@tool("get_biological_associations")
def get_biological_associations_tool(entity_uri: str) -> str:
    """Retrieve genetic and biological associations for an entity.

    Includes:
    - genetically_associated_with
    - associated_with
    - related_to
    - correlated_with
    - causes
    - affects
    - contributes_to
    - regulates
    - temporally_related_to

    Use the full entity URI returned by search_entities.

    Args:
        entity_uri: Full URI of the entity.
    """

    return _get_category_relationships(
        entity_uri=entity_uri,
        category="genetic_biological_associations",
    )


# ============================================================
# TOOL 4: MOLECULAR INTERACTIONS AND FUNCTIONS
# ============================================================

@tool("get_molecular_interactions")
def get_molecular_interactions_tool(entity_uri: str) -> str:
    """Retrieve molecular interaction and functional relationships.

    Includes:
    - interacts_with
    - physically_interacts_with
    - chemically_similar_to
    - catalyzes
    - gene_product_of
    - is_sequence_variant_of
    - has_input
    - produces
    - has_metabolite
    - disrupts

    Useful for exploring relationships between microbes, genes,
    proteins, metabolites, and biochemical functions.

    Use the full entity URI returned by search_entities.

    Args:
        entity_uri: Full URI of the entity.
    """

    return _get_category_relationships(
        entity_uri=entity_uri,
        category="molecular_interactions_functions",
    )


# ============================================================
# TOOL 5: DISEASE, PHENOTYPE, AND TREATMENT
# ============================================================

@tool("get_disease_phenotype_treatment")
def get_disease_phenotype_treatment_tool(entity_uri: str) -> str:
    """Retrieve disease, phenotype, biomarker, and treatment relationships.

    Includes:
    - has_phenotype
    - biomarker_for
    - manifestation_of
    - treats
    - applied_to_treat
    - contraindicated_in
    - disease_has_basis_in
    - drug_regulatory_status_world_wide

    Use the full entity URI returned by search_entities.

    Args:
        entity_uri: Full URI of the entity.
    """

    return _get_category_relationships(
        entity_uri=entity_uri,
        category="disease_phenotype_treatment",
    )


# ============================================================
# TOOL 6: PROCESSES, COMPOSITION, AND LOCATION
# ============================================================

@tool("get_process_composition_location")
def get_process_composition_location_tool(entity_uri: str) -> str:
    """Retrieve process, participation, composition, and location relationships.

    Includes:
    - participates_in
    - has_participant
    - has_part
    - has_member
    - located_in

    Use the full entity URI returned by search_entities.

    Args:
        entity_uri: Full URI of the entity.
    """

    return _get_category_relationships(
        entity_uri=entity_uri,
        category="processes_composition_location",
    )


# ============================================================
# TOOL 7: METADATA AND PROVENANCE
# ============================================================

@tool("get_metadata_provenance")
def get_metadata_provenance_tool(entity_uri: str) -> str:
    """Retrieve metadata and provenance associated with an entity.

    Includes:
    - label
    - comment
    - altLabel
    - link
    - source
    - type
    - is_pathogen

    This tool can return descriptive metadata, source information,
    and entity type information.

    Use the full entity URI returned by search_entities.

    Args:
        entity_uri: Full URI of the entity.
    """

    return _get_category_relationships(
        entity_uri=entity_uri,
        category="metadata_provenance",
    )


# ============================================================
# REGISTER TOOLS
# ============================================================

TOOLS = [
    search_entities_tool,
    get_taxonomy_relationships_tool,
    get_biological_associations_tool,
    get_molecular_interactions_tool,
    get_disease_phenotype_treatment_tool,
    get_process_composition_location_tool,
    get_metadata_provenance_tool,
]