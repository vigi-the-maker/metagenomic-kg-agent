"""Reusable MetagenomicKG query functions."""

import json
import re
from urllib.parse import urlparse

from .graphdb_client import run_sparql


def _sparql_string(value: str) -> str:
    """Encode a Python string as a quoted SPARQL string literal."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Search name must be a non-empty string.")
    # JSON string escaping is compatible with quoted SPARQL string literals
    # for the characters handled here (quotes, backslashes, and control chars).
    return json.dumps(value, ensure_ascii=False)


def _validate_entity_uri(entity_uri: str) -> str:
    """Clean and validate an entity URI before using it in SPARQL."""

    if not isinstance(entity_uri, str) or not entity_uri.strip():
        raise ValueError("Entity URI must be a non-empty string.")

    entity_uri = entity_uri.strip()

    # Handle malformed LangChain ReAct inputs such as:
    # entity_uri: "https://w3id.org/metagenomickg/Microbe:816695"
    match = re.fullmatch(
        r"""(?:entity_uri\s*:\s*)?["']?(https?://[^"' \t\r\n]+)["']?""",
        entity_uri,
    )

    if match:
        entity_uri = match.group(1)

    parsed = urlparse(entity_uri)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"Invalid entity URI: {entity_uri!r}. "
            "Provide a complete HTTP or HTTPS URI."
        )

    if re.search(r'[<>\s"{}|^`\\]', entity_uri):
        raise ValueError(
            "Entity URI contains characters that are unsafe in SPARQL IRI syntax."
        )

    return f"<{entity_uri}>"


def find_entities(name: str, limit: int = 10) -> list[dict[str, str]]:
    """Find entities by exact, case-insensitive label or alternative label."""
    limit = max(1, min(int(limit), 100))
    literal = _sparql_string(name)
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT DISTINCT ?entity ?label
    WHERE {{
        {{ ?entity rdfs:label ?label . }}
        UNION
        {{ ?entity skos:altLabel ?label . }}
        FILTER(LCASE(STR(?label)) = LCASE({literal}))
    }}
    LIMIT {limit}
    """
    return run_sparql(query)


def search_entities_detailed(name: str, limit: int = 10) -> list[dict[str, str]]:
    """Find entities and return labels, types, and optional descriptions."""
    limit = max(1, min(int(limit), 100))
    literal = _sparql_string(name)
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT DISTINCT ?entity ?label ?type ?description
    WHERE {{
        {{ ?entity rdfs:label ?label . }}
        UNION
        {{ ?entity skos:altLabel ?label . }}
        OPTIONAL {{ ?entity rdf:type ?type . }}
        OPTIONAL {{ ?entity rdfs:comment ?description . }}
        FILTER(LCASE(STR(?label)) = LCASE({literal}))
    }}
    LIMIT {limit}
    """
    return run_sparql(query)


def inspect_entity(entity_uri: str) -> list[dict[str, str]]:
    """Return selected metadata fields for an entity URI."""
    entity = _validate_entity_uri(entity_uri)
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    SELECT ?property ?value
    WHERE {{
        {entity} ?property ?value .
        FILTER(?property = rdf:type || ?property = rdfs:label ||
               ?property = rdfs:comment || ?property = dcterms:source)
    }}
    LIMIT 30
    """
    return run_sparql(query)


def get_relationships(entity_uri: str, limit: int = 30) -> list[dict[str, str]]:
    """Return incoming and outgoing non-metadata relationships for an entity."""
    entity = _validate_entity_uri(entity_uri)
    limit = max(1, min(int(limit), 200))
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?direction ?predicate ?related_entity ?related_label
    WHERE {{
        {{ {entity} ?predicate ?related_entity . BIND("outgoing" AS ?direction) }}
        UNION
        {{ ?related_entity ?predicate {entity} . BIND("incoming" AS ?direction) }}
        FILTER(?predicate != rdf:type && ?predicate != rdfs:label &&
               ?predicate != rdfs:comment)
        OPTIONAL {{ ?related_entity rdfs:label ?related_label . }}
    }}
    LIMIT {limit}
    """
    return run_sparql(query)


def count_triples() -> list[dict[str, str]]:
    """Count all triples in the configured repository."""
    return run_sparql("""
        SELECT (COUNT(*) AS ?total)
        WHERE { ?s ?p ?o . }
    """)

def get_relationships_by_predicates(
    entity_uri: str,
    predicates: list[str],
    limit: int = 30,
) -> list[dict[str, str]]:
    """Return incoming and outgoing relationships matching predicate names.

    Args:
        entity_uri: Full URI of the entity to inspect.
        predicates: Predicate local names, e.g. ["produces", "has_metabolite"].
        limit: Maximum number of relationship results to return.

    Returns:
        SPARQL results containing the direction, predicate URI, related entity,
        and optional related entity label.
    """
    entity = _validate_entity_uri(entity_uri)

    if not predicates:
        return []

    # Keep only valid predicate local names.
    # This prevents arbitrary strings from being inserted into the SPARQL query.
    valid_predicates = []

    for predicate in predicates:
        if not isinstance(predicate, str):
            continue

        predicate = predicate.strip()

        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", predicate):
            valid_predicates.append(predicate)

    if not valid_predicates:
        return []

    limit = max(1, min(int(limit), 200))

    # Safely encode predicate names as SPARQL string literals.
    predicate_values = ", ".join(
        _sparql_string(predicate)
        for predicate in valid_predicates
    )

    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT
        ?direction
        ?predicate
        ?predicate_name
        ?related_entity
        ?related_label
    WHERE {{
        {{
            {entity} ?predicate ?related_entity .
            BIND("outgoing" AS ?direction)
        }}
        UNION
        {{
            ?related_entity ?predicate {entity} .
            BIND("incoming" AS ?direction)
        }}

        # Extract the local name from either a URI fragment or path.
        BIND(
            REPLACE(
                STR(?predicate),
                "^.*[/#]",
                ""
            ) AS ?predicate_name
        )

        FILTER(?predicate_name IN ({predicate_values}))

        OPTIONAL {{
            ?related_entity rdfs:label ?related_label .
        }}
    }}
    LIMIT {limit}
    """

    return run_sparql(query)
def search_entities_flexible(
    name: str,
    limit: int = 10,
) -> list[dict[str, str]]:
    """Search entity labels using case-insensitive substring matching."""

    limit = max(1, min(int(limit), 100))
    literal = _sparql_string(name)

    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT DISTINCT ?entity ?label ?type
    WHERE {{
        {{
            ?entity rdfs:label ?label .
        }}
        UNION
        {{
            ?entity skos:altLabel ?label .
        }}

        OPTIONAL {{
            ?entity rdf:type ?type .
        }}

        FILTER(
            CONTAINS(
                LCASE(STR(?label)),
                LCASE({literal})
            )
        )
    }}
    LIMIT {limit}
    """

    return run_sparql(query)