"""Read-only SPARQL helpers for a GraphDB repository."""

import re
from typing import Any
from urllib.parse import urlparse

import requests

from .config import GRAPHDB_ENDPOINT, GRAPHDB_TIMEOUT

_SESSION = requests.Session()
_PREFIX_OR_BASE = re.compile(
    r"(?im)^[ \t]*(?:PREFIX[ \t]+\w*:[ \t]*<[^>]+>|BASE[ \t]+<[^>]+>)[ \t]*"
)


def run_sparql(query: str, timeout: int | None = None) -> list[dict[str, str]]:
    """Execute a read-only SPARQL SELECT query and return simple dictionaries.

    This deliberately permits SELECT queries only. It is a guardrail, not a
    complete SPARQL parser; only use this client against a trusted endpoint.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("SPARQL query must be a non-empty string.")

    cleaned_query = _PREFIX_OR_BASE.sub("", query).lstrip()
    if not re.match(r"(?is)^SELECT\b", cleaned_query):
        raise ValueError("Only SPARQL SELECT queries are permitted.")

    parsed = urlparse(GRAPHDB_ENDPOINT)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "GRAPHDB_ENDPOINT must be a valid http:// or https:// URL; "
            f"received {GRAPHDB_ENDPOINT!r}."
        )

    response = _SESSION.get(
        GRAPHDB_ENDPOINT,
        params={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=timeout or GRAPHDB_TIMEOUT,
    )
    response.raise_for_status()

    try:
        data: dict[str, Any] = response.json()
        bindings = data["results"]["bindings"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "GraphDB returned an unexpected response. Confirm that the endpoint "
            "is a SPARQL query endpoint and that the repository exists."
        ) from exc

    return [
        {key: value["value"] for key, value in row.items()}
        for row in bindings
    ]
