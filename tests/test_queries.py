"""Small unit tests for query helpers; these do not require GraphDB."""

import pytest

from metagenomic_kg_agent.queries import _sparql_string, _validate_entity_uri
from metagenomic_kg_agent.graphdb_client import run_sparql


def test_sparql_string_escapes_quotes_and_backslashes():
    encoded = _sparql_string('E. coli "strain" \\ test')
    assert encoded.startswith('"') and encoded.endswith('"')
    assert '\\"' in encoded


def test_entity_uri_requires_absolute_http_uri():
    assert _validate_entity_uri("https://example.org/entity/1") == "<https://example.org/entity/1>"
    with pytest.raises(ValueError):
        _validate_entity_uri("not-a-uri")


def test_only_select_queries_are_allowed():
    with pytest.raises(ValueError):
        run_sparql("DELETE WHERE { ?s ?p ?o }")
