import os
import re
import requests
import pandas as pd


class MGMLinkTools:

    def __init__(self, endpoint=None):

        self.endpoint = endpoint or os.getenv(
            "MGMLINK_GRAPHDB_ENDPOINT",
            "http://localhost:7200/repositories/MGM_KG"
        )

        self.BIOLINK = "https://w3id.org/biolink/vocab/"
        self.MGMLINK = "https://w3id.org/mgmlink/vocab/"

    def query(self, sparql):

        response = requests.get(
            self.endpoint,
            params={"query": sparql},
            headers={
                "Accept": "application/sparql-results+json"
            },
            timeout=60
        )

        response.raise_for_status()

        return response.json()["results"]["bindings"]

    @staticmethod
    def escape(value):

        return (
            str(value)
            .strip()
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

    def get_metabolites_for_microbe(self, microbe):

        parts = microbe.strip().split()

        genus = re.escape(parts[0])

        if len(parts) > 1:
            species = re.escape(parts[1])
            regex_pattern = f"{genus}[ ]+{species}"
        else:
            regex_pattern = genus

        regex_pattern = regex_pattern.replace("\\", "")

        sparql = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX biolink: <https://w3id.org/biolink/vocab/>
        PREFIX mgmlink: <https://w3id.org/mgmlink/vocab/>

        SELECT DISTINCT
            ?microbe
            ?microbe_label
            ?metabolite
            ?metabolite_label
            ?relationship

        WHERE {{

            ?association rdf:subject ?microbe ;
                         rdf:predicate ?relationship ;
                         rdf:object ?metabolite .

            FILTER(
                ?relationship = biolink:produces ||
                ?relationship = mgmlink:metabolizes
            )

            ?microbe rdfs:label ?microbe_label .

            OPTIONAL {{
                ?metabolite rdfs:label ?metabolite_label .
            }}

            FILTER(
                REGEX(
                    STR(?microbe_label),
                    "{regex_pattern}",
                    "i"
                )
            )
        }}

        ORDER BY ?microbe_label ?metabolite_label
        """

        results = self.query(sparql)

        records = []

        for r in results:

            relationship_uri = r["relationship"]["value"]

            relationship = (
                "produces"
                if relationship_uri == self.BIOLINK + "produces"
                else "metabolizes"
            )

            records.append({
                "microbe": r["microbe_label"]["value"],
                "microbe_uri": r["microbe"]["value"],
                "metabolite": r.get(
                    "metabolite_label", {}
                ).get("value", r["metabolite"]["value"]),
                "metabolite_uri": r["metabolite"]["value"],
                "relationship": relationship
            })

        return pd.DataFrame(records)

    def get_microbes_for_metabolite(self, metabolite):

        metabolite_pattern = re.escape(
            metabolite.strip()
        ).replace("\\", "")

        sparql = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX biolink: <https://w3id.org/biolink/vocab/>
        PREFIX mgmlink: <https://w3id.org/mgmlink/vocab/>

        SELECT DISTINCT
            ?microbe
            ?microbe_label
            ?metabolite
            ?metabolite_label
            ?relationship

        WHERE {{

            ?association rdf:subject ?microbe ;
                         rdf:predicate ?relationship ;
                         rdf:object ?metabolite .

            FILTER(
                ?relationship = biolink:produces ||
                ?relationship = mgmlink:metabolizes
            )

            ?microbe rdfs:label ?microbe_label .

            ?metabolite rdfs:label ?metabolite_label .

            FILTER(
                REGEX(
                    STR(?metabolite_label),
                    "{metabolite_pattern}",
                    "i"
                )
            )
        }}

        ORDER BY ?metabolite_label ?microbe_label
        """

        results = self.query(sparql)

        records = []

        for r in results:

            relationship_uri = r["relationship"]["value"]

            relationship = (
                "produces"
                if relationship_uri == self.BIOLINK + "produces"
                else "metabolizes"
            )

            records.append({
                "microbe": r["microbe_label"]["value"],
                "microbe_uri": r["microbe"]["value"],
                "metabolite": r["metabolite_label"]["value"],
                "metabolite_uri": r["metabolite"]["value"],
                "relationship": relationship
            })

        return pd.DataFrame(records)