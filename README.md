# MetagenomicKG Agent (VS Code project)

A small Python project based on `AGENT.ipynb`. It connects to a GraphDB SPARQL
repository and uses a local Ollama model through LangChain to answer questions
about entities and their graph relationships.

## Requirements

- macOS, Linux, or Windows
- Python 3.11 recommended
- A running GraphDB instance with a repository named `metagenomicKG`
- Ollama installed and running locally
- The configured Ollama model downloaded locally (default: `qwen2.5:7b`)

This project does not include the graph database itself or model weights.

## 1. Open in VS Code

1. Extract the ZIP file.
2. In VS Code, choose **File → Open Folder** and select `metagenomic-kg-agent`.
3. Open **Terminal → New Terminal**.

## 2. Create an environment

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, choose **Python: Select Interpreter** from the Command Palette and
select the interpreter inside `.venv`.

The project uses a `src/` layout, so install it in editable mode from the
project root:

```bash
python -m pip install -e .
```

## 3. Configure GraphDB and Ollama

Copy `.env.example` to `.env` and adjust the values if needed:

```bash
cp .env.example .env
```

Defaults:
- GraphDB endpoint: `http://localhost:7200/repositories/metagenomicKG`
- Ollama model: `qwen2.5:7b`
- GraphDB request timeout: 120 seconds

Make sure GraphDB is running and the repository name in the endpoint is correct.
Make sure Ollama is running and the selected model is available locally.

## 4. Run the agent

Ask a single question:

```bash
python main.py "Find Bifidobacterium longum and report its non-taxonomic relationships"
```

Show verbose agent logs:

```bash
python main.py --verbose "Find Escherichia coli in MetagenomicKG"
```

Start an interactive session:

```bash
python main.py
```

Type `exit` or `quit` to stop.

## 5. Query the graph directly from Python

After activating `.venv`, from the project root:

```bash
python -c "from metagenomic_kg_agent.queries import find_entities; print(find_entities('Bifidobacterium longum'))"
```

## Project layout

```text
metagenomic-kg-agent/
├── main.py
├── .vscode/launch.json
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── notebooks/
│   └── AGENT_original.ipynb
├── src/metagenomic_kg_agent/
│   ├── config.py
│   ├── graphdb_client.py
│   ├── queries.py
│   ├── tools.py
│   └── agent.py
└── tests/
    └── test_queries.py
```

## Notes and limitations

- The original notebook is preserved under `notebooks/`.
- The notebook's graph-query functions were moved into reusable modules.
- A naming collision in the notebook (a plain `find_entities` function later
  replaced by a LangChain tool with the same name) was removed.
- The GraphDB endpoint and Ollama model are configurable through `.env`.
- The SPARQL helper permits `SELECT` queries only. This is a basic guardrail,
  not a full SPARQL security boundary; use a trusted GraphDB instance.
- The unit tests validate basic input handling and do not require GraphDB or
  Ollama. The full agent has not been integration-tested here because it needs
  your local services and model.
- LangChain APIs evolve. If your installed package versions differ, check the
  error message and adjust the `langchain-classic` dependency/API accordingly.
