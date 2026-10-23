"""Project configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

GRAPHDB_ENDPOINT = os.getenv(
    "GRAPHDB_ENDPOINT",
    "http://localhost:7200/repositories/metagenomicKG",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
GRAPHDB_TIMEOUT = int(os.getenv("GRAPHDB_TIMEOUT", "120"))
