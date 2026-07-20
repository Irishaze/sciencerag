"""LLM/embedding backend config for sciencerag.priors (spec §9 OQ#2).

Decision (M1-9): LLM = DeepSeek via litellm; embedding = a local
SentenceTransformer model, independently swappable. Both are read from
environment variables (see .env.example) so switching providers later is a
config change, not a code change — except that changing SCIENCERAG_EMBEDDING_MODEL
requires rebuilding the retrieval index (spec §3.5: index and embedding model
version are bound together).
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_MODEL = "deepseek/deepseek-chat"
DEFAULT_EMBEDDING_MODEL = "st-all-MiniLM-L6-v2"


def get_llm_model() -> str:
    return os.environ.get("SCIENCERAG_LLM_MODEL", DEFAULT_LLM_MODEL)


def get_embedding_model() -> str:
    return os.environ.get("SCIENCERAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
