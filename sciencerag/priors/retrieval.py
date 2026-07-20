"""PaperQA2 wiring for sciencerag.priors (spec §3.2).

Builds a paper-qa Settings object from our LLM/embedding config (§9 OQ#2)
pointed at the internal literature corpus (corpus/papers/), and exposes a
thin query function. The retrieval index is cached under .pqa_index/
(project-local, gitignored) rather than the shared ~/.pqa/ cache so this
project's index doesn't mix with unrelated projects.
"""

from pathlib import Path

from paperqa import Settings, ask
from paperqa.agents.main import AnswerResponse

from sciencerag.common.config import get_embedding_model, get_llm_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "papers"
INDEX_DIR = REPO_ROOT / ".pqa_index"


def build_settings() -> Settings:
    settings = Settings(
        llm=get_llm_model(),
        embedding=get_embedding_model(),
        paper_directory=str(CORPUS_DIR),
    )
    settings.agent.index.index_directory = str(INDEX_DIR)
    return settings


def run_query(query: str) -> AnswerResponse:
    return ask(query, settings=build_settings())
