"""PaperQA2 wiring for sciencerag.priors (spec §3.2).

Builds a paper-qa Settings object from our LLM/embedding config (§9 OQ#2)
pointed at the internal literature corpus (corpus/papers/), and exposes a
thin query function. The retrieval index is cached under .pqa_index/
(project-local, gitignored) rather than the shared ~/.pqa/ cache so this
project's index doesn't mix with unrelated projects.
"""

from pathlib import Path

from paperqa import Context, Settings, ask
from paperqa.agents.main import AnswerResponse

from sciencerag.common.config import get_embedding_model, get_llm_model
from sciencerag.common.trace import new_trace_id
from sciencerag.priors.classify import classify
from sciencerag.priors.models import Coverage, Prior, PriorsResponse, SourcePaper

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "papers"
INDEX_DIR = REPO_ROOT / ".pqa_index"


def build_settings() -> Settings:
    llm = get_llm_model()
    settings = Settings(
        llm=llm,
        summary_llm=llm,
        embedding=get_embedding_model(),
        paper_directory=str(CORPUS_DIR),
    )
    # paper-qa defaults summary_llm/agent_llm to OpenAI's gpt-4o independently
    # of the top-level `llm` field. summary_llm=DeepSeek works fine, but
    # agent_llm=DeepSeek gets stuck: it never emits a "complete" tool call and
    # loops on generate_answer indefinitely (confirmed — burned $0.25+ before
    # being killed). Leave agent_llm on its OpenAI default; only llm and
    # summary_llm are DeepSeek.
    settings.agent.index.index_directory = str(INDEX_DIR)
    return settings


def run_query(query: str) -> AnswerResponse:
    return ask(query, settings=build_settings())


def _prior_from_context(context: Context) -> Prior:
    doc = context.text.doc
    kind, field = classify(context.context)
    return Prior(
        prior_id=f"pr_{context.id}",
        kind=kind,
        field=field,
        value={"summary": context.context},
        confidence=max(0.0, min(1.0, context.score / 10)),
        sources=[SourcePaper(doi=getattr(doc, "doi", None) or "", span=context.text.name)],
        notes=getattr(doc, "title", None),
    )


def build_priors_response(query: str) -> PriorsResponse:
    """Run a real PaperQA2 query and map its evidence contexts into priors."""
    response = run_query(query)
    contexts = response.session.contexts
    return PriorsResponse(
        priors=[_prior_from_context(context) for context in contexts],
        coverage=Coverage(internal_hits=len(contexts), external_hits=0, gaps=[]),
        trace_id=new_trace_id(),
    )
