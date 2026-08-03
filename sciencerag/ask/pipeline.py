"""Answer synthesis for sciencerag.ask (spec §6.2): entity/relation lookup
against the graph -> subgraph assembly -> grounded LLM synthesis, or a
documented fallback to component A's document retrieval when the graph has
no matching triples.
"""

from __future__ import annotations

import litellm

from sciencerag.common.config import get_llm_model
from sciencerag.priors.kg import KGTriple, get_triples_by_ids, query_kg, subgraph_from_triples
from sciencerag.priors.models import Source, SourceKGTriple, SourcePaper
from sciencerag.priors.retrieval import run_query

# Matches sciencerag/priors/extract.py's REQUEST_TIMEOUT_SECONDS convention
# for the same underlying litellm/DeepSeek call.
REQUEST_TIMEOUT_SECONDS = 90

SYSTEM_PROMPT = (
    "You answer questions about a thermoelectric-cooler (TEC) knowledge "
    "graph using ONLY the triples listed below — do not use outside "
    "knowledge. Every claim in your answer must be traceable to one of the "
    "given triples (cite it by triple_id). If the triples don't fully "
    "answer the question, say plainly what's missing instead of guessing."
)


def _format_triples(triples: list[KGTriple]) -> str:
    return "\n".join(
        f"- [{t.triple_id}] {t.subject} {t.relation} = "
        f"{t.object_value}{t.object_unit or ''} "
        f"(conditions: {t.conditions or 'none'}, confidence: {t.confidence})"
        for t in triples
    )


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    model = get_llm_model()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    try:
        response = litellm.completion(
            model=model, messages=messages, temperature=0, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except litellm.BadRequestError as e:
        # spec/extract.py precedent: some models reject a custom
        # temperature outright — fall back to the model's default rather
        # than treating it as a real failure.
        if "temperature" not in str(e):
            raise
        response = litellm.completion(model=model, messages=messages, timeout=REQUEST_TIMEOUT_SECONDS)
    return response.choices[0].message.content


def _answer_from_graph(question: str, triples: list[KGTriple]) -> str:
    user_prompt = f"Question: {question}\n\nKnown triples:\n{_format_triples(triples)}\n\nAnswer:"
    return _call_llm(SYSTEM_PROMPT, user_prompt)


def answer_question(question: str, max_hits: int = 10) -> dict:
    hits = query_kg(question, max_hits=max_hits)
    if hits:
        triples = get_triples_by_ids([hit.triple_id for hit in hits])
        answer = _answer_from_graph(question, triples)
        sources: list[Source] = [SourceKGTriple(triple_id=t.triple_id) for t in triples]
        return {
            "answer": answer,
            "subgraph": subgraph_from_triples(triples),
            "sources": sources,
            "fallback_used": False,
            "coverage_note": None,
        }

    # spec §6.2 cold-start fallback: no matching triples yet -> component
    # A's document retrieval (sciencerag.priors.retrieval.run_query), which
    # already does its own grounded synthesis over the literature corpus.
    response = run_query(question)
    contexts = response.session.contexts
    sources: list[Source] = []
    seen_dois: set[str] = set()
    for context in contexts:
        doi = getattr(context.text.doc, "doi", None)
        if doi and doi not in seen_dois:
            seen_dois.add(doi)
            sources.append(SourcePaper(doi=doi, span=context.text.name))
    return {
        "answer": response.session.answer,
        "subgraph": {"nodes": [], "edges": []},
        "sources": sources,
        "fallback_used": True,
        "coverage_note": (
            "knowledge graph had no matching triples for this question — "
            "answered from the literature corpus instead"
        ),
    }
