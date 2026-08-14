"""Answer synthesis for sciencerag.ask (spec §6.2): entity/relation lookup
against the graph -> subgraph assembly -> grounded LLM synthesis, or a
documented fallback to component A's document retrieval when the graph has
no matching triples.
"""

from __future__ import annotations

import re

import litellm

from sciencerag.common.config import get_llm_model
from sciencerag.common.text_clean import clean_text
from sciencerag.priors.kg import (
    KGRankingResult,
    KGTriple,
    query_kg_entities,
    rank_kg_entities,
    subgraph_from_triples,
)
from sciencerag.priors.models import Source, SourceKGTriple, SourcePaper
from sciencerag.priors.retrieval import run_query

# Matches sciencerag/priors/extract.py's REQUEST_TIMEOUT_SECONDS convention
# for the same underlying litellm/DeepSeek call.
REQUEST_TIMEOUT_SECONDS = 90

# query_kg_entities' relevance is keyword-overlap over query_terms (kg.py),
# so a query that shares just one incidental token with every stored
# entity's subject (e.g. "TEC") scores the same nonzero relevance against
# the whole graph regardless of whether any triple actually answers the
# question. Below this bar we don't trust the graph alone — mirrors
# sciencerag.priors.retrieval.MIN_EVIDENCE_RELEVANCE (0.5), same reasoning:
# an overlap score gates whether evidence is treated as authoritative, not
# just "nonzero".
MIN_KG_RELEVANCE = 0.5

SYSTEM_PROMPT = (
    "You answer questions about a thermoelectric-cooler (TEC) knowledge "
    "graph using ONLY the triples listed below — do not use outside "
    "knowledge. Every claim in your answer must be traceable to one of the "
    "given triples (cite it by triple_id). If the triples don't fully "
    "answer the question, say plainly what's missing instead of guessing. "
    "You will be told how many matching designs exist in total versus how "
    "many are included below — if that coverage line says designs were "
    "left out, say so explicitly in your answer. Never claim there is no "
    "other data unless the coverage line says the graph genuinely has "
    "nothing else. If a 'Ranking' block is given, it was computed by "
    "actually sorting every candidate's numeric value, not by keyword "
    "matching — treat it as the authoritative answer to 'which one is "
    "best/highest/lowest' and cite it directly; do not re-derive your own "
    "ranking from the raw triples below it, and do not silently ignore it. "
    "Some triples carry an 'evidence_detail' note — e.g. a triple validated "
    "against a real benchmark case at 0.8% deviation vs one at 4.9% "
    "deviation can carry the same confidence number despite very different "
    "real support; when evidence_detail is present on a triple you cite, "
    "reference it (how close the match actually was, which benchmark "
    "case) rather than just quoting the bare confidence number."
)


def _format_triples(triples: list[KGTriple]) -> str:
    def _object(t: KGTriple) -> str:
        if t.object_value is not None:
            return f"{t.object_value}{t.object_unit or ''}"
        return t.object_entity_label or t.object_entity_id or ""

    def _evidence_detail(t: KGTriple) -> str:
        if not t.evidence_detail:
            return ""
        d = t.evidence_detail
        pct = d.get("relative_deviation")
        pct_str = f"{pct * 100:.1f}%" if isinstance(pct, (int, float)) else "?"
        return f", evidence_detail: {pct_str} deviation vs benchmark case {d.get('benchmark_case_id', '?')}"

    return "\n".join(
        f"- [{t.triple_id}] {t.subject} {t.relation} = {_object(t)} "
        f"(conditions: {t.conditions or 'none'}, confidence: {t.confidence}{_evidence_detail(t)})"
        for t in triples
    )


def _format_ranking(ranking: KGRankingResult) -> str:
    label = ranking.relation_description or ranking.relation
    direction_word = "highest" if ranking.direction == "max" else "lowest"
    lines = [
        f"Ranking by {label} ({ranking.relation}), {direction_word} first — "
        f"{ranking.total_candidates} candidate(s) actually have this value:"
    ]
    for rank, entity in enumerate(ranking.ranked, start=1):
        lines.append(
            f"  {rank}. [{entity.triple.triple_id}] entity {entity.entity_id}: "
            f"{entity.value}{entity.unit or ''}"
        )
    return "\n".join(lines)


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


def _answer_from_graph(
    question: str,
    triples: list[KGTriple],
    total_matching_entities: int,
    entities_shown: int,
    ranking: KGRankingResult | None = None,
) -> str:
    coverage_line = (
        f"Coverage: all {total_matching_entities} matching design(s) are included below in full."
        if entities_shown >= total_matching_entities
        else (
            f"Coverage: only {entities_shown} of {total_matching_entities} matching design(s) "
            "are included below — the rest exist in the graph but are not shown here."
        )
    )
    ranking_block = f"\n\nRanking:\n{_format_ranking(ranking)}" if ranking else ""
    user_prompt = (
        f"Question: {question}\n\n{coverage_line}{ranking_block}\n\n"
        f"Known triples:\n{_format_triples(triples)}\n\nAnswer:"
    )
    return _call_llm(SYSTEM_PROMPT, user_prompt)


_CITATION_ID_RE = re.compile(r"\bpqac-[a-zA-Z0-9]{8}\b")
_PARENTHETICAL_RE = re.compile(r"\(([^()]*)\)")


def _backfill_citations(answer: str, contexts: list) -> str:
    """paper-qa's own PQASession.populate_formatted_answers_and_bib_from_raw_answer
    (paperqa/types.py) only resolves a pqac-xxxxxxxx citation key into its
    source docname when the key sits inside a parenthetical matching its
    exact CITATION_KEY_CONSTRAINTS grammar; anything looser — and our
    custom answer_length formatting instructions in priors/retrieval.py
    make that more likely, by asking the model to restructure paragraphs
    and equations around its baseline prompt — passes the raw pqac key
    straight through into session.answer unresolved. This is a second,
    more lenient pass over whatever paper-qa's own step left behind: any
    pqac-xxxxxxxx token still in the text, inside parens or not, gets
    replaced by its source docname, or dropped if it doesn't match any
    real context (a hallucinated key — same handling paper-qa itself uses
    for those)."""
    if not _CITATION_ID_RE.search(answer):
        return answer

    id_to_name = {c.id: c.text.name for c in contexts}

    def _sub_parenthetical(match: re.Match) -> str:
        inner = match.group(1)
        if not _CITATION_ID_RE.search(inner):
            return match.group(0)
        names = dict.fromkeys(
            id_to_name[key] for key in _CITATION_ID_RE.findall(inner) if key in id_to_name
        )
        return f"({', '.join(names)})" if names else ""

    answer = _PARENTHETICAL_RE.sub(_sub_parenthetical, answer)
    return _CITATION_ID_RE.sub(lambda m: id_to_name.get(m.group(0), ""), answer)


def answer_question(question: str, max_hits: int = 10) -> dict:
    result = query_kg_entities(question, max_entities=max_hits)
    strong_groups = [g for g in result.groups if g.relevance >= MIN_KG_RELEVANCE]
    # Separate from strong_groups on purpose: a comparison question's #1
    # result can legitimately have low text-relevance (its subject/
    # conditions may share few keywords with the query) while still being
    # exactly the entity the ranking asked for — text-overlap and
    # "is this the right answer" are different questions once an explicit
    # superlative word is present (see detect_ranking_direction).
    ranking = rank_kg_entities(question)

    if strong_groups or ranking:
        # Each selected group already carries ALL of its entity's triples
        # (query_kg_entities' whole point — see its docstring), never a
        # partial slice cut off mid-entity.
        triples = [t for group in strong_groups for t in group.triples]
        if ranking:
            ranked_triples = [entity.triple for entity in ranking.ranked]
            triples = list({t.triple_id: t for t in triples + ranked_triples}.values())
        entities_shown = len({t.entity_id for t in triples})
        answer = clean_text(
            _answer_from_graph(
                question, triples, result.total_matching_entities, entities_shown, ranking
            )
        )
        sources: list[Source] = [SourceKGTriple(triple_id=t.triple_id) for t in triples]
        coverage_note = (
            None
            if entities_shown >= result.total_matching_entities
            else (
                f"knowledge graph has {result.total_matching_entities} matching design(s); "
                f"only the {entities_shown} most relevant are reflected in this answer"
            )
        )
        return {
            "answer": answer,
            "subgraph": subgraph_from_triples(triples),
            "sources": sources,
            "fallback_used": False,
            "coverage_note": coverage_note,
        }

    # spec §6.2 cold-start fallback: no (sufficiently relevant) triples yet ->
    # component A's document retrieval (sciencerag.priors.retrieval.run_query),
    # which already does its own grounded synthesis over the literature corpus.
    response = run_query(question)
    contexts = response.session.contexts
    answer = clean_text(_backfill_citations(response.session.answer, contexts))

    # session.contexts is every piece of evidence PaperQA *retrieved*, not
    # what the model actually cited in the final answer — gather_evidence
    # routinely pulls in more context than generate_answer ends up using.
    # Listing all of it as "sources" overstates support for claims the
    # answer never made (confirmed against a real response: 2 of 3 listed
    # sources didn't appear anywhere in the answer text). A context's
    # docname is exactly the substring paper-qa (or our own backfill above)
    # writes into the text when it's actually cited, so membership here is
    # exact, not a heuristic.
    sources: list[Source] = []
    seen_dois: set[str] = set()
    for context in contexts:
        if not context.text.name or context.text.name not in answer:
            continue
        doi = getattr(context.text.doc, "doi", None)
        if doi and doi not in seen_dois:
            seen_dois.add(doi)
            sources.append(SourcePaper(doi=doi, span=context.text.name))
    if result.groups:
        coverage_note = (
            f"knowledge graph had {result.total_matching_entities} matching design(s) but none "
            f"reached the minimum relevance ({MIN_KG_RELEVANCE}) to answer from — "
            "answered from the literature corpus instead"
        )
    else:
        coverage_note = (
            "knowledge graph had no matching triples for this question — "
            "answered from the literature corpus instead"
        )
    return {
        "answer": answer,
        "subgraph": {"nodes": [], "edges": []},
        "sources": sources,
        "fallback_used": True,
        "coverage_note": coverage_note,
    }
