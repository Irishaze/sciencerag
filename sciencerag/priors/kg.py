"""Knowledge-graph storage + retrieval for sciencerag.priors (spec §3.2) and
sciencerag.ask (spec §6).

Real graph storage (NetworkX + JSON stub through M1-M4, upgraded here for
M5 since `ask` needs actual subgraph traversal — see the M1-era docstring
this replaces). Persisted as a flat JSON triple list under data/kg/ rather
than a real graph database: the spec explicitly scopes this project's
graph to that until `ask` needs more (§6.1: "v1 无需修改 schema"), and a
flat file keeps the M1-M4 cold-start behavior (query_kg returns hits from
whatever is actually stored, [] when nothing is) unchanged in shape.

Writes only happen through add_triple(), and only sciencerag.validate's §4.4
KG candidates (via an approval step — scripts/approve_kg_candidates.py,
spec §7: "v1 可用命令行脚本替代页面") ever call it — this module itself does
not gate who is allowed to write; that's the caller's job (spec §6.3: the
only path into the graph is candidate -> approval -> registration).
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Literal
import uuid

import jieba

from pydantic import BaseModel, Field, model_validator

GRAPH_PATH = Path("data/kg/graph.json")

# spec §3.8's numeric-groundedness tolerance (±2%, "容忍 LLM 抄证据时的四舍
# 五入") reused here for the same reason: two triples for the same
# entity_id/relation should be treated as the same finding if they're
# within ordinary rounding/reporting noise of each other, not flagged as a
# spurious conflict.
DUPLICATE_VALUE_RELATIVE_TOLERANCE = 0.02

# entity_type when no ontology-driven classification was supplied (either
# because the caller didn't pass one, or classification couldn't place a
# relation into any known type) — kg.py itself doesn't read data/kg/
# ontology.json or validate against it, that's sciencerag.validate.
# kg_candidates.py's job; this is just a safe, always-valid default so
# add_triple() never fails writing data over a missing classification.
DEFAULT_ENTITY_TYPE = "Unclassified"


class KGHit(BaseModel):
    triple_id: str
    text: str
    relevance: float


class KGSource(BaseModel):
    type: Literal["paper", "kg_triple", "run"] = "run"
    doi: str | None = None
    triple_id: str | None = None
    run_id: str | None = None


class KGTriple(BaseModel):
    triple_id: str
    # entity_id is the actual graph identity — a deterministic hash of
    # (subject, conditions), computed by compute_entity_id() and always
    # set by add_triple(). `subject` used to double as identity AND display
    # label, which meant every distinct design (different `conditions`)
    # sharing the same subject string collapsed onto one graph node —
    # confirmed via a real repro: 5 seeded TEC designs, indistinguishable
    # to query_kg, one answer falsely claimed "no other triples exist" for
    # data that was simply never retrieved. entity_id individuates; subject
    # stays purely descriptive from here on.
    entity_id: str
    entity_type: str
    subject: str
    relation: str
    # A triple's object is EITHER a number (object_value, the original and
    # still most common case — "this design achieves 71.7K") OR a link to
    # another entity (object_entity_id) — never both. Without this,
    # relations like "this simulation run used this material" had nowhere
    # to go: object_value is a float, and "which material" isn't a number.
    # Mirrors RDF's own literal-vs-resource distinction on the object
    # position of a triple, for the same reason.
    object_value: float | None = None
    object_unit: str | None = None
    object_entity_id: str | None = None
    # Link triples carry the target's own display label/type redundantly
    # (rather than requiring a separate triple whose *subject* is that
    # entity to exist) so subgraph_from_triples can render a real node for
    # it — e.g. Material has no numeric "achieves_X" facts of its own, only
    # incoming links, so nothing else would ever introduce that node.
    object_entity_label: str | None = None
    object_entity_type: str | None = None
    # AI-generated plain-Chinese phrase for what `relation` means to a
    # non-specialist (sciencerag/priors/ontology_generator.py:
    # describe_relation) — e.g. "achieves_delta_T_max_K" -> "能降到多低的
    # 温度". Optional/nullable: kg.py itself never generates this (same
    # "computed one layer up" split as entity_type), and older triples
    # written before this field existed simply don't have one.
    relation_description: str | None = None
    conditions: dict[str, float] = Field(default_factory=dict)
    confidence: float
    run_ids: list[str] = Field(default_factory=list)
    sources: list[KGSource] = Field(default_factory=list)
    # Set only when this triple was stored because it *disagreed* with an
    # existing one for the same entity_id/relation (spec §4.4: "存在但数值
    # 冲突 → 标记冲突,双方来源并列呈现,不自动覆盖") — both triples stay in
    # the graph, neither is silently overwritten.
    conflicts_with: str | None = None
    created_at: str

    @model_validator(mode="after")
    def _exactly_one_object_kind(self) -> "KGTriple":
        has_value = self.object_value is not None
        has_link = self.object_entity_id is not None
        if has_value == has_link:
            raise ValueError(
                "exactly one of object_value or object_entity_id must be set, "
                f"got object_value={self.object_value!r} object_entity_id={self.object_entity_id!r}"
            )
        return self


def compute_entity_id(subject: str, conditions: dict[str, float]) -> str:
    """Deterministic identity for the real-world thing a triple's subject
    refers to: same subject + same conditions always hashes to the same
    entity_id, so re-submitting an already-known design's data resolves to
    the same graph node instead of a new one, and two designs that differ
    in even one geometry parameter get distinct nodes. `subject` is folded
    into the hash too — a no-op today (it's a hardcoded constant, see
    sciencerag/validate/kg_candidates.py:SUBJECT) but removes a latent trap
    for whenever this project stops being single-material/single-topology
    (sim_params.json's own scope_note flags that as explicit future scope)."""
    canonical = json.dumps({"subject": subject, "conditions": conditions}, sort_keys=True)
    return f"tec_{hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:12]}"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_triples() -> list[KGTriple]:
    if not GRAPH_PATH.exists():
        return []
    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    return [KGTriple.model_validate(item) for item in data]


def _save_triples(triples: list[KGTriple]) -> None:
    """Write-to-temp-then-rename rather than an in-place write_text(): a
    concurrent _load_triples() landing mid-write on a plain write_text()
    call can read a truncated or partially-overwritten file (confirmed via
    a real concurrency test: parallel add_triple() calls produced json
    parse errors like "Expecting value" and "Extra data" from readers
    catching a write in progress). os.replace() is atomic on POSIX, so a
    reader always sees either the fully-old or fully-new file, never a
    partial one."""
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=GRAPH_PATH.parent, prefix=".graph_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps([t.model_dump() for t in triples], indent=2, ensure_ascii=False))
        os.replace(tmp_name, GRAPH_PATH)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


@contextmanager
def _graph_lock() -> Iterator[None]:
    """add_triple()'s read-modify-write (load whole file, mutate, save
    whole file) is not itself atomic — two concurrent callers can both
    load the same "before" state, each add their own triple, and the
    second save silently clobbers the first's addition. Confirmed via a
    real test: 20 concurrent add_triple() calls against the same graph
    lost 14 of them. An OS-level advisory lock around the whole
    load-mutate-save cycle serializes concurrent callers (both concurrent
    HTTP-adjacent callers and, more realistically, two people running
    scripts/approve_kg_candidates.py at the same time) so no addition is
    silently dropped. fcntl.flock is POSIX-only, matching this project's
    actual deployment target (Docker/Linux container, Mac dev)."""
    lock_path = GRAPH_PATH.parent / (GRAPH_PATH.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _find_matching_condition_triple(
    triples: list[KGTriple], entity_id: str, relation: str
) -> KGTriple | None:
    # Keyed on (entity_id, relation) — relation must stay part of the key:
    # entity_id alone would collapse triples that share sparse/identical
    # conditions (e.g. conditions={}) but describe different relations,
    # routing them into merge/conflict instead of adding independently.
    # Confirmed against tests/test_kg_graph.py's 20-way concurrency test,
    # which does exactly that (20 distinct relations, all conditions={}).
    for triple in triples:
        if triple.entity_id == entity_id and triple.relation == relation and triple.conflicts_with is None:
            return triple
    return None


def _values_agree(existing: KGTriple, object_value: float | None, object_entity_id: str | None) -> bool:
    """Whether a newly-submitted object (either kind) counts as "the same
    fact" as an already-stored triple, for merge-vs-conflict purposes. A
    link's target either matches exactly or it doesn't — there's no
    rounding-noise concept for "which entity", unlike a measured number."""
    if existing.object_value is not None and object_value is not None:
        tolerance = DUPLICATE_VALUE_RELATIVE_TOLERANCE * max(abs(existing.object_value), 1e-9)
        return abs(existing.object_value - object_value) <= tolerance
    return existing.object_entity_id == object_entity_id


def add_triple(
    *,
    subject: str,
    relation: str,
    conditions: dict[str, float],
    confidence: float,
    run_id: str,
    sources: list[KGSource],
    object_value: float | None = None,
    object_unit: str | None = None,
    object_entity_id: str | None = None,
    object_entity_label: str | None = None,
    object_entity_type: str | None = None,
    entity_type: str = DEFAULT_ENTITY_TYPE,
    relation_description: str | None = None,
) -> tuple[KGTriple, Literal["added", "merged", "conflict"]]:
    """The only write path into the graph (spec §6.3). Callers are expected
    to have already run this through human approval.

    Exactly one of object_value or object_entity_id must be given — a
    numeric fact ("this design achieves 71.7K") or a link to another
    entity ("this run used this material"), never both (see KGTriple's own
    validator, which is the actual enforcement; this raises earlier with a
    clearer message for the common case).

    `entity_type` is optional and defaults to DEFAULT_ENTITY_TYPE: kg.py
    doesn't itself know about the AI-generated ontology (data/kg/
    ontology.json) or validate against it — that classification happens
    one layer up, in sciencerag.validate.kg_candidates.py, which is the
    only real caller with enough context to pick a real type. Existing
    callers that don't pass one (tests, demo scripts) keep working
    unchanged."""
    if (object_value is None) == (object_entity_id is None):
        raise ValueError("pass exactly one of object_value or object_entity_id")
    if object_value is not None and not math.isfinite(object_value):
        # Last gate before permanent storage — API-level input validation
        # (ValidateRequest/ReportRequest) catches this for the normal path,
        # but scripts/approve_kg_candidates.py --file accepts a hand-
        # assembled JSON file that never goes through that validation, so
        # this check is the one thing every write path actually shares.
        raise ValueError(f"object_value must be finite, got {object_value!r}")
    entity_id = compute_entity_id(subject, conditions)
    with _graph_lock():
        triples = _load_triples()
        existing = _find_matching_condition_triple(triples, entity_id, relation)

        if existing is not None:
            if _values_agree(existing, object_value, object_entity_id):
                if run_id not in existing.run_ids:
                    existing.run_ids.append(run_id)
                existing.sources.extend(sources)
                _save_triples(triples)
                return existing, "merged"

            conflict = KGTriple(
                triple_id=f"kg_{uuid.uuid4().hex[:12]}",
                entity_id=entity_id,
                entity_type=entity_type,
                subject=subject,
                relation=relation,
                object_value=object_value,
                object_unit=object_unit,
                object_entity_id=object_entity_id,
                object_entity_label=object_entity_label,
                object_entity_type=object_entity_type,
                relation_description=relation_description,
                conditions=conditions,
                confidence=confidence,
                run_ids=[run_id],
                sources=sources,
                conflicts_with=existing.triple_id,
                created_at=_now_iso(),
            )
            triples.append(conflict)
            _save_triples(triples)
            return conflict, "conflict"

        new_triple = KGTriple(
            triple_id=f"kg_{uuid.uuid4().hex[:12]}",
            entity_id=entity_id,
            entity_type=entity_type,
            subject=subject,
            relation=relation,
            object_value=object_value,
            object_unit=object_unit,
            object_entity_id=object_entity_id,
            object_entity_label=object_entity_label,
            object_entity_type=object_entity_type,
            relation_description=relation_description,
            conditions=conditions,
            confidence=confidence,
            run_ids=[run_id],
            sources=sources,
            created_at=_now_iso(),
        )
        triples.append(new_triple)
        _save_triples(triples)
        return new_triple, "added"


def _render_text(triple: KGTriple) -> str:
    conditions = ", ".join(f"{k}={v}" for k, v in sorted(triple.conditions.items()))
    if triple.object_value is not None:
        unit = triple.object_unit or ""
        obj = f"{triple.object_value}{unit}"
    else:
        obj = triple.object_entity_label or triple.object_entity_id or ""
    # relation_description (e.g. "最优电流") is the only Chinese-language
    # content anywhere on a triple — relation/subject/conditions are all
    # English contract identifiers. Without it here, a Chinese natural-
    # language query has nothing Chinese to overlap with on this side no
    # matter how well the query itself gets tokenized (confirmed via a real
    # repro: "最优电流是什么" scored 0 against every triple even after CJK
    # tokenization was added, until this was included).
    description = f" {triple.relation_description}" if triple.relation_description else ""
    return f"{triple.subject} {triple.relation}{description} {obj} ({conditions})"


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]+")

# Function words jieba's search-mode segmentation routinely produces from
# ordinary question phrasing ("是什么", "如何", "使...最大") that would
# otherwise inflate len(query_terms) in _score_triples' overlap/len(query)
# scoring — diluting genuine content-word matches below
# sciencerag.ask.pipeline.MIN_KG_RELEVANCE (0.5) for no reason other than
# the question containing normal grammar. Content words (nouns/verbs the
# domain actually uses, e.g. "电流"/"温差"/"优化") are never in this list.
_CJK_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "这", "那", "之",
    "什么", "怎么", "怎样", "如何", "为什么", "多少", "多大",
    "使", "让", "能", "可以", "要", "会", "有", "对", "把", "被",
    "吗", "呢", "啊", "吧", "地", "得", "着", "过", "也", "都", "就",
    "上", "下", "中", "个", "些", "才", "还", "又", "并", "而",
}


def _tokenize(text: str) -> set[str]:
    tokens = set(_TOKEN_RE.findall(text.lower()))
    if _CJK_RE.search(text):
        tokens.update(
            word
            for word in jieba.cut_for_search(text)
            if _CJK_RE.fullmatch(word) and word not in _CJK_STOPWORDS
        )
    return tokens


def _score_triples(query: str, triples: list[KGTriple]) -> list[tuple[KGTriple, float]]:
    """Keyword-overlap scoring shared by query_kg and query_kg_entities —
    deliberately simple (no embeddings/vector index for the graph, spec
    §6.1 keeps v1's schema minimal). CJK text goes through jieba
    (_tokenize) rather than embeddings — same "simple keyword overlap"
    architecture, just with a real word boundary for Chinese instead of
    the regex-only ASCII splitting below.

    Tokenizes on alphanumeric runs — splits on whitespace/punctuation AND
    underscores. Two things this fixes, both confirmed live against real
    data before this existed: (1) _render_text glues a condition's
    parameter name directly to its value with no separator
    ("...(leg_length=0.8, pitch=0.29)"), so whitespace-only splitting made
    the whitespace-delimited token the literal string "leg_length=0.8," in
    full, never equal to a query term "leg_length"; (2) relation strings
    are one contiguous underscore-joined identifier
    ("literature_range_leg_length"), so even after fixing (1), a bare
    "leg_length" query still couldn't set-match that whole string — also
    splitting on "_" turns it into {"literature", "range", "leg",
    "length"}, which a "leg_length" query ({"leg", "length"}) now
    genuinely overlaps. Both are the case sciencerag/priors/retrieval.py's
    _kg_priors_for_query needs: querying by a bare contract parameter
    name is the most natural way to ask about one. Returns only triples
    with nonzero overlap, unsorted.
    """
    query_terms = _tokenize(query)
    if not query_terms:
        return []
    scored = []
    for triple in triples:
        text_terms = _tokenize(_render_text(triple))
        overlap = query_terms & text_terms
        if not overlap:
            continue
        scored.append((triple, len(overlap) / len(query_terms)))
    return scored


def query_kg(query: str, max_hits: int = 10) -> list[KGHit]:
    """Per-triple keyword-overlap search. Empty graph or no overlapping
    terms both correctly return [] (spec §3.2 cold-start: "图谱查询命中 0
    条,检索自动落到论文库"). Used as-is by sciencerag.validate.
    kg_candidates.py's dedup check, which wants per-triple relevance, not
    entity grouping — see query_kg_entities() for the grouped version
    sciencerag.ask uses."""
    triples = _load_triples()
    if not triples:
        return []
    scored = _score_triples(query, triples)
    hits = [
        KGHit(triple_id=triple.triple_id, text=_render_text(triple), relevance=score)
        for triple, score in scored
    ]
    hits.sort(key=lambda hit: hit.relevance, reverse=True)
    return hits[:max_hits]


class KGEntityGroup(BaseModel):
    entity_id: str
    relevance: float
    triples: list[KGTriple]


class KGEntityQueryResult(BaseModel):
    groups: list[KGEntityGroup]
    total_matching_entities: int
    entities_returned: int


def query_kg_entities(query: str, max_entities: int = 10) -> KGEntityQueryResult:
    """Like query_kg, but grouped by entity_id so an answer is never a
    partial slice of one design's data cut off mid-entity by a row-based
    cap. Confirmed via a real repro: a generic question against a 5-design,
    30-triple graph got max_hits=10 rows spanning parts of 2 designs, and
    the LLM confidently claimed "no other triples provide additional
    values" — 3 more designs' worth of data existed but were never
    retrieved into its context at all. Here, once an entity is selected
    (because at least one of its triples matched the query), ALL of that
    entity's triples come back together — not just the ones that
    individually scored a keyword hit — and total_matching_entities vs
    entities_returned lets the caller tell the truth about what got cut."""
    triples = _load_triples()
    if not triples:
        return KGEntityQueryResult(groups=[], total_matching_entities=0, entities_returned=0)
    scored = _score_triples(query, triples)
    if not scored:
        return KGEntityQueryResult(groups=[], total_matching_entities=0, entities_returned=0)

    best_score: dict[str, float] = {}
    for triple, score in scored:
        best_score[triple.entity_id] = max(best_score.get(triple.entity_id, 0.0), score)

    all_by_entity: dict[str, list[KGTriple]] = {}
    for triple in triples:
        all_by_entity.setdefault(triple.entity_id, []).append(triple)

    ranked_entity_ids = sorted(best_score, key=lambda eid: best_score[eid], reverse=True)
    selected = ranked_entity_ids[:max_entities]
    groups = [
        KGEntityGroup(entity_id=eid, relevance=best_score[eid], triples=all_by_entity[eid])
        for eid in selected
    ]
    return KGEntityQueryResult(
        groups=groups,
        total_matching_entities=len(ranked_entity_ids),
        entities_returned=len(selected),
    )


def get_triples_by_ids(triple_ids: list[str]) -> list[KGTriple]:
    id_set = set(triple_ids)
    return [t for t in _load_triples() if t.triple_id in id_set]


def subgraph_from_triples(triples: list[KGTriple]) -> dict:
    """Nodes/edges for a given triple set — the response shape
    sciencerag.ask returns alongside its answer (spec §6.2: "返回答案并附带
    子图"). Entity-node id is entity_id (opaque) with `subject` carried as
    `label` for display — see KGTriple.entity_id's docstring for why id and
    label used to be the same field and had to be split. Value-node id is
    scoped by triple_id rather than bare f"{relation}={value}": two
    different entities that happen to produce an identical relation+value
    used to collapse onto one shared node in the rendered graph.

    A link triple (object_entity_id set) renders as a real entity-to-entity
    edge instead of an entity-to-value one — this is what actually lets two
    different subject entities (e.g. two TECDesigns that share one Material)
    show up connected in the graph rather than as disconnected star
    clusters, which was a real complaint: with only numeric-fact triples,
    every entity only ever had edges to its own private value nodes and
    never to anything another entity could also point at."""
    nodes = {}
    edges = []
    for triple in triples:
        nodes.setdefault(
            triple.entity_id,
            {
                "id": triple.entity_id,
                "kind": "entity",
                "label": triple.subject,
                "entity_type": triple.entity_type,
            },
        )
        if triple.object_entity_id is not None:
            nodes.setdefault(
                triple.object_entity_id,
                {
                    "id": triple.object_entity_id,
                    "kind": "entity",
                    "label": triple.object_entity_label or triple.object_entity_id,
                    "entity_type": triple.object_entity_type,
                },
            )
            target = triple.object_entity_id
        else:
            # Deliberately just the number, not prefixed by relation/
            # relation_description here — the edge already carries that
            # (relation + description fields), and the frontend combines
            # "{edge description}：{this label}" when displaying either the
            # edge or the node's connections. Prefixing it here too doubled
            # up as "最多能降温多少度：最多能降温多少度：58.2K" (found
            # 2026-08-11) once the frontend started doing that combining.
            target = f"{triple.triple_id}:value"
            nodes.setdefault(
                target,
                {
                    "id": target,
                    "kind": "value",
                    "label": f"{triple.object_value}{triple.object_unit or ''}",
                    "entity_type": None,
                },
            )
        edges.append(
            {
                "source": triple.entity_id,
                "target": target,
                "relation": triple.relation,
                "description": triple.relation_description,
                "triple_id": triple.triple_id,
                "confidence": triple.confidence,
                "conflicts_with": triple.conflicts_with,
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges}


def full_graph() -> dict:
    """subgraph_from_triples() over every stored triple — the whole
    accumulated graph, not scoped to one question. Backs the frontend's
    standalone graph-browsing page (GET /sciencerag/graph); read-only, same
    as every other graph access path (spec §6.3's write-path constraint is
    untouched by this — it only reads)."""
    return subgraph_from_triples(_load_triples())


def get_subgraph(entities: list[str]) -> dict:
    """subgraph_from_triples(), scoped to triples touching any of
    `entities` by subject match."""
    triples = _load_triples()
    matched = [t for t in triples if t.subject in entities or any(e in t.relation for e in entities)]
    return subgraph_from_triples(matched)


def backfill_entity_fields(default_entity_type: str = DEFAULT_ENTITY_TYPE) -> int:
    """One-off migration for triples stored before entity_id/entity_type
    existed. Deliberately operates on raw JSON dicts, NOT through
    _load_triples()/KGTriple.model_validate(): both fields are required on
    KGTriple, so loading old records through the strict model would fail
    validation before this function ever got a chance to backfill them —
    this has to stay a raw-JSON transform for exactly that reason. Reuses
    _graph_lock() for the same torn-write/lost-update protection every
    other write path gets (see _graph_lock's and _save_triples' own
    docstrings for the concurrency bugs those fixed)."""
    with _graph_lock():
        if not GRAPH_PATH.exists():
            return 0
        data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        updated = 0
        for item in data:
            changed = False
            if not item.get("entity_id"):
                item["entity_id"] = compute_entity_id(item["subject"], item["conditions"])
                changed = True
            if not item.get("entity_type"):
                item["entity_type"] = default_entity_type
                changed = True
            if changed:
                updated += 1
        if updated:
            GRAPH_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return updated
