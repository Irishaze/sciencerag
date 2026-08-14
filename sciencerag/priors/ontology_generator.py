"""AI-generated entity/relation type ontology for the knowledge graph
(KGTriple.entity_type in sciencerag/priors/kg.py).

Identity (entity_id, kg.py's _compute_entity_id) and classification
(entity_type, here) are orthogonal — same split as MiroFish's
EntityNode{uuid, labels} (backend/app/services/zep_entity_reader.py:22-27):
uuid individuates, labels classify. This module only produces the type
*schema*; per-item classification against it happens one layer up in
sciencerag/validate/kg_candidates.py.

Unlike MiroFish's ontology_generator.py, which analyzes free-text research
literature per project (an LLM reading PDFs to invent categories for an
open-ended domain), sciencerag's KG only ever describes one thing today:
Bi2Te3 single-stage TEC designs and the scalar results a COMSOL run
produces for them (sim_params.json's own scope_note: "材料固定 Bi2Te3,不做
材料探索"). So the input here is that structured contract, not documents,
and the ontology is generated once (or regenerated on demand if the
contract changes), not per data point.

Also unlike MiroFish, which hard-requires exactly 10 entity types — that's
a Zep API limit (backend/app/services/graph_builder.py:321-323: "Zep API
限制：最多10个自定义实体类型"), not a design principle — this lets the
model size the ontology to what the contract actually contains. The one
rule kept from MiroFish's approach (genuinely worth keeping, not an
artifact of Zep): the ontology MUST include a fallback/catch-all entity
type, so classification never has to force something into a category it
doesn't belong to.
"""

from __future__ import annotations

import json
from pathlib import Path

import litellm
from pydantic import BaseModel, Field, model_validator

from sciencerag.common.config import get_llm_model
from sciencerag.priors.contract import CONTRACT
from sciencerag.validate.tec_bridge import SCALAR_UNITS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ONTOLOGY_PATH = REPO_ROOT / "data" / "kg" / "ontology.json"
CLASSIFICATION_CACHE_PATH = REPO_ROOT / "data" / "kg" / "entity_type_cache.json"
DESCRIPTION_CACHE_PATH = REPO_ROOT / "data" / "kg" / "relation_description_cache.json"

# Matches sciencerag/priors/extract.py's REQUEST_TIMEOUT_SECONDS convention
# for the same underlying litellm/DeepSeek call.
REQUEST_TIMEOUT_SECONDS = 90

_FALLBACK_TYPE_NAME = "Concept"

SYSTEM_PROMPT = """You are designing an entity/relation type ontology for a knowledge graph that will describe thermoelectric-cooler (TEC) simulation designs and their results.

You will be given the project's simulation parameter contract (which parameter categories exist, what geometry a design has) and the list of scalar result fields a simulation run can produce. Design a SMALL, appropriately-sized set of entity types and relation types that fit this specific, narrow domain — do not pad the list to hit some target count, and do not invent entity types for things outside what's described (no materials other than what's given, no operating conditions, no numerical solver settings).

Rules:
1. You MUST include exactly one fallback/catch-all entity type (set "is_fallback": true on it) for anything that doesn't fit the more specific types — never force something into a specific type it doesn't really belong to.
2. Entity type names: PascalCase. Relation type names: UPPER_SNAKE_CASE.
3. Every entity type needs a one-sentence description and, where relevant, a short list of attribute names (snake_case) it carries.
4. Every relation type needs a source and target entity type name (both must be names you defined in entity_types).
5. Output valid JSON only, no markdown code fences, no commentary — exactly this shape:

{"entity_types": [{"name": "...", "description": "...", "attributes": ["...", ...], "is_fallback": false}, ...],
 "edge_types": [{"name": "...", "description": "...", "source": "...", "target": "..."}, ...],
 "analysis_summary": "one paragraph explaining the design"}
"""


class OntologyEntityType(BaseModel):
    name: str
    description: str
    attributes: list[str] = Field(default_factory=list)
    is_fallback: bool = False


class OntologyEdgeType(BaseModel):
    name: str
    description: str
    source: str
    target: str


class Ontology(BaseModel):
    entity_types: list[OntologyEntityType]
    edge_types: list[OntologyEdgeType] = Field(default_factory=list)
    analysis_summary: str = ""

    @model_validator(mode="after")
    def _exactly_one_fallback(self) -> "Ontology":
        fallbacks = [t for t in self.entity_types if t.is_fallback]
        if len(fallbacks) == 1:
            return self
        if len(fallbacks) == 0:
            # Same defensive injection MiroFish's own
            # _validate_and_process does when the model forgets the
            # fallback types it was told to include (ontology_generator.py
            # concept_fallback/researcher_fallback dicts) — don't reject a
            # whole generation over one missing field, just add it.
            self.entity_types.append(
                OntologyEntityType(
                    name=_FALLBACK_TYPE_NAME,
                    description="Anything that doesn't fit a more specific entity type.",
                    is_fallback=True,
                )
            )
            return self
        # More than one — keep the first, un-flag the rest rather than
        # rejecting outright (ambiguous but not actually broken).
        seen_one = False
        for t in self.entity_types:
            if t.is_fallback:
                if seen_one:
                    t.is_fallback = False
                seen_one = True
        return self

    def fallback_entity_type(self) -> str:
        return next(t.name for t in self.entity_types if t.is_fallback)

    def entity_type_names(self) -> set[str]:
        return {t.name for t in self.entity_types}


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _build_user_message() -> str:
    geometry_free = CONTRACT["geometry_free"]
    categories = CONTRACT["categories"]
    lines = [
        f"## Domain scope\n{CONTRACT.get('scope_note', '')}",
        "\n## Parameter categories\n"
        + "\n".join(f"- {name}: {info['description']}" for name, info in categories.items()),
        "\n## Free geometry parameters (what varies between designs)\n"
        + "\n".join(f"- {p['name']} ({p['unit']}): {p['desc']}" for p in geometry_free),
        "\n## Scalar result fields a simulation run can produce\n"
        + "\n".join(f"- {field} ({unit})" for field, unit in SCALAR_UNITS.items()),
    ]
    return "\n".join(lines)


def generate_ontology() -> Ontology:
    """One LLM call, real cost — call this explicitly (scripts/
    generate_kg_ontology.py), never implicitly from a hot request path."""
    model = get_llm_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message()},
    ]
    try:
        response = litellm.completion(
            model=model, messages=messages, temperature=0, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except litellm.BadRequestError as e:
        if "temperature" not in str(e):
            raise
        response = litellm.completion(model=model, messages=messages, timeout=REQUEST_TIMEOUT_SECONDS)
    raw = response.choices[0].message.content
    data = json.loads(_strip_code_fences(raw))
    return Ontology.model_validate(data)


def save_ontology(ontology: Ontology) -> None:
    ONTOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ONTOLOGY_PATH.write_text(
        json.dumps(ontology.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_ontology() -> Ontology | None:
    if not ONTOLOGY_PATH.exists():
        return None
    return Ontology.model_validate(json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8")))


CLASSIFY_SYSTEM_PROMPT = """Given a knowledge-graph ontology, classify a relation by what kind of entity type its SUBJECT is (not what the relation measures or points at).

Important context on what "subject" identifies in this graph: the subject of an "achieves_X" relation is keyed on design geometry alone (kg.py's entity_id = hash of subject + geometry conditions) — re-simulating the exact same geometry in a different run resolves to the SAME subject/node, not a new one; it is the persistent design being described, not one specific execution of a simulation. Don't classify based on the surface reading of "achieves" sounding like a run's output — classify based on what persists across repeated runs of the same design.

Reply with ONLY the entity type name from the ontology, nothing else — no punctuation, no explanation. If none of the specific types genuinely fit, reply with the fallback type name."""


def _load_classification_cache() -> dict[str, str]:
    if not CLASSIFICATION_CACHE_PATH.exists():
        return {}
    return json.loads(CLASSIFICATION_CACHE_PATH.read_text(encoding="utf-8"))


def _save_classification_cache(cache: dict[str, str]) -> None:
    CLASSIFICATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def classify_relation(relation: str, ontology: Ontology) -> str:
    """AI-classifies which entity type a relation's subject belongs to,
    cached by relation name in data/kg/entity_type_cache.json.

    sciencerag's relations today all come from a small, mostly-fixed set
    (sciencerag/validate/tec_bridge.py's SCALAR_UNITS — 6 known fields), so
    caching means an LLM call only fires the first time a given relation
    name is ever seen; every later run of the exact same simulation reuses
    the cached answer instead of re-asking (and risking a differently-
    worded, run-to-run-inconsistent answer — this project has already hit
    that exact flakiness elsewhere, see priors/retrieval.py's AGENT_SEED).
    A genuinely new relation (the simulator starts reporting a result field
    that's never been seen before) is exactly the case that should, and
    does, trigger real AI classification."""
    cache = _load_classification_cache()
    valid_types = ontology.entity_type_names()
    if relation in cache and cache[relation] in valid_types:
        return cache[relation]

    model = get_llm_model()
    ontology_block = "\n".join(f"- {t.name}: {t.description}" for t in ontology.entity_types)
    messages = [
        {
            "role": "system",
            "content": f"{CLASSIFY_SYSTEM_PROMPT}\n\nFallback type: {ontology.fallback_entity_type()}",
        },
        {"role": "user", "content": f"Ontology entity types:\n{ontology_block}\n\nRelation: {relation}"},
    ]
    try:
        response = litellm.completion(
            model=model, messages=messages, temperature=0, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except litellm.BadRequestError as e:
        if "temperature" not in str(e):
            raise
        response = litellm.completion(model=model, messages=messages, timeout=REQUEST_TIMEOUT_SECONDS)

    answer = (response.choices[0].message.content or "").strip()
    entity_type = answer if answer in valid_types else ontology.fallback_entity_type()

    cache[relation] = entity_type
    _save_classification_cache(cache)
    return entity_type


DESCRIBE_SYSTEM_PROMPT = """A knowledge graph has a relation (edge) name from a thermoelectric-cooler (TEC) simulation ontology. Write ONE concise, standard scientific Chinese term for the quantity or fact it represents — the proper technical name as used in thermoelectric/materials-science literature, translated naturally into Chinese. NOT the literal untranslated English relation name, and NOT a casual/childish paraphrase — a real scientist reading it should recognize it immediately as the correct term.

Keep every term in the same style: a short noun phrase, 2-6 Chinese characters, no full sentences, no explanatory clauses.

Examples of the target style:
- achieves_delta_T_max_K -> 最大温差
- achieves_optimal_current_A -> 最优电流
- achieves_max_heat_dissipation_W -> 最大散热功率
- achieves_figure_of_merit_1_per_K -> 优值系数
- achieves_total_resistance_ohm -> 总电阻
- achieves_optimal_voltage_V -> 最优电压

Reply with ONLY the term, nothing else — no quotes, no punctuation, no explanation."""


def _load_description_cache() -> dict[str, str]:
    if not DESCRIPTION_CACHE_PATH.exists():
        return {}
    return json.loads(DESCRIPTION_CACHE_PATH.read_text(encoding="utf-8"))


def _save_description_cache(cache: dict[str, str]) -> None:
    DESCRIPTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DESCRIPTION_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def describe_relation(relation: str, ontology: Ontology) -> str:
    """AI-generated standard scientific Chinese term for a relation
    (e.g. "achieves_delta_T_max_K" -> "最大温差"), cached by relation name
    (data/kg/relation_description_cache.json) for the same reason
    classify_relation is: a small, mostly-fixed set of relation names, so
    this is a real LLM call once per never-before-seen relation and a free
    cache hit for every later use of the same one. Powers the graph UI's
    node/edge inspector, which used to show only raw untranslated relation
    names — deliberately a concise standard term, not a casual paraphrase;
    an earlier version tried "能带走多少热量"-style plain-speech and got
    real user feedback that it read as unscientific/inconsistent, not
    approachable."""
    cache = _load_description_cache()
    if relation in cache:
        return cache[relation]

    model = get_llm_model()
    ontology_block = "\n".join(f"- {t.name}: {t.description}" for t in ontology.entity_types)
    messages = [
        {"role": "system", "content": DESCRIBE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Ontology context:\n{ontology_block}\n\nRelation: {relation}"},
    ]
    try:
        response = litellm.completion(
            model=model, messages=messages, temperature=0, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except litellm.BadRequestError as e:
        if "temperature" not in str(e):
            raise
        response = litellm.completion(model=model, messages=messages, timeout=REQUEST_TIMEOUT_SECONDS)

    description = (response.choices[0].message.content or "").strip().strip("\"'。.")
    cache[relation] = description
    _save_description_cache(cache)
    return description
