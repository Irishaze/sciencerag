"""Batch evidence for candidate ranking (spec §3.4) — M6.

"若团队采用类 AI co-scientist 的「生成—辩论—排名」竞赛机制,竞赛逻辑归属于
Hermes...本服务通过批量证据请求为其提供支撑:给定 N 个候选实验规范,逐一返回
支持性与反驳性证据,使排序器(Elo/BTL)能够基于文献依据而非模型直觉来打分。"

Ranking itself is explicitly out of scope here (it's Hermes's job per
spec) — this module only classifies each retrieved evidence item's stance
toward one candidate description: supports / refutes / neutral. One LLM
call per candidate (not per evidence item) to bound cost; on a malformed
LLM response, degrades to "neutral" for every item in that candidate
rather than failing the whole batch (same "never half-broken" principle as
extract.py's own error handling).
"""

from __future__ import annotations

import json
import logging
from typing import Literal

import litellm
from pydantic import BaseModel, Field

from sciencerag.common.config import get_llm_model
from sciencerag.priors.extract import EvidenceItem, _strip_code_fences
from sciencerag.priors.retrieval import _build_evidence_table, run_query

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 90
MAX_RETRIES = 2

SYSTEM_PROMPT = (
    "You classify literature evidence against a candidate experimental "
    "configuration for a thermoelectric cooler (TEC). For EACH evidence "
    "item, decide whether it supports, refutes, or is neutral toward the "
    "candidate. Respond with ONLY a JSON array, one object per evidence "
    'item, each shaped {"label": "<the [E#] label>", "stance": '
    '"supports" | "refutes" | "neutral"}. Do not omit any label. Do not '
    "invent labels that weren't given."
)


class CandidateSpec(BaseModel):
    candidate_id: str
    description: str
    design_parameters: dict[str, float] = Field(default_factory=dict)


class EvidenceStance(BaseModel):
    doi: str | None
    span: str
    text: str
    relevance: float
    stance: Literal["supports", "refutes", "neutral"]


class CandidateEvidence(BaseModel):
    candidate_id: str
    supporting: list[EvidenceStance] = Field(default_factory=list)
    refuting: list[EvidenceStance] = Field(default_factory=list)
    neutral: list[EvidenceStance] = Field(default_factory=list)
    coverage: Literal["ok", "insufficient"]


def _classify_stances(
    candidate: CandidateSpec, evidence_table: dict[str, EvidenceItem]
) -> dict[str, Literal["supports", "refutes", "neutral"]]:
    evidence_block = "\n".join(
        f'[{label}] (source: {item.doi or "unknown"}) "{item.text}"'
        for label, item in evidence_table.items()
    )
    user_prompt = (
        f"Candidate configuration: {candidate.description}\n"
        f"Design parameters: {candidate.design_parameters}\n\n"
        f"Evidence:\n{evidence_block}\n\nClassification:"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    model = get_llm_model()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            try:
                response = litellm.completion(
                    model=model, messages=messages, temperature=0, timeout=REQUEST_TIMEOUT_SECONDS
                )
            except litellm.BadRequestError as e:
                if "temperature" not in str(e):
                    raise
                response = litellm.completion(model=model, messages=messages, timeout=REQUEST_TIMEOUT_SECONDS)
            raw = _strip_code_fences(response.choices[0].message.content)
            parsed = json.loads(raw)
            stances = {}
            for item in parsed:
                if item.get("label") in evidence_table and item.get("stance") in (
                    "supports",
                    "refutes",
                    "neutral",
                ):
                    stances[item["label"]] = item["stance"]
            if stances:
                return stances
        except Exception as e:  # noqa: BLE001 - degrade below, don't fail the batch
            logger.warning(
                "batch_evidence stance classification failed (attempt %d/%d) for %s: %s",
                attempt,
                MAX_RETRIES,
                candidate.candidate_id,
                e,
            )

    return {}  # every label defaults to "neutral" below


def get_candidate_evidence(candidate: CandidateSpec) -> CandidateEvidence:
    response = run_query(candidate.description)
    contexts = response.session.contexts
    if not contexts:
        return CandidateEvidence(candidate_id=candidate.candidate_id, coverage="insufficient")

    evidence_table, _below_threshold = _build_evidence_table(contexts)
    if not evidence_table:
        return CandidateEvidence(candidate_id=candidate.candidate_id, coverage="insufficient")

    stances = _classify_stances(candidate, evidence_table)

    result = CandidateEvidence(candidate_id=candidate.candidate_id, coverage="ok")
    for label, item in evidence_table.items():
        stance = stances.get(label, "neutral")
        entry = EvidenceStance(
            doi=item.doi, span=item.span, text=item.text, relevance=item.relevance, stance=stance
        )
        if stance == "supports":
            result.supporting.append(entry)
        elif stance == "refutes":
            result.refuting.append(entry)
        else:
            result.neutral.append(entry)
    return result


def get_batch_evidence(candidates: list[CandidateSpec]) -> list[CandidateEvidence]:
    return [get_candidate_evidence(candidate) for candidate in candidates]


class BatchEvidenceRequest(BaseModel):
    candidates: list[CandidateSpec] = Field(min_length=1)


class BatchEvidenceResponse(BaseModel):
    status: Literal["ok"] = "ok"
    results: list[CandidateEvidence]
    trace_id: str
