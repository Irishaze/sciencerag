"""Command-line approval for external papers (spec §3.5: "外部文献升级为
"可信"有两条路:人工确认,或满足预设规则自动转正...规则由团队制定,人工只处理
规则覆盖不到的情形").

Reads data/external_papers/pending.json (populated by
sciencerag/priors/external_retrieval.py's record_pending_papers, every time
a real /sciencerag/priors call with allow_external=true augments a
thin-coverage query). Approving a paper here only promotes its metadata to
data/external_papers/approved.json — it does NOT run it through the real
PDF ingestion pipeline (spec §3.5 steps 2-4, corpus/papers/), since the
abstract-only text this queue holds was a deliberate v1 scope decision (see
external_retrieval.py's module docstring). Treat "approved" as "confirmed
trustworthy source, still pending real full-text ingestion", not "now
indexed for retrieval".

    uv run python scripts/approve_external_papers.py --list
    uv run python scripts/approve_external_papers.py --approve 10.1234/x,10.1234/y
    uv run python scripts/approve_external_papers.py --auto-promote-hit-threshold 3
"""

from __future__ import annotations

import argparse
import json

from sciencerag.common.audit import log_audit_entry
from sciencerag.common.trace import new_trace_id
from sciencerag.priors.external_retrieval import (
    APPROVED_PATH,
    PendingExternalPaper,
    _load_pending,
    _save_pending,
)


def _load_approved() -> dict[str, PendingExternalPaper]:
    if not APPROVED_PATH.exists():
        return {}
    data = json.loads(APPROVED_PATH.read_text(encoding="utf-8"))
    return {item["doi"]: PendingExternalPaper.model_validate(item) for item in data}


def _save_approved(approved: dict[str, PendingExternalPaper]) -> None:
    APPROVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    APPROVED_PATH.write_text(
        json.dumps([p.model_dump() for p in approved.values()], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _print_pending(pending: dict[str, PendingExternalPaper]) -> None:
    if not pending:
        print("no pending external papers")
        return
    for doi, paper in sorted(pending.items(), key=lambda kv: -kv[1].hit_count):
        print(f"[{paper.hit_count}x] {doi} — {paper.title} ({paper.year})")


def approve(dois: list[str], operator: str, reason: str) -> None:
    pending = _load_pending()
    approved = _load_approved()
    for doi in dois:
        paper = pending.pop(doi, None)
        if paper is None:
            print(f"skipping {doi!r}: not in pending queue")
            continue
        approved[doi] = paper
        log_audit_entry(
            trace_id=new_trace_id("extappr"),
            endpoint="sciencerag.external_paper_approval",
            request={"doi": doi, "operator": operator, "reason": reason},
            evidence=[],
            output={"paper": paper.model_dump(), "status": "approved"},
        )
        print(f"approved {doi} ({paper.title})")
    _save_pending(pending)
    _save_approved(approved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--approve", type=str, default=None, help="comma-separated DOIs")
    parser.add_argument(
        "--auto-promote-hit-threshold",
        type=int,
        default=None,
        help="approve every pending paper with hit_count >= N (spec §3.5's repeat-hit rule)",
    )
    parser.add_argument("--operator", type=str, default="cli")
    parser.add_argument("--reason", type=str, default="")
    args = parser.parse_args()

    pending = _load_pending()
    _print_pending(pending)

    dois: list[str] = []
    if args.approve:
        dois.extend(item.strip() for item in args.approve.split(",") if item.strip())
    if args.auto_promote_hit_threshold is not None:
        auto_dois = [doi for doi, paper in pending.items() if paper.hit_count >= args.auto_promote_hit_threshold]
        if auto_dois:
            print(f"auto-promoting {len(auto_dois)} paper(s) with hit_count >= {args.auto_promote_hit_threshold}")
        dois.extend(auto_dois)

    if dois:
        approve(dois, args.operator, args.reason)


if __name__ == "__main__":
    main()
