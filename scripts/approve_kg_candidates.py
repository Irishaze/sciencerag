"""Command-line KG candidate approval (spec §7: "v1 可用命令行脚本替代页面";
§6.3: the only path into the graph is candidate -> approval -> registration).

Input is a JSON file holding a list of KGCandidate-shaped objects — e.g.
`update_package.kg_candidates` saved from a real /sciencerag/validate
response. Nothing is written to the graph unless --approve-all or
--approve is passed; with neither, this only previews what's in the file.

    uv run python scripts/approve_kg_candidates.py --file candidates.json --list
    uv run python scripts/approve_kg_candidates.py --file candidates.json --approve 0,2
    uv run python scripts/approve_kg_candidates.py --file candidates.json --approve-all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sciencerag.common.audit import log_audit_entry
from sciencerag.common.trace import new_trace_id
from sciencerag.priors.kg import KGSource, add_triple
from sciencerag.validate.models import KGCandidate


def load_candidates(path: Path) -> list[KGCandidate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [KGCandidate.model_validate(item) for item in data]


def _print_candidate(index: int, candidate: KGCandidate) -> None:
    print(
        f"[{index}] {candidate.subject} {candidate.relation} = "
        f"{candidate.object_value}{candidate.object_unit or ''} "
        f"(confidence={candidate.confidence}, run_id={candidate.run_id}, "
        f"dedup_status={candidate.dedup_status})"
    )


def approve(candidate: KGCandidate, operator: str, reason: str) -> None:
    triple, status = add_triple(
        subject=candidate.subject,
        relation=candidate.relation,
        object_value=candidate.object_value,
        object_unit=candidate.object_unit,
        conditions=candidate.conditions,
        confidence=candidate.confidence,
        run_id=candidate.run_id,
        sources=[KGSource(type="run", run_id=candidate.run_id)],
    )
    log_audit_entry(
        trace_id=new_trace_id("kgappr"),
        endpoint="sciencerag.kg_approval",
        request={"candidate": candidate.model_dump(), "operator": operator, "reason": reason},
        evidence=[],
        output={"triple": triple.model_dump(), "status": status},
    )
    print(f"  -> {status}: triple_id={triple.triple_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="preview only, default if no approval flag given")
    parser.add_argument("--approve", type=str, default=None, help="comma-separated 0-based indices")
    parser.add_argument("--approve-all", action="store_true")
    parser.add_argument("--operator", type=str, default="cli", help="recorded in the audit log")
    parser.add_argument("--reason", type=str, default="", help="recorded in the audit log")
    args = parser.parse_args()

    candidates = load_candidates(args.file)
    if not candidates:
        print("no candidates in file")
        return

    for index, candidate in enumerate(candidates):
        _print_candidate(index, candidate)

    if args.approve_all:
        indices = list(range(len(candidates)))
    elif args.approve:
        indices = [int(item) for item in args.approve.split(",") if item.strip()]
    else:
        return  # preview-only

    for index in indices:
        if not 0 <= index < len(candidates):
            print(f"skipping out-of-range index {index}", file=sys.stderr)
            continue
        approve(candidates[index], args.operator, args.reason)


if __name__ == "__main__":
    main()
