"""Command-line KG candidate approval (spec §7: "v1 可用命令行脚本替代页面";
§6.3: the only path into the graph is candidate -> approval -> registration).

Every /sciencerag/validate call that produces candidates queues them to
data/kg_candidates/pending/ automatically (see
sciencerag/validate/kg_candidate_store.py) — pick one from there instead of
hand-assembling a file:

    uv run python scripts/approve_kg_candidates.py --list-pending
    uv run python scripts/approve_kg_candidates.py --pending run_123_2026... --list
    uv run python scripts/approve_kg_candidates.py --pending run_123_2026... --approve 0,2
    uv run python scripts/approve_kg_candidates.py --pending run_123_2026... --approve-all

A manually-assembled file still works the same as before via --file:

    uv run python scripts/approve_kg_candidates.py --file candidates.json --approve-all

Nothing is written to the graph unless --approve-all or --approve is
passed; with neither, this only previews. Approving from a --pending file
archives it (data/kg_candidates/archive/) afterward so it stops showing up
in --list-pending; --file input is left untouched either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sciencerag.common.audit import log_audit_entry
from sciencerag.common.trace import new_trace_id
from sciencerag.priors.kg import KGSource, add_triple
from sciencerag.validate import kg_candidate_store
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-pending", action="store_true", help="list queued candidate batches and exit")
    parser.add_argument("--pending", type=str, default=None, help="stem of a queued batch (see --list-pending)")
    parser.add_argument("--file", type=Path, default=None, help="manually-assembled candidates JSON file")
    parser.add_argument("--list", action="store_true", help="preview only, default if no approval flag given")
    parser.add_argument("--approve", type=str, default=None, help="comma-separated 0-based indices")
    parser.add_argument("--approve-all", action="store_true")
    parser.add_argument("--operator", type=str, default="cli", help="recorded in the audit log")
    parser.add_argument("--reason", type=str, default="", help="recorded in the audit log")
    args = parser.parse_args()

    if args.list_pending:
        pending = kg_candidate_store.list_pending()
        if not pending:
            print("no pending candidate batches")
            return
        for entry in pending:
            print(f"{entry['stem']}  ({entry['count']} candidate(s))")
        return

    if args.pending and args.file:
        print("pass only one of --pending or --file", file=sys.stderr)
        sys.exit(1)
    if not args.pending and not args.file:
        print("pass --pending <stem> (see --list-pending) or --file <path>", file=sys.stderr)
        sys.exit(1)

    try:
        candidates = (
            kg_candidate_store.load_pending(args.pending) if args.pending else load_candidates(args.file)
        )
    except FileNotFoundError:
        source = f"pending batch {args.pending!r}" if args.pending else f"file {args.file}"
        print(f"no such {source} — check --list-pending or the --file path", file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"couldn't read candidates: {e}", file=sys.stderr)
        sys.exit(1)

    if not candidates:
        print("no candidates in batch")
        return

    for index, candidate in enumerate(candidates):
        _print_candidate(index, candidate)

    if args.approve_all:
        indices = list(range(len(candidates)))
    elif args.approve:
        try:
            indices = [int(item) for item in args.approve.split(",") if item.strip()]
        except ValueError:
            print(f"--approve must be comma-separated integers, got {args.approve!r}", file=sys.stderr)
            sys.exit(1)
    else:
        return  # preview-only

    for index in indices:
        if not 0 <= index < len(candidates):
            print(f"skipping out-of-range index {index}", file=sys.stderr)
            continue
        try:
            approve(candidates[index], args.operator, args.reason)
        except ValueError as e:
            # A single bad candidate (e.g. a non-finite object_value —
            # add_triple's own last-gate check) shouldn't abort the rest
            # of an otherwise-good batch.
            print(f"  -> REJECTED [{index}]: {e}", file=sys.stderr)

    if args.pending:
        kg_candidate_store.archive_pending(args.pending)


if __name__ == "__main__":
    main()
