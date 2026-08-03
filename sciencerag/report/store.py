"""Versioned on-disk storage for generated reports (spec §5: "报告随运行血缘
版本化存储"). Filenames key on run_id + generated_at so re-running a report
for the same run_id keeps prior versions instead of overwriting them —
M5's report-browsing panel reads this directory back."""

from pathlib import Path

from sciencerag.report.models import ReportResponse

REPORTS_DIR = Path("data/reports")


def _safe_timestamp(generated_at: str) -> str:
    return generated_at.replace(":", "").replace("+00:00", "Z")


def store_report(response: ReportResponse) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{response.run_id}_{_safe_timestamp(response.generated_at)}"
    json_path = REPORTS_DIR / f"{stem}.json"
    md_path = REPORTS_DIR / f"{stem}.md"
    json_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(response.markdown, encoding="utf-8")
    return json_path


def list_reports() -> list[dict]:
    """Newest first. Returns lightweight index entries, not full bodies —
    M5's report-browsing panel lists these, then fetches one by filename."""
    if not REPORTS_DIR.exists():
        return []
    entries = []
    for path in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
        entries.append({"filename": path.name, "stem": path.stem})
    return entries


def load_report(stem: str) -> ReportResponse:
    json_path = REPORTS_DIR / f"{stem}.json"
    return ReportResponse.model_validate_json(json_path.read_text(encoding="utf-8"))
