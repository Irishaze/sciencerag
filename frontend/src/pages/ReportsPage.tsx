import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { ApiError, getReport, listReports } from "../api";
import type { ReportDetail, ReportListEntry } from "../types";

const SEVERITY_LABEL: Record<string, string> = {
  info: "正常",
  warning: "警告",
  blocking: "阻断",
};

// Anomaly.check is currently always "ood" (energy_balance/pde_residual
// were removed, see sciencerag/validate/checks.py) — but a raw enum value
// like "ood" means nothing to a reader who isn't looking at the source, so
// it's labelled here the same way severity is, rather than printed as-is.
const CHECK_LABEL: Record<string, string> = {
  ood: "异常值检测",
};

const VERDICT_LABEL: Record<string, string> = {
  consistent: "与基准一致",
  deviation_found: "发现偏差",
  insufficient_benchmark: "基准数据不足",
};

// store.py names a report "{run_id}_{generated_at with ':' stripped and
// '+00:00' -> 'Z'}" — best-effort split back into the two parts for the
// list (run_id is the part a reader actually recognizes); falls back to
// showing the raw stem untouched if a run_id ever contains something that
// confuses the pattern, since this is presentation only, never load-bearing.
function splitStem(stem: string): { runId: string; timestamp: string | null } {
  const match = stem.match(/^(.*)_(\d{4}-\d{2}-\d{2}T\d{6}\.\d+.*)$/);
  if (!match) return { runId: stem, timestamp: null };
  return { runId: match[1], timestamp: match[2] };
}

export function ReportsPage() {
  const [entries, setEntries] = useState<ReportListEntry[]>([]);
  const [selectedStem, setSelectedStem] = useState<string | null>(null);
  const [detail, setDetail] = useState<ReportDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingList, setLoadingList] = useState(false);

  async function refresh() {
    setLoadingList(true);
    setError(null);
    try {
      const list = await listReports();
      setEntries(list);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setLoadingList(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function openReport(stem: string) {
    setSelectedStem(stem);
    setDetail(null);
    try {
      const report = await getReport(stem);
      setDetail(report);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }

  return (
    <div>
      <div className="page-intro">
        <p className="muted">每次仿真运行的检查结果和引用来源都存档在这里，可以按运行血缘往回查。</p>
        <button onClick={refresh} disabled={loadingList}>
          {loadingList ? "刷新中…" : "刷新列表"}
        </button>
      </div>
      {error && <div className="card error-card">{error}</div>}
      <div className="reports-layout">
        <ul className="report-list">
          {entries.length === 0 && !loadingList && <li className="muted">暂无报告</li>}
          {entries.map((entry) => {
            const { runId, timestamp } = splitStem(entry.stem);
            return (
              <li
                key={entry.stem}
                className={entry.stem === selectedStem ? "active" : ""}
                onClick={() => openReport(entry.stem)}
              >
                <span className="report-list-run-id">{runId}</span>
                {timestamp && <span className="report-list-time">{timestamp}</span>}
              </li>
            );
          })}
        </ul>
        <div className="report-detail">
          {!detail && <div className="muted">选择左侧一份报告查看。</div>}
          {detail && (
            <>
              <div className="report-meta-row">
                <span className={`badge ${detail.update_proposal_summary.blocked ? "warn" : ""}`}>
                  {detail.update_proposal_summary.blocked ? "已阻断" : "未阻断"}
                </span>
                <span className="badge">
                  {VERDICT_LABEL[detail.literature_comparison.verdict] ?? detail.literature_comparison.verdict}
                </span>
              </div>
              <div className="report-anomalies">
                {detail.anomalies_and_cautions.map((a, i) => (
                  <span key={i} className={`chip ${a.severity}`}>
                    {CHECK_LABEL[a.check] ?? a.check}：{SEVERITY_LABEL[a.severity] ?? a.severity}
                  </span>
                ))}
              </div>
              <div className="markdown-body">
                <ReactMarkdown>{detail.markdown}</ReactMarkdown>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
