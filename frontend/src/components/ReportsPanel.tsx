import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { ApiError, getReport, listReports } from "../api";
import type { ReportDetail, ReportListEntry } from "../types";

const SEVERITY_LABEL: Record<string, string> = {
  info: "正常",
  warning: "警告",
  blocking: "阻断",
};

export function ReportsPanel() {
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
    <div className="panel">
      <div className="row">
        <button onClick={refresh} disabled={loadingList}>
          {loadingList ? "刷新中…" : "刷新列表"}
        </button>
      </div>
      {error && <div className="card error-card">{error}</div>}
      <div className="reports-layout">
        <ul className="report-list">
          {entries.length === 0 && !loadingList && <li className="muted">暂无报告</li>}
          {entries.map((entry) => (
            <li
              key={entry.stem}
              className={entry.stem === selectedStem ? "active" : ""}
              onClick={() => openReport(entry.stem)}
            >
              {entry.stem}
            </li>
          ))}
        </ul>
        <div className="report-detail">
          {!detail && <div className="muted">选择左侧一份报告查看。</div>}
          {detail && (
            <>
              <div className="report-meta-row">
                <span className={`badge ${detail.update_proposal_summary.blocked ? "warn" : ""}`}>
                  {detail.update_proposal_summary.blocked ? "已阻断" : "未阻断"}
                </span>
                <span className="badge">{detail.literature_comparison.verdict}</span>
              </div>
              <div className="report-anomalies">
                {detail.anomalies_and_cautions.map((a, i) => (
                  <span key={i} className={`chip ${a.severity}`}>
                    {a.check}: {SEVERITY_LABEL[a.severity] ?? a.severity}
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
