import { useEffect, useState } from "react";
import { ApiError, approvePending, getPendingBatch, listPendingCandidates } from "../api";
import type { ApprovalResult, KGCandidate, PendingBatchSummary } from "../types";

const DEDUP_LABEL: Record<string, string> = {
  new: "新",
  duplicate_confirmed: "疑似重复",
  conflict: "与已有数据冲突",
};

const RESULT_LABEL: Record<string, string> = {
  added: "已写入",
  merged: "已合并",
  conflict: "写入但冲突",
  error: "失败",
};

function candidateObject(c: KGCandidate): string {
  if (c.object_value !== null) {
    return `${c.object_value}${c.object_unit ?? ""}`;
  }
  return c.object_entity_label ?? c.object_entity_id ?? "";
}

export function ApprovalPage() {
  const [batches, setBatches] = useState<PendingBatchSummary[]>([]);
  const [selectedStem, setSelectedStem] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<KGCandidate[]>([]);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const [operator, setOperator] = useState("");
  const [reason, setReason] = useState("");
  const [results, setResults] = useState<ApprovalResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  async function refreshList() {
    setLoadingList(true);
    setError(null);
    try {
      setBatches(await listPendingCandidates());
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setLoadingList(false);
    }
  }

  useEffect(() => {
    refreshList();
  }, []);

  async function openBatch(stem: string) {
    setSelectedStem(stem);
    setCandidates([]);
    setSelectedIndices(new Set());
    setResults(null);
    setError(null);
    try {
      const detail = await getPendingBatch(stem);
      setCandidates(detail.candidates);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }

  function toggleIndex(i: number) {
    setSelectedIndices((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  }

  async function submitApproval(approveAll: boolean) {
    if (!selectedStem) return;
    const count = approveAll ? candidates.length : selectedIndices.size;
    if (count === 0) return;
    if (!window.confirm(`确认批准这 ${count} 条候选并写入知识图谱？此操作不可撤销。`)) return;

    setSubmitting(true);
    setError(null);
    try {
      const response = await approvePending(selectedStem, {
        approve_all: approveAll,
        indices: approveAll ? undefined : Array.from(selectedIndices),
        operator: operator || "web",
        reason,
      });
      setResults(response.results);
      // The batch is archived server-side regardless of partial/full
      // approval (matches scripts/approve_kg_candidates.py's own
      // behavior) — it won't show up in the pending list anymore.
      setCandidates([]);
      setSelectedStem(null);
      refreshList();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <div className="page-intro">
        <p className="muted">
          知识候选先经过这里人工确认，才会写入知识图谱——跟命令行工具{" "}
          <code className="approval-cli-ref">scripts/approve_kg_candidates.py</code> 是同一套逻辑，选一个用就够了。
        </p>
        <button onClick={refreshList} disabled={loadingList}>
          {loadingList ? "刷新中…" : "刷新列表"}
        </button>
      </div>

      {error && <div className="card error-card">{error}</div>}

      {results && (
        <div className="approval-results">
          <h2>批准结果</h2>
          <ul className="approval-result-list">
            {results.map((r) => (
              <li key={r.index} className="approval-result-row">
                <span className={`approval-result-dot ${r.status}`} />
                [{r.index}] {RESULT_LABEL[r.status] ?? r.status}
                {r.triple_id && <span className="approval-run-id">triple_id={r.triple_id}</span>}
                {r.error && <span className="muted">{r.error}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="approval-layout">
        <div className="approval-queue">
          <p className="approval-queue-label">待审批批次</p>
          <ul className="approval-queue-list">
            {batches.length === 0 && !loadingList && <li className="muted">当前没有待审批的批次</li>}
            {batches.map((b) => (
              <li
                key={b.stem}
                className={`approval-queue-item ${b.stem === selectedStem ? "active" : ""}`}
                onClick={() => openBatch(b.stem)}
              >
                <span className="approval-queue-stem">{b.stem}</span>
                <span className="approval-queue-count">{b.count}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="approval-review">
          {!selectedStem && <div className="approval-empty">选择左侧一个批次，开始审批候选内容。</div>}
          {selectedStem && candidates.length > 0 && (
            <>
              <div className="approval-review-header">
                <h2>{selectedStem}</h2>
                <span className="approval-review-count">
                  已选 {selectedIndices.size} / {candidates.length} 条
                </span>
              </div>

              <ul className="approval-candidate-list">
                {candidates.map((c, i) => {
                  const selected = selectedIndices.has(i);
                  return (
                    <li
                      key={i}
                      className={`approval-candidate ${selected ? "selected" : ""}`}
                      onClick={() => toggleIndex(i)}
                    >
                      <input
                        type="checkbox"
                        className="approval-candidate-checkbox"
                        checked={selected}
                        onChange={() => toggleIndex(i)}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <div className="approval-candidate-body">
                        <div className="approval-candidate-top">
                          <span className="approval-candidate-claim">
                            {c.relation_description || c.relation}
                          </span>
                          <span className={`approval-badge ${c.dedup_status}`}>
                            {DEDUP_LABEL[c.dedup_status] ?? c.dedup_status}
                          </span>
                        </div>
                        <p className="approval-candidate-sentence">
                          {c.subject} → {candidateObject(c)}
                        </p>
                        <div className="approval-candidate-foot">
                          <div className="approval-confidence">
                            <span className="approval-confidence-track">
                              <span
                                className="approval-confidence-fill"
                                style={{ width: `${Math.round(c.confidence * 100)}%` }}
                              />
                            </span>
                            <span className="approval-confidence-value">{c.confidence.toFixed(2)}</span>
                          </div>
                          <span className="approval-run-id">{c.run_id}</span>
                        </div>
                      </div>
                    </li>
                  );
                })}
              </ul>

              <div className="approval-actions">
                <div className="approval-actions-row">
                  <input
                    placeholder="审批人（可选）"
                    value={operator}
                    onChange={(e) => setOperator(e.target.value)}
                  />
                  <input
                    placeholder="备注（可选）"
                    value={reason}
                    onChange={(e) => setReason(e.target.value)}
                  />
                </div>
                <div className="approval-actions-buttons">
                  <button
                    onClick={() => submitApproval(false)}
                    disabled={submitting || selectedIndices.size === 0}
                  >
                    批准选中的 {selectedIndices.size} 条
                  </button>
                  <button className="secondary" onClick={() => submitApproval(true)} disabled={submitting}>
                    批准全部 {candidates.length} 条
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
