import { useState } from "react";
import { ApiError, ask } from "../api";
import type { AskResponse } from "../types";
import { GraphView } from "./GraphView";

export function AskPanel() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);

  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await ask(question.trim());
      setResult(response);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.category}: ${e.message}` : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <div className="row">
        <textarea
          placeholder="向知识图谱提问，例如：Bi2Te3 单级 TEC 的 delta_T_max_K 是多少？"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleAsk();
          }}
        />
        <button onClick={handleAsk} disabled={loading}>
          {loading ? "查询中…" : "提问"}
        </button>
      </div>

      {error && <div className="card error-card">请求失败：{error}</div>}

      {result && (
        <>
          <div className="card">
            <div className="answer-row">
              <p className="answer-text">{result.answer}</p>
              <span className={`badge ${result.fallback_used ? "warn" : ""}`}>
                {result.fallback_used ? "回退到文献检索" : "图谱命中"}
              </span>
            </div>
            {result.coverage_note && <p className="muted">{result.coverage_note}</p>}
            <div className="muted" style={{ marginTop: 10 }}>
              来源：
            </div>
            <ul className="source-list">
              {result.sources.map((s, i) =>
                s.type === "paper" ? (
                  <li key={i}>
                    论文 DOI: {s.doi}
                    {s.span ? ` (${s.span})` : ""}
                  </li>
                ) : (
                  <li key={i}>知识图谱三元组: {s.triple_id}</li>
                )
              )}
            </ul>
          </div>
          <div className="card">
            <GraphView subgraph={result.subgraph} />
          </div>
        </>
      )}
    </div>
  );
}
