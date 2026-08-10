import { Link } from "react-router-dom";
import type { AskResponse } from "../types";

export function AnswerCard({ result }: { result: AskResponse }) {
  return (
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
              论文 DOI：{s.doi}
              {s.span ? ` (${s.span})` : ""}
            </li>
          ) : (
            <li key={i}>知识图谱三元组：{s.triple_id}</li>
          )
        )}
      </ul>
      {!result.fallback_used && result.subgraph.nodes.length > 0 && (
        <p className="muted" style={{ marginTop: 10 }}>
          这次命中的三元组也能在<Link to="/graph">知识图谱</Link>页面看到完整的图，并支持点击浏览。
        </p>
      )}
    </div>
  );
}
