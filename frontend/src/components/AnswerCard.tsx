import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { Link } from "react-router-dom";
import { DocumentIcon, GraphIcon, TripleIcon } from "./icons";
import type { AskResponse } from "../types";

// The LLM answer is prompted to write equations as $...$ / $$...$$ (what
// remark-math parses), but models default to LaTeX's own \(...\) / \[...\]
// delimiters often enough that we normalize both into dollar form here
// rather than relying on prompt compliance alone.
function normalizeMathDelimiters(text: string): string {
  return text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, expr: string) => `\n\n$$${expr}$$\n\n`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, expr: string) => `$${expr}$`);
}

export function AnswerCard({ result }: { result: AskResponse }) {
  return (
    <div className="card answer-card">
      <div className="answer-row">
        <div className="answer-text">
          <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
            {normalizeMathDelimiters(result.answer)}
          </ReactMarkdown>
        </div>
        <span className={`answer-source-badge ${result.fallback_used ? "fallback" : ""}`}>
          {result.fallback_used ? <DocumentIcon size={15} /> : <GraphIcon size={15} />}
          {result.fallback_used ? "回退到文献检索" : "图谱命中"}
        </span>
      </div>

      {result.coverage_note && <p className="answer-coverage-note">{result.coverage_note}</p>}

      <div className="answer-sources">
        <div className="answer-sources-label">来源</div>
        <ul className="answer-source-list">
          {result.sources.map((s, i) => (
            <li key={i} className="answer-source-item">
              {s.type === "paper" ? (
                <>
                  <DocumentIcon size={15} />
                  <span>
                    论文 DOI：{s.doi}
                    {s.span ? ` (${s.span})` : ""}
                  </span>
                </>
              ) : (
                <>
                  <TripleIcon size={15} />
                  <span className="mono">{s.triple_id}</span>
                </>
              )}
            </li>
          ))}
        </ul>
      </div>

      {!result.fallback_used && result.subgraph.nodes.length > 0 && (
        <Link to="/graph" className="answer-graph-link">
          <GraphIcon size={16} />
          在知识图谱页面查看完整的图，支持点击浏览
        </Link>
      )}
    </div>
  );
}
