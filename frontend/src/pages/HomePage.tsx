import { AnswerCard } from "../components/AnswerCard";
import { TecDiagram } from "../components/TecDiagram";
import { useAsk } from "../hooks/useAsk";
import { Link } from "react-router-dom";

const STATS = [
  { label: "可溯源", body: "问答和报告里的每一个结论，都能追溯到原始出处" },
  { label: "物理校验优先", body: "仿真结果要先通过守恒与残差检查，才能进入知识图谱" },
];

const WORKFLOW = [
  { n: "01", title: "先验检索", body: "从知识图谱和内部文献中查找依据，图谱优先，覆盖不足时再检索文献" },
  { n: "02", title: "仿真验证", body: "做守恒检查、PDE 残差验算，并与已知案例、文献结果对比——不可信的结果到此为止" },
  { n: "03", title: "知识积累", body: "只有通过检查的结果才会成为候选知识，经人工审批后写入图谱" },
  { n: "04", title: "问答与报告", body: "回答有理有据，报告可追溯完整的运行血缘" },
];

const FEATURES = [
  {
    to: "/ask",
    tag: "问答",
    title: "向系统提问",
    body: "优先查知识图谱，没有匹配结果时会转向文献检索——这次答案用的是哪种来源，回答里会如实标注。",
    metric: "图谱命中、文献回退，两种来源都标注清楚",
  },
  {
    to: "/graph",
    tag: "知识图谱",
    title: "浏览积累的知识",
    body: "点开任意一条结论，能看到它来自哪次仿真、置信度多少、跟哪些结论相关。图谱只能通过审批流程写入，这里始终是只读浏览。",
    metric: "点节点看全部相关结论，点连线看单条详情",
  },
  {
    to: "/reports",
    tag: "报告",
    title: "按运行血缘查报告",
    body: "每次仿真运行的异常检查结果、跟基准和文献的对比结论都会留痕，每个数字都有引用支撑。",
    metric: "定量论断皆可溯源",
  },
];

export function HomePage() {
  const { question, setQuestion, maxHits, setMaxHits, loading, error, result, submit } = useAsk();

  return (
    <div>
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-copy">
            <p className="hero-eyebrow">热电制冷器 · Science RAG</p>
            <h1 className="hero-title">让每一条关于TEC的结论都能查到对应出处</h1>
            <p className="hero-sub">仿真结果需要经过物理一致性检查，才有资格成为知识图谱里的一条结论。</p>

            <div className="stat-row">
              {STATS.map((s) => (
                <div key={s.label} className="stat">
                  <strong>{s.label}</strong>
                  <span>{s.body}</span>
                </div>
              ))}
            </div>

            <div className="workflow">
              <div className="workflow-label">工作流程</div>
              <ol className="workflow-list">
                {WORKFLOW.map((w) => (
                  <li key={w.n}>
                    <span className="workflow-num">{w.n}</span>
                    <div>
                      <strong>{w.title}</strong>
                      <p>{w.body}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          </div>

          <div className="hero-question-panel">
            <div className="hero-question-label">
              <span>02 / 研究问题</span>
              <span className="muted">接口 · sciencerag.ask</span>
            </div>
            <textarea
              className="hero-question-input"
              placeholder="输入你想问的问题，例如：Bi2Te3 单级 TEC 的 delta_T_max_K 大概是多少？"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
              }}
            />
            <div className="hero-question-meta">
              <label htmlFor="max-hits">知识图谱检索条数上限（{maxHits}）</label>
              <input
                id="max-hits"
                type="range"
                min={1}
                max={20}
                value={maxHits}
                onChange={(e) => setMaxHits(Number(e.target.value))}
              />
            </div>
            <button className="hero-question-submit" onClick={() => submit()} disabled={loading}>
              {loading ? "查询中…" : "提问"} {!loading && "→"}
            </button>
          </div>
        </div>
      </section>

      <section className="page results-section">
        {error && <div className="card error-card">请求失败：{error}</div>}

        {result ? (
          <AnswerCard result={result} />
        ) : (
          <div className="results-placeholder">
            <div className="results-placeholder-diagram">
              <TecDiagram />
            </div>
            <p className="muted">{loading ? "查询中…" : "问一个问题，答案和依据都会显示在这里。"}</p>
          </div>
        )}
      </section>

      <section className="page feature-section">
        <div className="card-grid">
          {FEATURES.map((f) => (
            <Link key={f.to} to={f.to} className="feature-card">
              <span className="feature-tag">{f.tag}</span>
              <h2>{f.title}</h2>
              <p>{f.body}</p>
              <span className="feature-metric">{f.metric}</span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
