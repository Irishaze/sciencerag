import { AnswerCard } from "../components/AnswerCard";
import { TecDiagram } from "../components/TecDiagram";
import { useAsk } from "../hooks/useAsk";
import { Link } from "react-router-dom";

const STATS = [
  { label: "可溯源", body: "每条回答、每份报告的每个数字都能点回它的来源" },
  { label: "物理校验优先", body: "仿真结果先过守恒 / 残差检查，通过了才进知识图谱" },
];

const WORKFLOW = [
  { n: "01", title: "先验检索", body: "从内部文献与知识图谱找依据，图谱优先，覆盖不足再查文献" },
  { n: "02", title: "仿真验证", body: "守恒检查、PDE 残差、跟已知案例/文献对比，不可信的结果到此为止" },
  { n: "03", title: "知识积累", body: "通过检查的结果才会成为候选知识，人工审批后写入图谱" },
  { n: "04", title: "问答与报告", body: "有据可依的回答、可追溯运行血缘的报告" },
];

const FEATURES = [
  {
    to: "/ask",
    tag: "问答",
    title: "向系统提问",
    body: "先查知识图谱有没有匹配的答案；没有的话诚实回退到文献检索，不会不声不响地换一条路给你一个看似确定的答案。",
    metric: "图谱命中 / 文献回退 两种来源都标注清楚",
  },
  {
    to: "/graph",
    tag: "知识图谱",
    title: "浏览积累的知识",
    body: "每一条结论都能点开看：是从哪次仿真来的、置信度多少、跟哪些结论相关。图谱只能通过审批流程写入，浏览始终是只读的。",
    metric: "点节点看全部相关结论，点连线看单条详情",
  },
  {
    to: "/reports",
    tag: "报告",
    title: "按运行血缘查报告",
    body: "每一次仿真运行的异常检查结果、跟基准/文献的对比结论，全部留痕，每条数字都有引用支撑。",
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
            <p className="hero-eyebrow">热电制冷器 · 科学 RAG</p>
            <h1 className="hero-title">让每一条关于 TEC 的结论都能查到它的来源</h1>
            <p className="hero-sub">
              仿真结果先过一遍物理一致性检查——冷热两侧的能量守恒、偏微分方程残差都对得上，
              才有资格成为知识图谱里的一条结论。
            </p>

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
              <span className="muted">引擎：sciencerag.ask</span>
            </div>
            <textarea
              className="hero-question-input"
              placeholder="输入你想问的问题，例如：Bi2Te3 单级 TEC 的 delta_T_max_K 是多少？规律是什么？"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
              }}
            />
            <div className="hero-question-meta">
              <label htmlFor="max-hits">图谱检索条数上限（{maxHits}）</label>
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
            <p className="muted">{loading ? "查询中…" : "问一个问题，回答和依据会显示在这里。"}</p>
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
