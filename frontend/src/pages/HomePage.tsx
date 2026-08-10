import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { TecDiagram } from "../components/TecDiagram";

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
  const [question, setQuestion] = useState("");
  const navigate = useNavigate();

  function handleSubmit() {
    const trimmed = question.trim();
    if (!trimmed) return;
    navigate("/ask", { state: { initialQuestion: trimmed } });
  }

  return (
    <div>
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-copy">
            <p className="hero-eyebrow">热电制冷器 · 科学 RAG</p>
            <h1 className="hero-title">让每一条关于 TEC 的结论都能查到它的来源</h1>
            <p className="hero-sub">
              仿真结果先过一遍物理一致性检查——冷热两侧的能量守恒、偏微分方程残差都对得上，
              才有资格成为知识图谱里的一条结论。问答、报告里的每个数字，都能一路点回它的出处。
            </p>
          </div>
          <div className="hero-diagram-wrap">
            <TecDiagram />
          </div>
        </div>
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

      <section className="ask-cta">
        <div className="ask-cta-inner">
          <div className="ask-cta-copy">
            <h2>有问题，直接问</h2>
            <p className="muted">回车换行，⌘/Ctrl + 回车 或点击按钮提交，会跳到问答页面查看回答。</p>
          </div>
          <div className="ask-cta-form">
            <textarea
              placeholder="例如：Bi2Te3 单级 TEC 的 delta_T_max_K 是多少？"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit();
              }}
            />
            <button onClick={handleSubmit}>提问 →</button>
          </div>
        </div>
      </section>
    </div>
  );
}
