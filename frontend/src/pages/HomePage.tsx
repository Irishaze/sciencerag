import { Link } from "react-router-dom";

const SECTIONS = [
  {
    to: "/ask",
    title: "问答",
    body: "向系统提问，先查知识图谱，没有匹配的三元组时诚实回退到文献检索，两种情况响应里都会明确标注。",
  },
  {
    to: "/graph",
    title: "知识图谱",
    body: "浏览目前累积的全部三元组，点击节点或连线查看具体来源、置信度。图谱只能通过审批流程写入，这里始终是只读的。",
  },
  {
    to: "/reports",
    title: "报告",
    body: "按运行血缘浏览历史报告——异常检查结果、跟基准/文献的对比结论、每条定量论断的引用。",
  },
];

export function HomePage() {
  return (
    <div className="panel">
      <div className="card-grid">
        {SECTIONS.map((s) => (
          <Link key={s.to} to={s.to} className="home-card">
            <h2>{s.title}</h2>
            <p>{s.body}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
