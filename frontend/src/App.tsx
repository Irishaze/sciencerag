import { useState } from "react";
import "./App.css";
import { AskPanel } from "./components/AskPanel";
import { ReportsPanel } from "./components/ReportsPanel";

type Tab = "ask" | "reports";

export default function App() {
  const [tab, setTab] = useState<Tab>("ask");

  return (
    <div className="page">
      <header className="top">
        <div className="eyebrow">sciencerag</div>
        <h1>ScienceRAG Workbench</h1>
        <p className="lede">
          spec §7 v1 web 前端 — 问答 + 图谱可视化 + 报告浏览。文献/知识候选审批面板按 spec §7
          规定以命令行脚本形式提供（<code>scripts/approve_kg_candidates.py</code>），不在本页面内。
        </p>
      </header>

      <nav className="tabs">
        <button className={tab === "ask" ? "tab active" : "tab"} onClick={() => setTab("ask")}>
          问答
        </button>
        <button className={tab === "reports" ? "tab active" : "tab"} onClick={() => setTab("reports")}>
          报告浏览
        </button>
      </nav>

      {tab === "ask" && <AskPanel />}
      {tab === "reports" && <ReportsPanel />}
    </div>
  );
}
