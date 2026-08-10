import { NavLink, Route, Routes } from "react-router-dom";
import "./App.css";
import { AskPage } from "./pages/AskPage";
import { GraphPage } from "./pages/GraphPage";
import { HomePage } from "./pages/HomePage";
import { ReportsPage } from "./pages/ReportsPage";

const NAV_ITEMS = [
  { to: "/", label: "主页", end: true },
  { to: "/ask", label: "问答" },
  { to: "/graph", label: "知识图谱" },
  { to: "/reports", label: "报告" },
];

export default function App() {
  return (
    <div className="page">
      <header className="top">
        <div className="eyebrow">sciencerag</div>
        <h1>ScienceRAG Workbench</h1>
        <p className="lede">
          spec §7 v1 web 前端。文献/知识候选审批面板按 spec §7 规定以命令行脚本形式提供
          （<code>scripts/approve_kg_candidates.py</code>），不在本站内。
        </p>
      </header>

      <nav className="tabs">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => (isActive ? "tab active" : "tab")}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/ask" element={<AskPage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/reports" element={<ReportsPage />} />
      </Routes>
    </div>
  );
}
