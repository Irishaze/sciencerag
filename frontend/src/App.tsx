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
    <div className="app-shell">
      <nav className="site-nav">
        <div className="site-nav-inner">
          <NavLink to="/" className="wordmark">
            <span className="wordmark-dot" />
            ScienceRAG
          </NavLink>
          <div className="tabs">
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
          </div>
        </div>
      </nav>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route
          path="/ask"
          element={
            <div className="page">
              <AskPage />
            </div>
          }
        />
        <Route
          path="/graph"
          element={
            <div className="page">
              <GraphPage />
            </div>
          }
        />
        <Route
          path="/reports"
          element={
            <div className="page">
              <ReportsPage />
            </div>
          }
        />
      </Routes>
    </div>
  );
}
