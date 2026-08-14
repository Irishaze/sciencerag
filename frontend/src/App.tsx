import { NavLink, Route, Routes } from "react-router-dom";
import "./App.css";
import { ApprovalPage } from "./pages/ApprovalPage";
import { GraphPage } from "./pages/GraphPage";
import { HomePage } from "./pages/HomePage";
import { ReportsPage } from "./pages/ReportsPage";
import { useAsk } from "./hooks/useAsk";

const NAV_ITEMS = [
  { to: "/", label: "主页", end: true },
  { to: "/graph", label: "知识图谱" },
  { to: "/reports", label: "报告" },
  { to: "/approval", label: "知识候选审批" },
];

export default function App() {
  // Lifted here (not inside HomePage) so the last question/answer survives
  // navigating away and back — HomePage unmounts on route change like any
  // other route, which would otherwise reset useAsk's local state on every
  // return trip to "/".
  const ask = useAsk();

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
        <Route path="/" element={<HomePage ask={ask} />} />
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
        <Route
          path="/approval"
          element={
            <div className="page">
              <ApprovalPage />
            </div>
          }
        />
      </Routes>
    </div>
  );
}
