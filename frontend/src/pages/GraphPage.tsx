import { useEffect, useState } from "react";
import { ApiError, getFullGraph } from "../api";
import { GraphView } from "../components/GraphView";
import type { Subgraph } from "../types";

export function GraphPage() {
  const [graph, setGraph] = useState<Subgraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const data = await getFullGraph();
      setGraph(data);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="panel">
      <div className="row">
        <p className="muted" style={{ flex: 1, margin: 0 }}>
          知识图谱里目前累积的全部三元组，仅支持只读浏览——新的结论需要先经过审批才能写入。
        </p>
        <button onClick={refresh} disabled={loading}>
          {loading ? "加载中…" : "刷新"}
        </button>
      </div>
      {error && <div className="card error-card">{error}</div>}
      {graph && (
        <div className="card">
          <GraphView subgraph={graph} height={520} emptyMessage="知识图谱目前是空的——还没有知识候选通过审批入库。" />
        </div>
      )}
    </div>
  );
}
