import { useMemo, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { Subgraph, SubgraphEdge } from "../types";

type Props = { subgraph: Subgraph; height?: number; emptyMessage?: string };

const ENTITY_COLOR = "#1f6f78";
const VALUE_COLOR = "#a15c2e";

export function GraphView({ subgraph, height = 340, emptyMessage }: Props) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<SubgraphEdge | null>(null);

  const graphData = useMemo(
    () => ({
      nodes: subgraph.nodes.map((n) => ({ id: n.id, kind: n.kind })),
      links: subgraph.edges.map((e) => ({ ...e })),
    }),
    [subgraph]
  );

  const edgesForSelectedNode = useMemo(
    () =>
      selectedNodeId
        ? subgraph.edges.filter((e) => e.source === selectedNodeId || e.target === selectedNodeId)
        : [],
    [subgraph, selectedNodeId]
  );

  if (subgraph.nodes.length === 0) {
    return <div className="graph-empty">{emptyMessage ?? "图谱目前是空的。"}</div>;
  }

  return (
    <div className="graph-wrap">
      <div className="graph-canvas">
        <ForceGraph2D
          graphData={graphData}
          height={height}
          nodeLabel={(node: any) => node.id}
          nodeColor={(node: any) =>
            node.id === selectedNodeId ? "#ab3b2c" : node.kind === "entity" ? ENTITY_COLOR : VALUE_COLOR
          }
          linkLabel={(link: any) => `${link.relation} (confidence ${link.confidence.toFixed(2)})`}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          linkWidth={(link: any) =>
            selectedEdge && link.triple_id === selectedEdge.triple_id ? 3 : 1
          }
          onLinkClick={(link: any) => {
            setSelectedEdge(link as SubgraphEdge);
            setSelectedNodeId(null);
          }}
          onNodeClick={(node: any) => {
            setSelectedNodeId(node.id);
            setSelectedEdge(null);
          }}
          onBackgroundClick={() => {
            setSelectedNodeId(null);
            setSelectedEdge(null);
          }}
        />
      </div>
      <div className="graph-inspector">
        {selectedEdge && (
          <>
            <div className="graph-inspector-title">选中的三元组</div>
            <dl>
              <dt>主体</dt>
              <dd>{selectedEdge.source}</dd>
              <dt>关系</dt>
              <dd>{selectedEdge.relation}</dd>
              <dt>数值</dt>
              <dd>{selectedEdge.target}</dd>
              <dt>置信度</dt>
              <dd>{selectedEdge.confidence.toFixed(2)}</dd>
              <dt>triple_id</dt>
              <dd className="mono">{selectedEdge.triple_id}</dd>
            </dl>
          </>
        )}
        {selectedNodeId && (
          <>
            <div className="graph-inspector-title">{selectedNodeId}</div>
            <p className="muted">与该节点相关的三元组，共 {edgesForSelectedNode.length} 条：</p>
            <ul className="node-edge-list">
              {edgesForSelectedNode.map((e) => (
                <li key={e.triple_id} onClick={() => setSelectedEdge(e)}>
                  <span className="mono">{e.relation}</span>
                  <span className="muted"> · conf {e.confidence.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </>
        )}
        {!selectedEdge && !selectedNodeId && (
          <div className="graph-inspector-hint">点击节点查看相关的全部三元组，点击连线查看单条详情。</div>
        )}
      </div>
    </div>
  );
}
