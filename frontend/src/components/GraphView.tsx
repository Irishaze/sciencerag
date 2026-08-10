import { useMemo, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { Subgraph, SubgraphEdge } from "../types";

type Props = { subgraph: Subgraph };

const ENTITY_COLOR = "#1f6f78";
const VALUE_COLOR = "#a15c2e";

export function GraphView({ subgraph }: Props) {
  const [selected, setSelected] = useState<SubgraphEdge | null>(null);

  const graphData = useMemo(
    () => ({
      nodes: subgraph.nodes.map((n) => ({ id: n.id, kind: n.kind })),
      links: subgraph.edges.map((e) => ({ ...e })),
    }),
    [subgraph]
  );

  if (subgraph.nodes.length === 0) {
    return (
      <div className="graph-empty">
        该回答没有子图 — 走的是文献回退路径，不是从知识图谱检索的。
      </div>
    );
  }

  return (
    <div className="graph-wrap">
      <div className="graph-canvas">
        <ForceGraph2D
          graphData={graphData}
          height={340}
          nodeLabel={(node: any) => node.id}
          nodeColor={(node: any) => (node.kind === "entity" ? ENTITY_COLOR : VALUE_COLOR)}
          linkLabel={(link: any) => `${link.relation} (confidence ${link.confidence.toFixed(2)})`}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          onLinkClick={(link: any) => setSelected(link as SubgraphEdge)}
          onNodeClick={(node: any) => {
            const edge = subgraph.edges.find((e) => e.source === node.id || e.target === node.id);
            if (edge) setSelected(edge);
          }}
        />
      </div>
      <div className="graph-inspector">
        {selected ? (
          <>
            <div className="graph-inspector-title">选中的三元组</div>
            <dl>
              <dt>主体</dt>
              <dd>{selected.source}</dd>
              <dt>关系</dt>
              <dd>{selected.relation}</dd>
              <dt>取值</dt>
              <dd>{selected.target}</dd>
              <dt>置信度</dt>
              <dd>{selected.confidence.toFixed(2)}</dd>
              <dt>triple_id</dt>
              <dd className="mono">{selected.triple_id}</dd>
            </dl>
          </>
        ) : (
          <div className="graph-inspector-hint">点击节点或连线查看来源与置信度。</div>
        )}
      </div>
    </div>
  );
}
