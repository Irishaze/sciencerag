import { Fragment, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { Subgraph, SubgraphEdge } from "../types";

type Props = { subgraph: Subgraph; height?: number; emptyMessage?: string };

const VALUE_COLOR = "#a15c2e";
const UNCLASSIFIED_COLOR = "#6b6b6b";
const SELECTED_COLOR = "#ab3b2c";
// Same red as --blocking in index.css (App.css can't reach CSS vars from
// inside canvas-drawn force-graph links, so it's re-declared here).
const CONFLICT_COLOR = "#ab3b2c";
const DEFAULT_LINK_COLOR = "#999999";

// entity_type is an open-ended, AI-generated set of names (sciencerag/
// priors/ontology_generator.py) — not a fixed enum we can hardcode a
// color per name for. Hash the name into a small fixed palette instead so
// the same type always gets the same color across renders/sessions
// without needing to know the type names in advance.
const ENTITY_TYPE_PALETTE = ["#1f6f78", "#3d5a80", "#5f7a3d", "#7a4f8f", "#9c4f6b", "#8f5f2e"];

function colorForEntityType(entityType: string | null): string {
  if (!entityType) return VALUE_COLOR;
  if (entityType === "Unclassified") return UNCLASSIFIED_COLOR;
  let hash = 0;
  for (let i = 0; i < entityType.length; i++) {
    hash = (hash * 31 + entityType.charCodeAt(i)) >>> 0;
  }
  return ENTITY_TYPE_PALETTE[hash % ENTITY_TYPE_PALETTE.length];
}

// Older triples (written before describe_relation existed) have
// description: null — fall back to the raw technical name rather than
// showing nothing.
function phraseFor(edge: SubgraphEdge): string {
  return edge.description || edge.relation;
}

export function GraphView({ subgraph, height = 340, emptyMessage }: Props) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<SubgraphEdge | null>(null);
  // The force layout starts every node at (0,0) and settles outward from
  // there — left uncentered, the canvas's default viewport (also anchored
  // near its own origin) only ever shows whichever corner of the spread-out
  // result happens to land near (0,0), so most of the graph sits off-screen
  // until the user manually pans/zooms. zoomToFit on every engine-settle
  // (initial load and any subsequent "refresh") re-frames the whole result.
  const fgRef = useRef<any>(null);

  // Edges only carry node ids (source/target) — this is the id -> display
  // label lookup used everywhere a raw id would otherwise show up in the
  // UI (node/value ids are opaque now, see types.ts's SubgraphNode).
  const labelById = useMemo(() => {
    const map = new Map<string, string>();
    for (const n of subgraph.nodes) map.set(n.id, n.label);
    return map;
  }, [subgraph]);
  const labelFor = (id: string) => labelById.get(id) ?? id;

  // conflicts_with only points one way (the later-written triple ->
  // the one it contradicts, see kg.py:KGTriple.conflicts_with) — walk
  // both directions so the graph flags EITHER side of a conflicting pair,
  // not just the one that happens to carry the pointer.
  const edgeByTripleId = useMemo(() => {
    const map = new Map<string, SubgraphEdge>();
    for (const e of subgraph.edges) map.set(e.triple_id, e);
    return map;
  }, [subgraph]);
  const conflictPartner = useMemo(() => {
    const map = new Map<string, string>();
    for (const e of subgraph.edges) {
      if (!e.conflicts_with) continue;
      map.set(e.triple_id, e.conflicts_with);
      map.set(e.conflicts_with, e.triple_id);
    }
    return map;
  }, [subgraph]);

  const graphData = useMemo(
    () => ({
      nodes: subgraph.nodes.map((n) => ({ id: n.id, kind: n.kind, label: n.label, entity_type: n.entity_type })),
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

  // entity_type is open-ended (see colorForEntityType above) — so unlike a
  // fixed-category legend, this one is built from whatever types actually
  // appear in the current subgraph, not a hardcoded list that would drift
  // out of sync with the ontology.
  const entityTypeLegend = useMemo(() => {
    const types = new Set<string>();
    for (const n of subgraph.nodes) {
      if (n.kind === "entity" && n.entity_type) types.add(n.entity_type);
    }
    return Array.from(types).sort();
  }, [subgraph]);
  const hasValueNodes = useMemo(() => subgraph.nodes.some((n) => n.kind === "value"), [subgraph]);

  if (subgraph.nodes.length === 0) {
    return <div className="graph-empty">{emptyMessage ?? "图谱目前是空的。"}</div>;
  }

  return (
    <div className="graph-wrap">
      <div className="graph-canvas">
        <ForceGraph2D
          ref={fgRef}
          graphData={graphData}
          height={height}
          // Default cooldownTime is 15s — onEngineStop (and so zoomToFit,
          // above) wouldn't fire until then, leaving the graph looking
          // unfit/off-center for most of that wait. This graph is small
          // enough (tens of nodes) to settle well within 2s.
          cooldownTime={2000}
          onEngineStop={() => fgRef.current?.zoomToFit(400, 40)}
          nodeLabel={(node: any) => node.label}
          nodeColor={(node: any) =>
            node.id === selectedNodeId ? SELECTED_COLOR : colorForEntityType(node.kind === "entity" ? node.entity_type : null)
          }
          linkLabel={(link: any) =>
            conflictPartner.has(link.triple_id) ? `⚠️ 与另一条数据冲突 · ${phraseFor(link)}` : phraseFor(link)
          }
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          linkColor={(link: any) => (conflictPartner.has(link.triple_id) ? CONFLICT_COLOR : DEFAULT_LINK_COLOR)}
          linkLineDash={(link: any) => (conflictPartner.has(link.triple_id) ? [3, 2] : null)}
          linkWidth={(link: any) =>
            selectedEdge && link.triple_id === selectedEdge.triple_id
              ? 3
              : conflictPartner.has(link.triple_id)
                ? 2
                : 1
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
        <div className="graph-legend">
          {entityTypeLegend.map((type) => (
            <div key={type} className="graph-legend-item">
              <span className="graph-legend-dot" style={{ background: colorForEntityType(type) }} />
              {type}
            </div>
          ))}
          {hasValueNodes && (
            <div className="graph-legend-item">
              <span className="graph-legend-dot" style={{ background: VALUE_COLOR }} />
              数值
            </div>
          )}
          {conflictPartner.size > 0 && (
            <div className="graph-legend-item">
              <span className="graph-legend-line" />
              有冲突
            </div>
          )}
        </div>
      </div>
      <div className="graph-inspector">
        {selectedEdge && (
          <>
            <div className="graph-inspector-title">
              {labelFor(selectedEdge.source)} · {phraseFor(selectedEdge)}
            </div>
            <p className="graph-inspector-plain">
              <strong>{labelFor(selectedEdge.source)}</strong>{"的"}
              <strong>{phraseFor(selectedEdge)}</strong>{"是 "}
              <strong>{labelFor(selectedEdge.target)}</strong>
            </p>
            {conflictPartner.has(selectedEdge.triple_id) &&
              (() => {
                const partner = edgeByTripleId.get(conflictPartner.get(selectedEdge.triple_id)!);
                return (
                  <div className="graph-conflict-notice">
                    <strong>⚠️ 与另一条数据冲突：</strong>
                    {partner ? (
                      <>
                        同一个字段（{phraseFor(selectedEdge)}）还有另一条取值 <strong>{labelFor(partner.target)}</strong>
                        {"（置信度 " + partner.confidence.toFixed(2) + "），"}
                        两条都保留在图谱里，未自动取舍。
                        <button type="button" className="graph-conflict-jump" onClick={() => setSelectedEdge(partner)}>
                          查看另一条 →
                        </button>
                      </>
                    ) : (
                      "冲突的另一条数据不在当前子图范围内。"
                    )}
                  </div>
                );
              })()}
            <details className="graph-inspector-technical">
              <summary>技术细节</summary>
              <dl>
                <dt>关系名</dt>
                <dd className="mono">{selectedEdge.relation}</dd>
                <dt>置信度</dt>
                <dd>{selectedEdge.confidence.toFixed(2)}</dd>
                <dt>triple_id</dt>
                <dd className="mono">{selectedEdge.triple_id}</dd>
                {Object.keys(selectedEdge.conditions).length > 0 && (
                  <>
                    <dt>测量条件</dt>
                    <dd className="mono">
                      {Object.entries(selectedEdge.conditions)
                        .map(([key, value]) => `${key}=${value}`)
                        .join("，")}
                    </dd>
                  </>
                )}
                {selectedEdge.evidence_detail &&
                  Object.entries(selectedEdge.evidence_detail).map(([key, value]) =>
                    value === null || value === undefined ? null : (
                      <Fragment key={key}>
                        <dt>{key}</dt>
                        <dd>{String(value)}</dd>
                      </Fragment>
                    )
                  )}
              </dl>
            </details>
          </>
        )}
        {selectedNodeId && (
          <>
            <div className="graph-inspector-title">{labelFor(selectedNodeId)}</div>
            <p className="muted">已知的相关信息，共 {edgesForSelectedNode.length} 条：</p>
            <ul className="node-edge-list">
              {edgesForSelectedNode.map((e) => (
                <li
                  key={e.triple_id}
                  className={conflictPartner.has(e.triple_id) ? "conflict" : undefined}
                  onClick={() => setSelectedEdge(e)}
                >
                  {conflictPartner.has(e.triple_id) && <span title="与另一条数据冲突">⚠️ </span>}
                  {phraseFor(e)}
                  {"："}
                  {labelFor(e.target === selectedNodeId ? e.source : e.target)}
                </li>
              ))}
            </ul>
          </>
        )}
        {!selectedEdge && !selectedNodeId && (
          <div className="graph-inspector-hint">点击节点查看相关信息，点击连线查看单条详情。</div>
        )}
      </div>
    </div>
  );
}
