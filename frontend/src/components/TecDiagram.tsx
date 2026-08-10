// Stylized cross-section of a single-stage TEC module: alternating P/N legs
// between two ceramic plates, wired in series (the zigzag copper bridges),
// cold side pulling heat up (Qc) and hot side rejecting it (Qh). This is
// the actual subject of the whole system — not a decorative graphic — so
// it's built as a real (if simplified) schematic, not an abstract shape.
const LEG_COUNT = 7;
const WIDTH = 460;
const HEIGHT = 210;
const PLATE_HEIGHT = 12;
const LEG_TOP = 46;
const LEG_BOTTOM = 152;
const MARGIN_X = 34;

const legSpan = (WIDTH - MARGIN_X * 2) / LEG_COUNT;
const legWidth = legSpan * 0.52;

export function TecDiagram() {
  const legs = Array.from({ length: LEG_COUNT }, (_, i) => {
    const x = MARGIN_X + i * legSpan + (legSpan - legWidth) / 2;
    const isCold = i % 2 === 0;
    return { x, isCold };
  });

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="tec-diagram"
      role="img"
      aria-label="TEC 模块截面示意图：冷热两侧陶瓷板之间的 P/N 型热电臂交替排列"
    >
      <text x={WIDTH / 2} y={20} textAnchor="middle" className="tec-diagram-label cold">
        COLD SIDE · Qc ↑
      </text>

      <rect x={MARGIN_X - 10} y={LEG_TOP - PLATE_HEIGHT} width={WIDTH - (MARGIN_X - 10) * 2} height={PLATE_HEIGHT} rx={2} className="tec-plate cold" />

      {legs.map((leg, i) => (
        <g key={i}>
          <rect
            x={leg.x}
            y={LEG_TOP}
            width={legWidth}
            height={LEG_BOTTOM - LEG_TOP}
            className={leg.isCold ? "tec-leg n" : "tec-leg p"}
          />
          <text x={leg.x + legWidth / 2} y={(LEG_TOP + LEG_BOTTOM) / 2 + 4} textAnchor="middle" className="tec-leg-label">
            {leg.isCold ? "N" : "P"}
          </text>
          {/* series bridges: alternate top/bottom connectors between
              adjacent legs, the way a real module is actually wired */}
          {i < legs.length - 1 &&
            (i % 2 === 0 ? (
              <rect x={leg.x + legWidth} y={LEG_TOP - 4} width={legSpan - legWidth} height={4} className="tec-bridge" />
            ) : (
              <rect x={leg.x + legWidth} y={LEG_BOTTOM} width={legSpan - legWidth} height={4} className="tec-bridge" />
            ))}
        </g>
      ))}

      <rect x={MARGIN_X - 10} y={LEG_BOTTOM} width={WIDTH - (MARGIN_X - 10) * 2} height={PLATE_HEIGHT} rx={2} className="tec-plate hot" />

      <text x={WIDTH / 2} y={HEIGHT - 6} textAnchor="middle" className="tec-diagram-label hot">
        HOT SIDE · Qh ↓
      </text>
    </svg>
  );
}
