// Small hand-drawn line icons, all sharing one stroke language (1.6px,
// round caps/joins, currentColor) so they read as one family wherever
// they're reused — not a generic icon-library import.

const BASE = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

// Two points joined by the same dashed stroke GraphView uses for a
// conflicting edge — a claim and the source it traces back to.
export function TraceIcon({ size = 22 }: { size?: number }) {
  return (
    <svg {...BASE} width={size} height={size}>
      <circle cx="6" cy="18" r="2.3" />
      <circle cx="18" cy="6" r="2.3" />
      <path d="M8.2 15.8 L15.8 8.2" strokeDasharray="2.4 2.4" />
    </svg>
  );
}

export function ShieldCheckIcon({ size = 22 }: { size?: number }) {
  return (
    <svg {...BASE} width={size} height={size}>
      <path d="M12 3 L19 6 V11 C19 16 16 19.5 12 21 C8 19.5 5 16 5 11 V6 Z" />
      <path d="M8.5 12 L10.8 14.5 L15.5 9.5" />
    </svg>
  );
}

// Echoes the actual force-graph nodes/edges on /graph — the same
// three-dot shape a real subgraph collapses to, not a generic "network"
// glyph.
export function GraphIcon({ size = 26 }: { size?: number }) {
  return (
    <svg {...BASE} width={size} height={size}>
      <circle cx="6" cy="7" r="2.1" />
      <circle cx="18" cy="7" r="2.1" />
      <circle cx="12" cy="18" r="2.1" />
      <path d="M8 8.1 L10.3 16.1 M16 8.1 L13.7 16.1 M8.1 7 H15.9" />
    </svg>
  );
}

export function DocumentIcon({ size = 26 }: { size?: number }) {
  return (
    <svg {...BASE} width={size} height={size}>
      <path d="M6 3 H14 L18 7 V21 H6 Z" />
      <path d="M14 3 V7 H18" />
      <path d="M9 12 H15 M9 15 H15 M9 18 H12.5" />
    </svg>
  );
}

// A single graph node with two stub edges — one KG triple, not the whole
// three-node subgraph GraphIcon stands for.
export function TripleIcon({ size = 16 }: { size?: number }) {
  return (
    <svg {...BASE} width={size} height={size}>
      <circle cx="12" cy="12" r="2.4" />
      <path d="M12 6.5 V9.2 M12 14.8 V17.5" />
    </svg>
  );
}
