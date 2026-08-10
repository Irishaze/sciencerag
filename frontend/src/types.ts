// Mirrors sciencerag/ask/models.py, sciencerag/report/models.py, and the
// Source union in sciencerag/priors/models.py — kept in sync by hand since
// this is a v1 frontend, not generated from the JSON Schema files.

export type SourcePaper = { type: "paper"; doi: string; span: string | null };
export type SourceKGTriple = { type: "kg_triple"; triple_id: string };
export type Source = SourcePaper | SourceKGTriple;

export type SubgraphNode = { id: string; kind: "entity" | "value" };
export type SubgraphEdge = {
  source: string;
  target: string;
  relation: string;
  triple_id: string;
  confidence: number;
};
export type Subgraph = { nodes: SubgraphNode[]; edges: SubgraphEdge[] };

export type AskResponse = {
  status: "ok";
  answer: string;
  subgraph: Subgraph;
  sources: Source[];
  fallback_used: boolean;
  coverage_note: string | null;
  trace_id: string;
};

export type ErrorResponse = {
  status: "error";
  error: { category: string; message: string; detail: Record<string, unknown> };
  trace_id: string;
};

export type ReportListEntry = { filename: string; stem: string };

export type Anomaly = {
  check: "energy_balance" | "pde_residual" | "ood";
  severity: "info" | "warning" | "blocking";
  evidence: Record<string, unknown>;
};

export type Deviation = {
  field: string;
  source: "prior_comparison" | "benchmark_comparison";
  reference_id: string;
  actual: number;
  reference_min: number | null;
  reference_max: number | null;
  verdict: "within_range" | "deviation";
};

export type Evaluation = {
  verdict: "consistent" | "deviation_found" | "insufficient_benchmark";
  deviations: Deviation[];
  sources: Source[];
};

export type KeyResult = {
  field: string;
  value: number;
  unit: string | null;
  confidence_label: "high" | "check_flagged" | "no_anomaly_data";
};

export type UpdatePackage = {
  surrogate_update: {
    recommended_training_samples: { run_id: string; region: string; reason: string }[];
    loss_reweighting: Record<string, number>;
    hyperparameter_direction: string;
  } | null;
  kg_candidates: unknown[];
  blocked: boolean;
};

export type ReportDetail = {
  status: "ok";
  run_id: string;
  generated_at: string;
  objective_and_constraints: { objective: string | null; constraints: Record<string, number> };
  spec_summary: Record<string, number>;
  key_results: KeyResult[];
  literature_comparison: Evaluation;
  anomalies_and_cautions: Anomaly[];
  update_proposal_summary: UpdatePackage;
  citations: Source[];
  markdown: string;
  trace_id: string;
};
