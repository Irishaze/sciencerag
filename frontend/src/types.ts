// Mirrors sciencerag/ask/models.py, sciencerag/report/models.py, and the
// Source union in sciencerag/priors/models.py — kept in sync by hand since
// this is a v1 frontend, not generated from the JSON Schema files.

export type SourcePaper = { type: "paper"; doi: string; span: string | null };
export type SourceKGTriple = { type: "kg_triple"; triple_id: string };
export type Source = SourcePaper | SourceKGTriple;

// `id` is opaque (entity_id for entity nodes, a triple-scoped key for
// value nodes) — `label` is the human-readable text to actually display.
// See sciencerag/priors/kg.py:KGTriple.entity_id's docstring for why these
// used to be the same field and had to be split.
export type SubgraphNode = {
  id: string;
  kind: "entity" | "value";
  label: string;
  entity_type: string | null;
};
export type SubgraphEdge = {
  source: string;
  target: string;
  relation: string;
  // AI-generated plain-Chinese phrase for what `relation` means (see
  // sciencerag/priors/ontology_generator.py:describe_relation) — null for
  // older triples written before this field existed.
  description: string | null;
  triple_id: string;
  confidence: number;
  // triple_id of the edge this one contradicts on the same subject+relation
  // (see sciencerag/priors/kg.py:KGTriple.conflicts_with) — null when this
  // edge doesn't conflict with anything currently in the graph.
  conflicts_with: string | null;
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

// Mirrors sciencerag/validate/models.py:KGCandidate and
// sciencerag/kg_approval/models.py.
export type KGCandidate = {
  subject: string;
  relation: string;
  object_value: number | null;
  object_unit: string | null;
  object_entity_id: string | null;
  object_entity_label: string | null;
  object_entity_type: string | null;
  relation_description: string | null;
  conditions: Record<string, number>;
  confidence: number;
  run_id: string;
  entity_type: string;
  dedup_status: "new" | "duplicate_confirmed" | "conflict";
  supporting_evidence: Record<string, unknown>;
};

export type PendingBatchSummary = { stem: string; count: number };
export type PendingBatchDetail = { stem: string; candidates: KGCandidate[] };

export type ApprovalResult = {
  index: number;
  status: "added" | "merged" | "conflict" | "error";
  triple_id: string | null;
  error: string | null;
};
export type ApproveResponse = { stem: string; results: ApprovalResult[]; archived: boolean };

export type Anomaly = {
  // energy_balance/pde_residual were removed (see sciencerag/validate/
  // checks.py) — both depended on a compositional multi-pair model that
  // was never a substitute for real multi-pair COMSOL calibration data.
  check: "ood";
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
