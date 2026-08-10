import type { AskResponse, ErrorResponse, ReportDetail, ReportListEntry } from "./types";

// Same-origin in production (this app is built and served by the FastAPI
// app itself, see sciencerag/app.py); Vite's dev server proxies /sciencerag
// to the backend in dev (see vite.config.ts) so this stays empty in both.
const BASE = "";

export class ApiError extends Error {
  category: string;
  traceId: string;
  constructor(body: ErrorResponse) {
    super(body.error.message);
    this.category = body.error.category;
    this.traceId = body.trace_id;
  }
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = await response.json();
  if (!response.ok || json.status === "error") {
    throw new ApiError(json as ErrorResponse);
  }
  return json as T;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  const json = await response.json();
  if (!response.ok || json.status === "error") {
    throw new ApiError(json as ErrorResponse);
  }
  return json as T;
}

export function ask(question: string, maxHits = 10): Promise<AskResponse> {
  return postJson<AskResponse>("/sciencerag/ask", { question, max_hits: maxHits });
}

export function listReports(): Promise<ReportListEntry[]> {
  return getJson<ReportListEntry[]>("/sciencerag/reports");
}

export function getReport(stem: string): Promise<ReportDetail> {
  return getJson<ReportDetail>(`/sciencerag/reports/${encodeURIComponent(stem)}`);
}
