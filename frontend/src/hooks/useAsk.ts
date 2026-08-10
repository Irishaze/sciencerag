import { useState } from "react";
import { ApiError, ask } from "../api";
import type { AskResponse } from "../types";

/** Shared question/answer state — used by both the homepage's inline Q&A
 * and the dedicated /ask page, so the two don't drift into two slightly
 * different implementations of the same request. */
export function useAsk(initialQuestion = "") {
  const [question, setQuestion] = useState(initialQuestion);
  const [maxHits, setMaxHits] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);

  async function submit(questionOverride?: string) {
    const q = (questionOverride ?? question).trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      const response = await ask(q, maxHits);
      setResult(response);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.category}: ${e.message}` : String(e));
    } finally {
      setLoading(false);
    }
  }

  return { question, setQuestion, maxHits, setMaxHits, loading, error, result, submit };
}
