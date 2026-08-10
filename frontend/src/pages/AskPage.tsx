import { useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { AnswerCard } from "../components/AnswerCard";
import { useAsk } from "../hooks/useAsk";

type LocationState = { initialQuestion?: string } | null;

export function AskPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { question, setQuestion, loading, error, result, submit } = useAsk();

  // Homepage's Q&A box can land here with a question pre-filled via router
  // state (navigate("/ask", { state: { initialQuestion } })) — fires it
  // automatically instead of making the user retype and click again.
  // Cleared afterward (replace, no new history entry) so a later refresh
  // on /ask doesn't silently re-ask it. Ref guards StrictMode's
  // double-invoke in dev from firing the request twice.
  const consumedInitial = useRef(false);
  useEffect(() => {
    if (consumedInitial.current) return;
    const state = location.state as LocationState;
    const initial = state?.initialQuestion;
    if (!initial) return;
    consumedInitial.current = true;
    setQuestion(initial);
    submit(initial);
    navigate(location.pathname, { replace: true, state: {} });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="panel">
      <div className="row">
        <textarea
          placeholder="向知识图谱提问，例如：Bi2Te3 单级 TEC 的 delta_T_max_K 大概是多少？"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
          }}
        />
        <button onClick={() => submit()} disabled={loading}>
          {loading ? "查询中…" : "提问"}
        </button>
      </div>

      {error && <div className="card error-card">请求失败：{error}</div>}

      {result && <AnswerCard result={result} />}
    </div>
  );
}
