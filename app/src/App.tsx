import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { Status } from "./types";
import ModelPanel from "./components/ModelPanel";
import SpecPanel from "./components/SpecPanel";
import TracePanel from "./components/TracePanel";
import TrialsPanel from "./components/TrialsPanel";

type Tab = "model" | "spec" | "trace" | "trials";

const TABS: { id: Tab; label: string; needsModel: boolean }[] = [
  { id: "model", label: "Model", needsModel: false },
  { id: "spec", label: "Spec", needsModel: true },
  { id: "trace", label: "Trace & edit", needsModel: true },
  { id: "trials", label: "Trials & ledger", needsModel: true },
];

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [tab, setTab] = useState<Tab>("model");
  const [error, setError] = useState<string | null>(null);
  // Bumped when the working model changes (keep/unkeep/spec), so panels refetch baselines.
  const [epoch, setEpoch] = useState(0);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.status());
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const changed = useCallback(() => {
    setEpoch((x) => x + 1);
    refresh();
  }, [refresh]);

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="dot" /> Weightless
        </div>
        <nav>
          {TABS.map((t) => (
            <button
              key={t.id}
              className={tab === t.id ? "nav active" : "nav"}
              disabled={t.needsModel && !status}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        {status && (
          <div className="side-stats">
            <Stat label="trials" value={status.trials} />
            <Stat label="passed gates" value={status.passed} />
            <Stat label="kept" value={status.snapshot.trials.length} />
            <div className="muted small">session {status.session_id}</div>
            <div className="muted small">snapshot {status.snapshot.id ?? "base"}</div>
          </div>
        )}
      </aside>
      <main>
        <header className="topbar">
          {status ? (
            <>
              <strong>{status.info.id}</strong>
              <span className="chip">{status.info.architecture}</span>
              <span className="chip">{status.info.device}</span>
              <span className="chip">{status.info.dtype}</span>
              <span className="chip">{status.info.backend}</span>
              <span className="muted small mono">{status.info.fingerprint.slice(0, 12)}</span>
            </>
          ) : (
            <span className="muted">No model loaded</span>
          )}
        </header>
        {error && (
          <div className="error" onClick={() => setError(null)}>
            {error}
          </div>
        )}
        <section className="content">
          {tab === "model" && <ModelPanel status={status} onLoaded={changed} onError={setError} />}
          {tab === "spec" && status && <SpecPanel status={status} onChanged={changed} onError={setError} />}
          {tab === "trace" && status && <TracePanel status={status} epoch={epoch} onTrial={refresh} onError={setError} />}
          {tab === "trials" && status && <TrialsPanel status={status} epoch={epoch} onChanged={changed} onError={setError} />}
        </section>
      </main>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
