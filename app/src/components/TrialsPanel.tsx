import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExploreOptions, Status, TrialRecord } from "../types";
import TrialCard, { siteLabel } from "./TrialCard";

interface Props {
  status: Status;
  epoch: number;
  onChanged: () => void;
  onError: (e: string) => void;
}

export default function TrialsPanel({ status, epoch, onChanged, onError }: Props) {
  const [trials, setTrials] = useState<TrialRecord[]>([]);
  const [opts, setOpts] = useState<ExploreOptions>({ budget: 20, max_gap: 2, margin: 0.5, auto_keep: false });
  const [running, setRunning] = useState(false);
  const [selected, setSelected] = useState<TrialRecord | null>(null);
  const [exportPath, setExportPath] = useState("");
  const [vectors, setVectors] = useState(false);
  const [exported, setExported] = useState<string | null>(null);

  const load = () => api.trials(200, false).then(setTrials).catch((e) => onError(String(e)));

  useEffect(() => {
    load();
  }, [epoch, status.session_id]);

  useEffect(() => {
    const un = api.onTrial((t) => setTrials((xs) => [t, ...xs]));
    return () => {
      un.then((f) => f());
    };
  }, []);

  const explore = async () => {
    setRunning(true);
    try {
      await api.explore(opts);
      await load();
      onChanged();
    } catch (e) {
      onError(String(e));
    } finally {
      setRunning(false);
    }
  };

  const act = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
      await load();
      onChanged();
    } catch (e) {
      onError(String(e));
    }
  };

  const lastKept = status.snapshot.trials[status.snapshot.trials.length - 1];
  const kept = new Set(status.snapshot.trials);
  // A trial can be kept only if it passed against the current model state.
  const current = (t: TrialRecord) => t.session_id === status.session_id && t.snapshot_id === status.snapshot.id;
  const num = (k: keyof ExploreOptions) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setOpts({ ...opts, [k]: Number(e.target.value) });

  return (
    <div className="split">
      <div className="stack grow">
        <div className="card">
          <h2>Explore</h2>
          <p className="muted">
            Tries promoting the runner-up (and any end-of-turn token) at every near-tie step of every probe. Each
            trial, passed or rejected, becomes a ledger row for the navigator.
          </p>
          <div className="row wrap">
            <label>
              budget <input type="number" value={opts.budget} onChange={num("budget")} />
            </label>
            <label>
              max gap <input type="number" step="0.1" value={opts.max_gap} onChange={num("max_gap")} />
            </label>
            <label>
              margin <input type="number" step="0.1" value={opts.margin} onChange={num("margin")} />
            </label>
            <label>
              <input type="checkbox" checked={opts.auto_keep} onChange={(e) => setOpts({ ...opts, auto_keep: e.target.checked })} />{" "}
              auto-keep
            </label>
            <button className="primary" disabled={running} onClick={explore}>
              {running ? "Running…" : "Run"}
            </button>
          </div>
        </div>
        <div className="card">
          <h2>Trials this session</h2>
          <table className="table clickable">
            <thead>
              <tr>
                <th>site</th>
                <th>proposer</th>
                <th>gap</th>
                <th>stage</th>
                <th>verdict</th>
                <th>ms</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {trials.map((t) => (
                <tr key={t.id} className={selected?.id === t.id ? "hl" : ""} onClick={() => setSelected(t)}>
                  <td className="mono small">{siteLabel(t)}</td>
                  <td className="small">{t.proposer}</td>
                  <td className="mono">{t.site.gap_before?.toFixed(2) ?? ""}</td>
                  <td>{t.verdict.stage}</td>
                  <td>
                    <span className={t.verdict.keep ? "chip good" : "chip bad"}>
                      {t.verdict.keep ? "pass" : t.verdict.reasons[0] ?? "fail"}
                    </span>
                  </td>
                  <td className="mono">{t.wall_ms}</td>
                  <td onClick={(e) => e.stopPropagation()}>
                    {kept.has(t.id) ? (
                      <button
                        disabled={t.id !== lastKept}
                        title={t.id !== lastKept ? "Unkeep the most recent edit first" : ""}
                        onClick={() => act(() => api.unkeep(t.id))}
                      >
                        Unkeep
                      </button>
                    ) : t.verdict.keep && !current(t) ? (
                      <button title="Gated on an earlier model state" onClick={() => act(() => api.retest(t.id))}>
                        Re-test
                      </button>
                    ) : (
                      <button disabled={!t.verdict.keep} onClick={() => act(() => api.keep(t.id))}>
                        Keep
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="stack side">
        {selected && <TrialCard trial={selected} />}
        <div className="card">
          <h2>Export ledger</h2>
          <p className="muted small">JSONL, one trial per line, joined with its session. Navigator training input.</p>
          <input className="mono" placeholder="/path/to/trials.jsonl" value={exportPath} onChange={(e) => setExportPath(e.target.value)} />
          <label className="small">
            <input type="checkbox" checked={vectors} onChange={(e) => setVectors(e.target.checked)} /> include delta vectors
          </label>
          <button
            disabled={!exportPath}
            onClick={() =>
              api
                .exportLedger(exportPath, vectors)
                .then((n) => setExported(`Wrote ${n} trials`))
                .catch((e) => onError(String(e)))
            }
          >
            Export
          </button>
          {exported && <div className="muted small">{exported}</div>}
        </div>
      </div>
    </div>
  );
}
