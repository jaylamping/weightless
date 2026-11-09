import { useEffect, useState } from "react";
import { api } from "../api";
import type { Status, Trace, TrialRecord } from "../types";
import TrialCard from "./TrialCard";

interface Props {
  status: Status;
  epoch: number;
  onTrial: () => void;
  onError: (e: string) => void;
}

// Near-ties (small top-2 gap) are where single edits work, so the gap drives the colour.
function gapClass(gap: number) {
  if (gap < 0.5) return "tok tie0";
  if (gap < 2) return "tok tie1";
  if (gap < 5) return "tok tie2";
  return "tok";
}

export default function TracePanel({ status, epoch, onTrial, onError }: Props) {
  const [traces, setTraces] = useState<Trace[] | null>(null);
  const [probe, setProbe] = useState(0);
  const [step, setStep] = useState<number | null>(null);
  const [margin, setMargin] = useState(0.5);
  const [busy, setBusy] = useState(false);
  const [trial, setTrial] = useState<TrialRecord | null>(null);

  useEffect(() => {
    setTraces(null);
    setStep(null);
    api.baseline().then(setTraces).catch((e) => onError(String(e)));
  }, [epoch, status.session_id]);

  const probes = status.spec.probes;
  const trace = traces?.[probe];
  const s = step !== null ? trace?.steps[step] : undefined;

  const tryTarget = async (target: number) => {
    if (step === null) return;
    setBusy(true);
    try {
      setTrial(await api.tryNudge(probes[probe].id, step, target, margin));
      onTrial();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="split">
      <div className="stack grow">
        <div className="card">
          <div className="row">
            <select value={probe} onChange={(e) => (setProbe(Number(e.target.value)), setStep(null))}>
              {probes.map((p, i) => (
                <option key={p.id} value={i}>
                  {p.role === "control" ? "◇ " : "◆ "}
                  {p.id}
                </option>
              ))}
            </select>
            <span className="muted grow">{probes[probe]?.prompt.text}</span>
            {trace && (
              <span className="muted small">
                {trace.steps.length} tokens · {trace.stop} · {trace.wall_ms} ms
              </span>
            )}
          </div>
        </div>
        <div className="card">
          {!traces && <div className="muted">Tracing baseline…</div>}
          {trace && (
            <div className="tokens">
              {trace.steps.map((st, i) => (
                <span
                  key={i}
                  className={gapClass(st.gap) + (i === step ? " sel" : "")}
                  title={`step ${i} · gap ${st.gap.toFixed(2)}`}
                  onClick={() => (setStep(i), setTrial(null))}
                >
                  {st.text.replace(/\n/g, "⏎")}
                </span>
              ))}
            </div>
          )}
          <div className="legend muted small">
            <span className="tok tie0">gap &lt; 0.5</span>
            <span className="tok tie1">&lt; 2</span>
            <span className="tok tie2">&lt; 5</span> click a token to see its candidates
          </div>
        </div>
        {trial && <TrialCard trial={trial} />}
      </div>

      <div className="card side">
        <h2>Step {step ?? "–"}</h2>
        {s ? (
          <>
            <div className="muted small">
              gap {s.gap.toFixed(3)} · ‖h‖ {s.hidden_norm.toFixed(1)}
            </div>
            <div className="row">
              <label className="muted small">margin</label>
              <input type="number" step="0.1" value={margin} onChange={(e) => setMargin(Number(e.target.value))} />
            </div>
            <table className="table">
              <tbody>
                {s.top.map((c, i) => (
                  <tr key={c.id} className={i === 0 ? "hl" : ""}>
                    <td className="mono">{JSON.stringify(c.text)}</td>
                    <td className="mono right">{c.logit.toFixed(2)}</td>
                    <td>
                      {i > 0 && (
                        <button disabled={busy || !status.capabilities.full_logits} onClick={() => tryTarget(c.id)}>
                          Try
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">
              Try builds the rank-1 edit W[k] += (δ/‖h‖²)·h on {status.info.lm_head_tensor}, runs every probe and
              control with it, gates the result and logs it to the ledger.
            </p>
          </>
        ) : (
          <div className="muted">Pick a token.</div>
        )}
      </div>
    </div>
  );
}
