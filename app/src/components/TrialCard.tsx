import type { TrialRecord } from "../types";

export function siteLabel(t: TrialRecord) {
  const o = t.delta.origin;
  if (!o) return t.delta.tensor;
  return `${o.probe_id}[${o.step}] ${JSON.stringify(o.from_text)} → ${JSON.stringify(o.target_text)}`;
}

export default function TrialCard({ trial }: { trial: TrialRecord }) {
  const v = trial.verdict;
  return (
    <div className={v.keep ? "card trial pass" : "card trial fail"}>
      <div className="row">
        <strong className="grow">{v.keep ? "Passed all gates" : "Rejected"}</strong>
        <span className="chip">{v.stage}</span>
        <span className="muted small">{trial.wall_ms} ms</span>
      </div>
      <div className="mono small">{siteLabel(trial)}</div>
      {v.reasons.length > 0 && (
        <div className="row wrap">
          {v.reasons.map((r) => (
            <span key={r} className="chip bad">
              {r}
            </span>
          ))}
        </div>
      )}
      <div className="row wrap">
        {v.gates.map((g) => (
          <span key={g.name} className={g.pass ? "chip good" : "chip bad"} title={JSON.stringify(g.metrics)}>
            {g.name}
          </span>
        ))}
      </div>
      <table className="table small">
        <thead>
          <tr>
            <th>probe</th>
            <th>role</th>
            <th>len</th>
            <th>diverges at</th>
            <th>predicted</th>
            <th>answer</th>
          </tr>
        </thead>
        <tbody>
          {trial.results.map((r) => (
            <tr key={r.probe_id} title={r.new_text ?? ""}>
              <td className="mono">{r.probe_id}</td>
              <td>{r.role}</td>
              <td className="mono">
                {r.base_len}
                {r.new_len !== null && r.new_len !== r.base_len ? ` → ${r.new_len}` : ""}
              </td>
              <td className="mono">{r.first_divergence ?? (r.identical ? "same" : "–")}</td>
              <td className="mono">{r.predicted_first_flip ?? "–"}</td>
              <td>{r.answer_ok === null ? "" : r.answer_ok ? "ok" : "lost"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="muted small">
        cos to controls {trial.site.max_cos_controls.toFixed(3)} · cos to other probes{" "}
        {trial.site.max_cos_other_probes.toFixed(3)} · max |Δ| {trial.site.delta_max_abs.toExponential(2)}
      </div>
    </div>
  );
}
