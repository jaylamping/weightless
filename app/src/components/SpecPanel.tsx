import { useState } from "react";
import { api } from "../api";
import type { Status } from "../types";

interface Props {
  status: Status;
  onChanged: () => void;
  onError: (e: string) => void;
}

// The spec is edited as JSON for now: every field stays reachable. A form editor can come later.
export default function SpecPanel({ status, onChanged, onError }: Props) {
  const [text, setText] = useState(JSON.stringify(status.spec, null, 2));
  const [saving, setSaving] = useState(false);

  const apply = async () => {
    setSaving(true);
    try {
      await api.setSpec(JSON.parse(text));
      onChanged();
    } catch (e) {
      onError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="stack">
      <div className="card">
        <h2>Spec</h2>
        <p className="muted">
          The goal, the probes an edit should improve, and the controls it must leave unchanged. Applying a new spec
          starts a new ledger session from the base model (kept edits are reverted), so rows never mix specs.
        </p>
        <textarea className="mono code" spellCheck={false} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="row">
          <button className="primary" disabled={saving} onClick={apply}>
            Apply spec
          </button>
          <button onClick={() => setText(JSON.stringify(status.spec, null, 2))}>Reset</button>
        </div>
      </div>
    </div>
  );
}
