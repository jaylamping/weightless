import { useEffect, useState } from "react";
import { api, type DeviceChoice } from "../api";
import type { Status, TensorInfo } from "../types";

interface Props {
  status: Status | null;
  onLoaded: () => void;
  onError: (e: string) => void;
}

export default function ModelPanel({ status, onLoaded, onError }: Props) {
  const [path, setPath] = useState(localStorage.getItem("modelPath") ?? "");
  const [device, setDevice] = useState<DeviceChoice>("auto");
  const [dtype, setDtype] = useState("");
  const [loading, setLoading] = useState(false);
  const [tensors, setTensors] = useState<TensorInfo[]>([]);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    if (status) api.tensors().then(setTensors).catch((e) => onError(String(e)));
  }, [status?.info.fingerprint]);

  const load = async () => {
    setLoading(true);
    try {
      localStorage.setItem("modelPath", path);
      await api.loadModel(path, device, dtype || null);
      onLoaded();
    } catch (e) {
      onError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const shown = tensors.filter((t) => t.name.includes(filter)).slice(0, 400);

  return (
    <div className="stack">
      <div className="card">
        <h2>Load a model</h2>
        <p className="muted">
          A local folder with config.json, tokenizer.json and safetensors. Qwen3 dense models work today (try
          Qwen/Qwen3-0.6B); Qwen3.5 comes with the mistral.rs port. Nothing is size-capped.
        </p>
        <div className="row">
          <input className="grow mono" placeholder="/path/to/model" value={path} onChange={(e) => setPath(e.target.value)} />
          <select value={device} onChange={(e) => setDevice(e.target.value as DeviceChoice)}>
            <option value="auto">auto device</option>
            <option value="cpu">cpu</option>
            <option value="cuda">cuda</option>
            <option value="metal">metal</option>
          </select>
          <select value={dtype} onChange={(e) => setDtype(e.target.value)}>
            <option value="">default dtype</option>
            <option value="f32">f32</option>
            <option value="bf16">bf16</option>
            <option value="f16">f16</option>
          </select>
          <button className="primary" disabled={!path || loading} onClick={load}>
            {loading ? "Loading…" : "Load"}
          </button>
        </div>
      </div>

      {status && (
        <div className="grid2">
          <div className="card">
            <h2>Model</h2>
            <table className="kv">
              <tbody>
                {Object.entries(status.info).map(([k, v]) => (
                  <tr key={k}>
                    <td className="muted">{k}</td>
                    <td className="mono">{String(v)}</td>
                  </tr>
                ))}
                <tr>
                  <td className="muted">ledger</td>
                  <td className="mono">{status.ledger_path}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="card">
            <h2>Backend capabilities</h2>
            <div className="caps">
              {Object.entries(status.capabilities).map(([k, v]) => (
                <div key={k} className={v === false || v === "none" ? "cap off" : "cap on"}>
                  <span>{k}</span>
                  <span className="mono">{Array.isArray(v) ? v.join(", ") : String(v)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {status && (
        <div className="card">
          <div className="row">
            <h2 className="grow">Tensors ({tensors.length})</h2>
            <input placeholder="filter" value={filter} onChange={(e) => setFilter(e.target.value)} />
          </div>
          <table className="table">
            <thead>
              <tr>
                <th>name</th>
                <th>shape</th>
                <th>dtype</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((t) => (
                <tr key={t.name} className={t.name === status.info.lm_head_tensor ? "hl" : ""}>
                  <td className="mono">{t.name}</td>
                  <td className="mono">[{t.shape.join(", ")}]</td>
                  <td className="mono">{t.dtype}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
