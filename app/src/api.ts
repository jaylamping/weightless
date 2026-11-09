import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { ExploreOptions, Prompt, Snapshot, Spec, Status, TensorInfo, Trace, TrialRecord } from "./types";

export type DeviceChoice = "auto" | "cpu" | "cuda" | "metal";

export const api = {
  loadModel: (path: string, device: DeviceChoice, dtype: string | null) =>
    invoke<Status>("load_model", { path, options: { device, dtype } }),
  status: () => invoke<Status | null>("status"),
  tensors: () => invoke<TensorInfo[]>("tensors"),
  setSpec: (spec: Spec) => invoke<void>("set_spec", { spec }),
  setDialRequest: (userWords: string, request: unknown) =>
    invoke<string>("set_dial_request", { userWords, request }),
  baseline: () => invoke<Trace[]>("baseline"),
  tracePrompt: (prompt: Prompt, maxNewTokens: number, topK: number) =>
    invoke<Trace>("trace_prompt", { prompt, maxNewTokens, topK }),
  tryNudge: (probeId: string, step: number, target: number, margin: number) =>
    invoke<TrialRecord>("try_nudge", { probeId, step, target, margin }),
  explore: (options: ExploreOptions) => invoke<TrialRecord[]>("explore", { options }),
  keep: (trialId: string) => invoke<Snapshot>("keep", { trialId }),
  unkeep: (trialId: string) => invoke<Snapshot>("unkeep", { trialId }),
  trials: (limit: number, allSessions: boolean) => invoke<TrialRecord[]>("trials", { limit, allSessions }),
  exportLedger: (path: string, vectors: boolean) => invoke<number>("export_ledger", { path, vectors }),
  onTrial: (cb: (t: TrialRecord) => void): Promise<UnlistenFn> => listen<TrialRecord>("trial", (e) => cb(e.payload)),
};
