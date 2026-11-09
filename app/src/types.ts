// Mirrors the serde types in crates/weightless-core.

export type TapLevel = "none" | "final_only" | "every_layer";
export type DeltaSupport = "none" | "lm_head_only" | "any_linear";

export interface Capabilities {
  run: boolean;
  full_logits: boolean;
  hidden_taps: TapLevel;
  weight_read: boolean;
  delta_edits: DeltaSupport;
  per_request_deltas: boolean;
  batched_trials: boolean;
  dtypes: string[];
  devices: string[];
}

export interface ModelInfo {
  id: string;
  path: string;
  architecture: string;
  vocab_size: number;
  hidden_size: number;
  num_layers: number;
  tied_embeddings: boolean;
  lm_head_tensor: string;
  fingerprint: string;
  backend: string;
  backend_version: string;
  dtype: string;
  device: string;
  quantization: string;
}

export interface TensorInfo {
  name: string;
  shape: number[];
  dtype: string;
}

export interface Prompt {
  text: string;
  format: { kind: "raw" | "chat" };
  thinking: boolean;
}

export interface Probe {
  id: string;
  role: "probe" | "control";
  prompt: Prompt;
  required_answer?: string | null;
}

export interface Spec {
  goal: string;
  objective: "shorter" | "controls_only";
  probes: Probe[];
  max_new_tokens: number;
  top_k: number;
  max_abs_delta: number;
}

export interface TokenLogit {
  id: number;
  text: string;
  logit: number;
}

export interface Step {
  token_id: number;
  text: string;
  top: TokenLogit[];
  gap: number;
  hidden_norm: number;
}

export interface Trace {
  prompt_tokens: number;
  steps: Step[];
  text: string;
  stop: "eos" | "max_tokens";
  wall_ms: number;
}

export interface ProbeResult {
  probe_id: string;
  role: "probe" | "control";
  base_len: number;
  new_len: number | null;
  first_divergence: number | null;
  tokens_changed: number | null;
  identical: boolean | null;
  answer_ok: boolean | null;
  base_has_answer?: boolean | null;
  predicted_first_flip: number | null;
  new_text: string | null;
}

export interface GateResult {
  name: string;
  pass: boolean;
  metrics: unknown;
}

export interface Verdict {
  keep: boolean;
  reasons: string[];
  gates: GateResult[];
  stage: "preflight" | "decode";
}

export interface NudgeOrigin {
  probe_id: string;
  step: number;
  from_token: number;
  from_text: string;
  target_token: number;
  target_text: string;
  gap_before: number;
  margin: number;
  hidden_norm: number;
}

export interface SiteFeatures {
  gap_before: number | null;
  hidden_norm: number | null;
  max_cos_controls: number;
  max_cos_other_probes: number;
  delta_max_abs: number;
}

export interface TrialRecord {
  id: string;
  session_id: string;
  decision_id: string;
  request_id: string | null;
  snapshot_id: string | null;
  created_at: string;
  proposer: string;
  rationale: string | null;
  delta: { id: string; tensor: string; rank: number; rows: number[]; scale: number; max_abs: number; origin: NudgeOrigin | null };
  site: SiteFeatures;
  results: ProbeResult[];
  verdict: Verdict;
  wall_ms: number;
  kept: boolean;
}

export interface Snapshot {
  id: string | null;
  trials: string[];
}

export interface Status {
  info: ModelInfo;
  capabilities: Capabilities;
  session_id: string;
  spec: Spec;
  snapshot: Snapshot;
  trials: number;
  passed: number;
  kept: number;
  ledger_path: string;
}

export interface ExploreOptions {
  budget: number;
  max_gap: number;
  margin: number;
  auto_keep: boolean;
}
