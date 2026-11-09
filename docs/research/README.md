# Research phase

Weightless is a Tauri (Rust + React) desktop app for editing open-weight model weights directly, with no gradient training. A user's chosen agent (frontier provider, local model, or router) drives the work through a small built-in model, and an engine gates every edit before it is kept.

Background:

- Nusrat & Nusrat, *Weight-space programming for auditing clinical decision networks*, arXiv:2611.05341 (3 Nov 2026). Code: <https://github.com/hnusrat1/weightspace-rt>
- Joey's Qwen3.5-9B concision campaign (8 Nov 2026), done with mlx-lm in [weight-space](https://github.com/jaylamping/weight-space). Key lessons: rank-1 `lm_head` edits only at near-ties, a per-prompt keep rule with controls, one resident model, in-memory apply and revert, and a ledger that includes rejects.

## Gaps

See [GAPS.md](GAPS.md): joe-review fixes, plus what this research track didn't cover yet.

## Tracks

| # | Track | Status |
|---|---|---|
| 1 | Rust GPU inference engines | [report](01-inference-engines.md), [Mac benchmark](../../bench/track1/RESULTS.md) |
| 2 | Model formats, size limits (no hard 10B cap), edit deltas vs snapshots | [report](02-formats-sizes-deltas.md) |
| 3 | Weight-editing methods (ROME/MEMIT, rank-1, steering, model diffing, paper code) | not started |
| 4 | Interpretability for the "see everything" view (logit lens, SAEs, attribution) | not started |
| 5 | Concurrent workers on one loaded model (batched trials, multi-LoRA serving) | not started |
| 6 | Testing and A/B suite: tiered gates, regression sets, paired A/B of snapshots, robustness across backends and decode paths, LLM-judge | [state-of-the-art report](06-testing-ab.md) |
| 7 | Agent harness and providers (MCP, subscription OAuth, local models, routers, admission test) | not started |
| 8 | Built-in navigator model (base model, ledger schema, training) | not started |
| 9 | Desktop architecture (Tauri v2, in-process vs sidecar, IPC, packaging) | not started |
| 10 | Visualization, someday (wgpu vs WebGPU, LOD tensor rendering) | not started |
| 11 | Fast inference with weights in system RAM (Strata, NInfer, MoE expert offload, memory planner) | [report](11-ram-offload-inference.md) |
