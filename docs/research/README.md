# Research phase

Weightless is a Tauri (Rust + React) desktop app for editing open-weight model weights directly, with no gradient training. A user's chosen agent (frontier provider, local model, or router) drives the work through a small built-in model, and an engine gates every edit before it is kept.

Background:

- Nusrat & Nusrat, *Weight-space programming for auditing clinical decision networks*, arXiv:2611.05341 (3 Nov 2026). Code: <https://github.com/hnusrat1/weightspace-rt>
- Joey's Qwen3.5-9B concision campaign (8 Nov 2026), done with mlx-lm in [weight-space](https://github.com/jaylamping/weight-space). Key lessons: rank-1 `lm_head` edits only at near-ties, a per-prompt keep rule with controls, one resident model, in-memory apply and revert, and a ledger that includes rejects.

## Plan changes adopted from track 11 (Joey, 9 Nov 2026)

From [the RAM-offload report](11-ram-offload-inference.md):

1. **Memory planner before every load.** Read free VRAM, RAM and the Windows commit limit, keep a reserve for other apps (SOLIDWORKS) and the OS, cap pinned memory, and re-check free VRAM per request. Warn with numbers when a model won't fit; never discover it by thrashing. On Windows, warn if the NVIDIA *Sysmem Fallback* policy isn't set to "Prefer No Sysmem Fallback" for our engine.
2. **Placement is part of the `Backend` contract** (`plan`, `PlacementPlan`, capability flags `offload_layers`, `offload_experts`, `gpu_expert_cache`, `elastic_vram`, `gate_mode`). See [track 1](01-inference-engines.md).
3. **Dense models:** quantized body plus full-precision edited tensors, all on the GPU (Qwen3.5-9B ≈ 6 GB on Joey-PC). Layer offload only as a last resort, and only for batched scoring.
4. **MoE models:** expert offload with a GPU expert cache. llama.cpp's `-ot`/`--moe-cache-mib` first (wave 2); build it into the primary backend later, reading Strata and llama.cpp (both MIT).
5. **Edits stay compute-time overlays** and never get written into cached, offloaded or quantized copies. Edited tensors are pinned to the GPU.
6. **Gate mode:** every keep/reject decision runs with fixed placement, no adaptive cache swaps, a fixed kernel path, no speculative drafting and the deployment KV dtype. The A/A noise floor is measured in this mode (track 6).
7. **Batch trials into large passes** so offload cost is paid once per pass (track 5).
8. **Elastic VRAM and idle unload,** so the engine can give memory back when Joey opens another GPU app.
9. **Later, chat speed only:** MTP speculative decoding, prompt lookup and big prompt chunks.
10. **Router traces** (which experts fire per token) are recorded in the trace and ledger as navigator training data (track 8).

## Gaps

See [GAPS.md](GAPS.md): joe-review fixes, plus what this research track didn't cover yet.

## Tracks

| # | Track | Status |
|---|---|---|
| 1 | Rust GPU inference engines | [report](01-inference-engines.md), [Mac benchmark](../../bench/track1/RESULTS.md) |
| 2 | Model formats, size limits (no hard 10B cap), edit deltas vs snapshots | [report](02-formats-sizes-deltas.md) |
| 3 | Weight-editing methods (ROME/MEMIT, rank-1, steering, model diffing, paper code) | not started |
| 4 | Interpretability for the "see everything" view (logit lens, SAEs, attribution) | not started |
| 5 | Concurrent workers on one loaded model (batched trials, multi-LoRA serving; batch to amortize offload, per track 11) | not started |
| 6 | Testing and A/B suite: tiered gates, regression sets, paired A/B of snapshots, robustness across backends and decode paths, LLM-judge | [state-of-the-art report](06-testing-ab.md) |
| 7 | Agent harness and providers (MCP, subscription OAuth, local models, routers, admission test) | not started |
| 8 | Built-in navigator model (base model, ledger schema, training) | not started |
| 9 | Desktop architecture (Tauri v2, in-process vs sidecar, IPC, packaging) | not started |
| 10 | Visualization, someday (wgpu vs WebGPU, LOD tensor rendering) | not started |
| 11 | Fast inference with weights in system RAM (Strata, NInfer, MoE expert offload, memory planner) | [report](11-ram-offload-inference.md) |
