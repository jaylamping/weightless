# Research phase

Weightless is a Tauri (Rust + React) desktop app for editing open-weight model weights directly, with no gradient training. A user's chosen agent (frontier provider, local model, or router) drives the work through a small built-in model, and an engine gates every edit before it is kept.

Background:

- Nusrat & Nusrat, *Weight-space programming for auditing clinical decision networks*, arXiv:2611.05341 (3 Nov 2026). Code: <https://github.com/hnusrat1/weightspace-rt>
- Joey's Qwen3.5-9B concision campaign (8 Nov 2026), done with mlx-lm in [weight-space](https://github.com/jaylamping/weight-space). Key lessons: rank-1 `lm_head` edits only at near-ties, a per-prompt keep rule with controls, one resident model, in-memory apply and revert, and a ledger that includes rejects.

## Plan changes adopted from track 11 (Joey, 9 Nov 2026)

From [the RAM-offload report](11-ram-offload-inference.md):

1. **Memory planner before every load.** Read free VRAM, RAM and the Windows commit limit, keep a reserve for other apps (SOLIDWORKS) and the OS, cap pinned memory, and re-check free VRAM per request. Warn with numbers when a model won't fit; never discover it by thrashing. On Windows, warn if the NVIDIA *Sysmem Fallback* policy isn't set to "Prefer No Sysmem Fallback" for our engine.
2. **Placement is part of the `Backend` contract** (`plan` → `load(plan)`, `PlacementPlan`, `begin_gate`/`end_gate`, and `Capabilities` flags `offload_layers`, `offload_experts`, `gpu_expert_cache`, `elastic_vram`, `deterministic_gate_mode`, `top_logprobs`). See [track 1](01-inference-engines.md).
3. **Dense models:** quantized body plus full-precision edited tensors, all on the GPU. For Qwen3.5-9B that is about 7 GB with KV and overhead (see track 11 §5), so on Joey-PC it fits when the apps holding ~10 GB of VRAM are closed, not alongside them. Alongside them it needs a smaller body (3-bit or NVFP4) or a quantized `lm_head` with full-precision overlays of only the edited rows (track 2's rule); to be measured. Layer offload only as a last resort, and only for batched scoring.
4. **MoE models:** expert offload with a GPU expert cache. llama.cpp's `-ot`/`--moe-cache-mib` first (wave 2); build it into the primary backend later, reading Strata and llama.cpp (both MIT).
5. **While editing and gating, edits stay compute-time overlays** and never get written into cached, offloaded or quantized copies, and edited tensors are GPU-resident. Exports are different: they may merge or requantize, then re-gate the exported artifact (track 2 §4).
6. **Gate mode** (backend-wide state, not a per-request flag): every keep/reject decision runs with fixed placement, no adaptive cache swaps, a fixed kernel path, no speculative drafting and the deployment KV dtype. The A/A noise floor is measured in this mode (track 6).
7. **Batch trials into large passes** so offload cost is paid once per pass (track 5).
8. **Elastic VRAM and idle unload,** so the engine can give memory back when Joey opens another GPU app.
9. **Later, chat speed only:** MTP speculative decoding, prompt lookup and big prompt chunks.
10. **Router traces** (which experts fire per token) are recorded in the trace and ledger as navigator training data (track 8).

### Serving engines inside the app (Joey, 9 Nov 2026)

Joey wants fast serving engines like Strata in the app too, not just as research. Plan:

- **"Serve" mode.** After editing, run the edited model through a fast engine and expose it on a local OpenAI/Anthropic-compatible API, so Joey can chat with it or point coding agents at it. Editing and gating stay on the primary backend; serving engines are adapters that need `run` plus `top_logprobs`, but no taps or apply/revert. *(Proposed; Joey to confirm.)*
- **How edits get there:** export the deltas to the engine's format (merged safetensors, patched GGUF `output.weight`, or a LoRA where accepted, per track 2), then re-gate the exported artifact on that engine using its top-k logprobs, since quantization and kernels can flip near-tie edits. An engine without logprobs only gets generation-level checks (probe lengths, required answers), and the ledger marks its edits "not margin-checked".
- **Candidates:** llama.cpp (`llama-server`, already wave 2), Strata (MIT, GGUF, MoE models only: Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, and DeepSeek V4.1-Flash through a separate engine; logprob support unchecked), mistral.rs server, mlx-lm server on Mac. NInfer is Linux-only and 5090/Spark-tuned, so later.
- **Packaging:** run each engine as a managed sidecar process (download per GPU type, like Strata's app does), with the memory planner deciding whether it fits next to the editing engine or replaces it.
- **Scoreboard:** serving engines join the parity harness, so we can see whether an edit survives each one.

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
