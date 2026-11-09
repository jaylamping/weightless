# Track 11: Fast inference with weights in system RAM

*Status: desk research, 9 Nov 2026. Nothing was run on Joey's machines. Speed figures are the projects' own numbers unless stated, and none were reproduced by us.*

Joey asked how tools like "Strata" and "Infer" run very large models quickly on a gaming GPU by offloading to system RAM, and what Weightless could borrow. This report answers that, then maps each technique onto what Weightless needs: hidden-state access, in-place edits, near-tie logit gaps, and many batched trials.

## TL;DR

- **Strata is real** ([Niko1221/Strata](https://github.com/Niko1221/Strata), MIT, C++/CUDA/HIP/Metal engine built partly on ggml, Tauri desktop app). It runs Qwen3.8-Flash-Next, marketed as "125B", at about **87 tok/s on a 12 GB RTX 5070 with 64 GB RAM** (maintainer's numbers).
- **"Infer" is most likely [NInfer](https://github.com/Neroued/ninfer)**, a C++/CUDA engine that does CPU expert offload on an RTX 5090, and which Strata credits for ideas. It could also be **[PowerInfer](https://github.com/Tiiny-AI/PowerInfer)**, the older hot/cold-neuron engine. Both are covered below. This is a guess from the name.
- **The trick is not that RAM is fast.** It's that these models are **sparse mixture-of-experts (MoE)**. Only ~10 of 24,576 experts run per token in Flash-Next, so each token touches a few GB of weights, not 125B parameters. The engines then:
  1. keep everything that runs on **every** token (attention, routers, shared experts, output head, KV cache) on the GPU,
  2. fill the rest of VRAM with a **cache of the most-used experts**, learned from routing statistics,
  3. let the **CPU compute the cache misses in place** in pinned RAM, in parallel with the GPU, instead of copying them over PCIe,
  4. use very low-bit quantization (2 to 3 bits per weight),
  5. stack **speculative decoding** on top (the model's own MTP head drafts 3 tokens, about 2.4 to 3.2 are accepted per pass), worth about 1.6 to 1.8×,
  6. read prompts in **huge chunks** (8,192 tokens), so each expert crosses PCIe once per chunk rather than once per token.
- **None of this makes a dense bf16 model fast.** Qwen3.5-9B bf16 on Joey-PC is bound by physics: every token reads all 18 GB of weights, so the 12 to 13 GB that don't fit in the ~6 GB of free VRAM cap decode at roughly 4 to 6 tok/s whether the CPU computes them or PCIe streams them. The fix for the dense case is a **quantized body plus full-precision edited tensors**, which fits in VRAM (§5).
- **Weightless's workload is unusually offload-friendly.** Scoring many trials is batched, prefill-like work. Offload cost is paid per forward pass, not per sequence, so a batch of 64 trials pays the PCIe or CPU cost once. Strata's own prompt reading runs at 1,000 to 2,000 tok/s with experts streaming.
- **Likely cause of the Joey-PC lockup:** on Windows, the NVIDIA driver's *Sysmem Fallback* silently spills VRAM overflow into shared system memory instead of failing, and pinned (page-locked) RAM can starve the OS. Strata documents both. Our engine needs a memory planner with hard budgets before it runs anything big on that machine (§6).

## 1. What the tools actually are

### Strata

| | |
|---|---|
| Repo | [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata), MIT. Engine 0.1.58, app 0.2.12 at time of writing. |
| Shape | C++/CUDA/HIP/SYCL/Metal engine in `src/`, a Python server (OpenAI- and Anthropic-compatible API), a **Tauri** desktop app plus Node host. Same overall shape as Weightless. |
| Models | Qwen3.8-Flash-Next (MoE, 24,576 experts, 10 routed per token, 48 layers, MTP head), Qwen3.6-35B-A3B, and DeepSeek V4.1-Flash through a separate engine. **MoE only.** |
| Hardware | NVIDIA RTX 20 to 50, AMD RDNA3/4, Intel Arc (Linux), Apple Silicon, Strix Halo, DGX Spark |
| Built on | Parts of llama.cpp/ggml (MIT). Credits ideas to [Splash](https://github.com/incoai/splash), [NInfer](https://github.com/Neroued/ninfer) and [HyperQwen](https://github.com/syv-ai/HyperQwen). |

Maintainer's measurements ([DETAILS.md](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)), RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, Windows, MTP on:

| Pack | Decode @1K | Decode @128K | Prompt @32K |
|---|---:|---:|---:|
| Q2_0 | 87.3 tok/s | 73.7 tok/s | 2,171 tok/s |
| IQ3_S | 52.4 tok/s | 45.5 tok/s | 1,624 tok/s |

An independent comparison on an RTX 3060 12 GB ([InsiderLLM](https://insiderllm.com/guides/strata-vs-llama-cpp-rtx-3060/)) found Strata 6 to 7× faster than stock llama.cpp at reading prompts. Two llama.cpp flags (`-b 4096 -ub 4096`) closed about half of that gap. With drafting off, Strata decoded about 2.3 to 2.5× faster than llama.cpp (about 20 vs 8 to 9 tok/s). Drafting added another 43 to 55%. Speed only, no quality measurement.

### NInfer (probably the "Infer" Joey means)

[Neroued/ninfer](https://github.com/Neroued/ninfer): native C++/CUDA for specific models on an RTX 5090 and DGX Spark. CUDA Graph decode, MTP/DFlash speculative decoding, shared-prefix cache, **CPU expert offload** for Qwen3.6-35B-A3B and Qwen3.8-Flash-Next on the 5090. Uses its own `.ninfer` artifact format with mixed NVFP4/Q8/NF4 weights. Uses `liburing` (Linux async I/O) and `hwloc` (CPU topology), so it is Linux-only and tuned to two specific machines. License not checked.

### PowerInfer (the other "Infer")

[Tiiny-AI/PowerInfer](https://github.com/Tiiny-AI/PowerInfer), MIT, 2023 to 2025. It applies the same "hot on GPU, cold on CPU" idea to **individual neurons of dense models**. A small predictor guesses which FFN neurons will fire, and only those are computed. Up to 11× faster than llama.cpp on Falcon-40B. **Only works on ReLU-family models** (ReLU, ReGLU, squared ReLU), which excludes Qwen, Llama 3 and Mistral as shipped. Effectively superseded by real MoE models. Its successors (SmallThinker, PowerInfer-2) target phones.

### Related engines worth knowing

| Engine | Idea | Notes |
|---|---|---|
| **llama.cpp** `-ot` / `--override-tensor`, `--cpu-moe`, `--n-cpu-moe N` | Regex-place tensors per device. The standard recipe is `-ngl 999 -ot "exps=CPU"`: everything on GPU except routed experts. ([guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)) | Static placement. MIT. Our wave-2 backend. |
| **llama.cpp** `--moe-cache-mib N` (merged 7 Oct 2026, PR #29887) | A GPU LRU cache for experts that live in RAM. For batches of 32 tokens or fewer, hits run on GPU and misses upload over PCIe. | Author's numbers: RTX 4090 Flash-Next Q4_0 25.0 → 40.7 tok/s. One user reported a regression on GLM 5.3 Flash. ([summary](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)) |
| **KTransformers** (Tsinghua, Apache-2.0) | Attention and KV on GPU; routed experts on CPU with AMX/AVX-512 INT4/INT8 kernels; NUMA-aware. Runs 671B DeepSeek on one 24 GB GPU plus ~512 GB RAM. ([overview](https://www.noze.it/en/insights/ktransformers-hybrid-cpu-gpu-inference/)) | Shines on Xeon servers with AMX. A 14900K has AVX2 but **no AVX-512 or AMX**, so the fast kernels don't apply to Joey-PC. |
| **LLM in a Flash** (Apple, [2312.11514](https://arxiv.org/abs/2312.11514)) | Dense model streamed from SSD; predict sparse FFN activations, reuse recently loaded neurons ("windowing"), and store rows and columns together so reads are large ("row-column bundling"). | Relevant to Macs with less RAM than the model. Needs activation sparsity. |
| **FlexGen**, **AirLLM** | Stream layers from RAM or disk one at a time. FlexGen schedules big batches so each layer's load is shared by many sequences. | Very slow for one chat, but the batching insight is exactly Weightless's trial-scoring case. |
| **mistral.rs** (our primary base) | Per-GPU layer mapping (`-n "0:44;1:20"`); when the model doesn't fit, automatic mapping puts the **remaining whole layers on the CPU**, "much slower", and disables paged attention. ([docs](https://ericlbuehler.github.io/mistral.rs/guides/perf/distributed-inference/)) | No expert-level offload or GPU expert cache found in its docs. A per-tensor "topology" file exists but I couldn't read its format. |

## 2. Why it's fast: the bandwidth arithmetic

Decoding one token is memory-bound: the time is roughly the bytes of weights read divided by the bandwidth they're read at. Rough figures for Joey-PC (not measured):

| Path | Bandwidth |
|---|---|
| RTX 5080 VRAM (GDDR7) | ~960 GB/s |
| DDR5 dual-channel, read by the i9-14900K | ~70 to 90 GB/s |
| PCIe 5.0 x16, RAM → GPU | ~64 GB/s theoretical, ~50 GB/s in practice |
| NVMe SSD | ~5 to 12 GB/s |

**Dense Qwen3.5-9B, bf16, ~6 GB VRAM free:** about 18 GB per token, 6 GB from VRAM and 12 GB from RAM.

- CPU computes the RAM part: 12 GB / ~80 GB/s ≈ 150 ms → **~6 tok/s ceiling**
- Stream it over PCIe: 12 GB / ~50 GB/s ≈ 240 ms → **~4 tok/s ceiling**
- All in VRAM (if it fit): 18 GB / 960 GB/s ≈ 19 ms → ~50 tok/s ceiling

**MoE Flash-Next at ~2.5 bits:** about 6B active parameters ≈ 2 GB per token. Most of it is attention, shared weights and cached experts already in VRAM. Only the cache misses, a few hundred MB, hit the CPU. Then MTP multiplies by about 1.7.

So "offloading to RAM" works when **most of what each token touches is already on the GPU and what isn't is small**. Sparsity (MoE) makes the second part true, a learned cache makes the first part true, and low-bit quantization shrinks both. For a dense model, every token touches every weight, so the RAM share always runs at RAM speed.

## 3. Technique catalogue

| # | Technique | What it does | Who does it | Helps dense? | Helps MoE? |
|---|---|---|---|---|---|
| T1 | **Hot path pinned to GPU** | Attention, norms, routers, shared experts, embeddings/output head and KV stay on the GPU always | All of them | Yes | Yes |
| T2 | **Expert offload, CPU computes in place** | Routed experts live in pinned RAM; CPU runs the ones not on the GPU, in parallel with GPU work | Strata, KTransformers, llama.cpp `-ot`, NInfer | No | **Big** |
| T3 | **Adaptive GPU expert cache** | Free VRAM fills with the most-routed experts. Strata learns a per-user profile and saves it across restarts; llama.cpp uses LRU | Strata, llama.cpp `--moe-cache-mib` | No | **Big** |
| T4 | **Routing look-ahead prefetch** | Predict the next layer's experts from the current hidden state and start their copy early | Strata (`STRATA_LOOKAHEAD`), DAOP/DALI papers | No | Medium |
| T5 | **Pinned memory + async copy streams** | `cudaHostAlloc` / `cudaHostRegister` so the GPU can DMA without staging; copies overlap compute on a separate stream | Everyone | Yes (streaming) | Yes |
| T6 | **Big prompt chunks / ring buffer** | Each offloaded weight crosses PCIe once per chunk of 4K to 8K tokens; next layer streams while this layer computes | Strata (`--prefill`), llama.cpp `-b/-ub` | **Yes** | Yes |
| T7 | **Layer streaming** | Whole layers copied in turn, double-buffered | FlexGen, AirLLM, mistral.rs CPU fallback | Only for big batches | Rarely |
| T8 | **Low-bit weights** | 2 to 4 bit (IQ2/IQ3/Q4, NVFP4), often mixed per tensor | Everyone | Yes | Yes |
| T9 | **KV cache quantization** | int8 KV, or 4-bit V with a Hadamard rotation; frees VRAM for experts | Strata (`--kv int8/k8v4/q4_0`) | Yes | Yes |
| T10 | **KV streaming** | Long context's KV in RAM, only the part attention reads stays resident (~13.7 KB RAM per token for Flash-Next) | Strata | Yes (long ctx) | Yes |
| T11 | **Speculative decoding** | MTP head or prompt lookup drafts tokens, one pass verifies them | Strata, NInfer, Splash (DFlash2) | Yes | Yes |
| T12 | **Prefix / conversation cache** | Follow-ups read only the new tokens | Strata, NInfer, KTransformers | Yes | Yes |
| T13 | **Elastic VRAM** | Shrink the expert cache on demand so another app can have VRAM; experts fall back to CPU | Strata (`POST /v1/vram`, `--idle-unload`) | n/a | Yes |
| T14 | **Activation-sparse neurons** | Predict which dense FFN neurons fire; skip the rest | PowerInfer, LLM in a Flash | ReLU-family only | n/a |

## 4. How each technique interacts with editing

Weightless isn't a chat server. The engine has to expose hidden states and full logits, apply and revert edits without reloading, and decide **near-tie edits where a logit gap under ~2 decides keep or reject** (track 1, track 6). That last point makes numerical stability matter more here than in any chat engine.

**Hidden-state access: easy.** In every hybrid design the residual stream comes back to the GPU after each layer, because attention runs there. Capturing `h` at any layer costs the same as without offload. MoE engines also expose **router choices per token**: Strata has `--dump-routing`. That's free extra trace data for the navigator model (which experts fire on the prompts an edit affects).

**Where the edited tensor lives: needs a rule.** With offload, a tensor can exist as a RAM master, a GPU cached copy, and a quantized form all at once. Applying an edit to one copy and not the others is a silent bug. Rule: **edits never mutate cached copies. They are applied as low-rank overlays at compute time** (track 2 already chose f32 row/factor overlays). For `lm_head` that's `logits += (h·v) u` on whichever device holds `h`. For an offloaded expert, the CPU kernel and the GPU kernel each need the overlay path, or edited experts get pinned to GPU. Edited tensors should be marked **pinned-to-GPU, full-precision** in the placement plan.

**Determinism: the big risk.** Strata documents that its CPU kernels round single-token and multi-token expert groups differently, so drafting, cache state, or a resumed conversation can change a greedy answer. Its adaptive cache swaps are "not bit-exact from run to run". Their fix is a reproducibility mode: same kernels for every group, no adaptive swaps, no PCIe split (`STRATA_IQ_MT_MIN=1 --prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`). Track 1 already saw Joey's kept edits flip between mlx's batch and stream decode paths. So **gating must run in a pinned "gate mode"** (fixed placement, no adaptive swaps, fixed kernel choice, drafting off or proven equivalent), and the A/A noise floor from GAPS.md must be measured in that mode. Chat can use the fast, adaptive mode.

**Quantized base: already decided.** Edits stay as f32 overlays and never merge into quantized blocks (track 2). Offload doesn't change that. It makes it more important, since offload-friendly models are the low-bit ones.

**KV quantization: changes logits.** An int8 or 4-bit KV cache perturbs logits by about as much as bf16 merge rounding. The gate must run with the same KV setting the user will run, or stay at bf16 KV for gating.

**Speculative decoding: fine for output, watch the drafts.** Drafts are verified by the target model, so greedy output is unchanged up to the numerical issue above. If an edit touches a tensor the MTP head shares (it may share `lm_head` or embeddings; check per model), drafts follow the edit and acceptance stays high. If it doesn't, drafts get rejected more often near edited tokens and it gets slower. Not wrong, just slower.

**Batched trials: the good news.** Offload cost is per forward pass, not per sequence. Scoring 64 trials teacher-forced on a probe set is one big prefill-like batch, so each offloaded weight crosses PCIe or gets read by the CPU once for all 64. This is FlexGen's insight and why Strata's prompt reading reaches 1,000 to 2,000 tok/s while decode is 50 to 90. **The "1000 workers" design should batch trials into big passes**, which is what makes large and offloaded models usable for Weightless at all.

## 5. What this means for Joey-PC (RTX 5080 16 GB, ~10 GB used by other apps, 64 GB RAM, i9-14900K)

**Dense Qwen3.5-9B (today's model).** Don't offload layers. Fit it instead:

| Plan | VRAM | Notes |
|---|---|---|
| bf16 everything | ~18 GB + KV | Doesn't fit the ~6 GB free |
| **Q4 body (~4 GB) + `lm_head` bf16 (2.03 GB) + f32 row overlays** | **~6 GB + KV** | Tight; body quantized, the tensor we edit kept at full precision. Track 2 rule, applied. The input embedding (another ~1B params, if untied) is a row lookup and can stay in RAM. |
| Q4 body on GPU, `lm_head` in RAM, logits on CPU | ~4 GB + KV | `lm_head` is 1.02B params. CPU GEMV ≈ 25 ms per token at bf16, amortized across a trial batch. Viable for scoring, slow for chat. |
| Q8 body | ~8 GB + `lm_head` | Only when SOLIDWORKS isn't running |

Quantizing the body changes `h`, which changes which edits are near-ties. An edit found on the Q4 body must be re-gated on whatever precision it ships at (track 6 already requires this).

**MoE models.** With ~6 GB of free VRAM and 64 GB RAM, Joey-PC is close to Strata's minimum tier (12 GB card, 64 GB RAM): Qwen3.6-35B-A3B should run comfortably and Flash-Next at a few bits should run, with fewer cached experts and so slower than the published numbers. The 14900K's 8 P-cores with AVX2 are the CPU expert compute; its E-cores should be avoided for those threads (Strata has thread-placement settings for this).

## 6. The lockup on Joey-PC and how not to repeat it

Not diagnosed, but Strata's docs describe two Windows behaviours that match "the machine locked up":

1. **Sysmem Fallback.** On Windows, when a CUDA allocation exceeds free VRAM, the NVIDIA driver by default spills the excess to shared system memory instead of failing. Startup looks normal and speed collapses. Fix: NVIDIA Control Panel → *CUDA – Sysmem Fallback Policy* → **Prefer No Sysmem Fallback**, per program, for our engine's `.exe`. ([Strata](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md), [explainer](https://runaihome.com/blog/shared-gpu-memory-slow-local-ai-sysmem-fallback-fix-2026/))
2. **Pinned-memory exhaustion.** Page-locked RAM can't be paged out. Pinning too much starves Windows itself. Strata keeps 640 MiB of pinnable memory free, caps pinned expert memory at ~29 GiB on Windows, and leaves ~10 GB of RAM for the system.

What the Weightless engine should do, regardless of which backend runs:

- **A memory planner before load.** Read free VRAM and available RAM and commit limit, subtract a reserve (other apps, OS headroom), then pick placement and quantization. If the model doesn't fit, say so with numbers (the "warn, don't block" rule) instead of discovering it by thrashing.
- **Re-check free VRAM per request**, since SOLIDWORKS can grab VRAM mid-session.
- **Cap pinned memory** and never map more of a model file than RAM minus headroom.
- **Elastic VRAM and idle unload** (Strata T13), so Joey can open SOLIDWORKS without killing the engine.
- On Windows, detect the sysmem-fallback setting and warn if it's on.

## 7. Recommendations

1. **Add placement and memory to the `Backend` contract** (track 1). Capability flags: `offload_layers`, `offload_experts`, `gpu_expert_cache`, `elastic_vram`, `deterministic_gate_mode`. A `PlacementPlan` names each tensor's home device, dtype and pinned/edited status. The planner in §6 produces it.
2. **Dense models: quantized body plus full-precision edited tensors, all on GPU.** Use layer offload only as a last resort, and only for batched scoring.
3. **MoE models: llama.cpp-style tensor overrides first, then an expert cache.** The wave-2 llama.cpp backend gives `-ot`, `--n-cpu-moe` and `--moe-cache-mib` almost free. For the primary mistral.rs-based backend, expert offload has to be built. Strata (MIT) and llama.cpp (MIT) are the code to read: CPU expert kernels, pinned arena, routing look-ahead, cache profile.
4. **Batch trials into large passes** so offload cost is shared. This belongs in track 5 (concurrent workers).
5. **Gate mode.** Fixed placement, no adaptive swaps, fixed kernel path, drafting off, and the same KV dtype as deployment. Measure the A/A noise floor in that mode. Add "placement mode" to the parity harness's pinned decode path.
6. **Later, for chat speed only:** MTP speculative decoding and prompt chunking. Not needed for editing.
7. **Skip:** PowerInfer-style neuron sparsity (ReLU-only), KTransformers' AMX kernels (no AMX on the 14900K), SSD streaming (too slow for interactive trials).

## Open questions and checks (not done, need Joey's OK for anything heavy)

- Does mistral.rs's topology file allow per-tensor CPU placement, for example experts only? Read the source.
- Does Qwen3.5/3.8's MTP head share `lm_head`? Determines whether edits carry into drafts.
- A light test on Joey-PC, **only after Joey clears it**: Qwen3.5-9B at Q4 body + bf16 `lm_head` through llama.cpp with `-ot "output=CUDA0"`, checking VRAM use and tok/s with SOLIDWORKS open. Optional: install Strata and run Qwen3.6-35B-A3B to see the MoE case first-hand.
- Is Joey-PC's Sysmem Fallback policy at the driver default? Worth checking before any more GPU runs.

## Sources

- Strata: [repo/README](https://github.com/Niko1221/Strata), [HOW_IT_WORKS](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md), [DETAILS](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md), [AGENTS.md](https://github.com/Niko1221/Strata/blob/main/AGENTS.md), [stratallm.org](https://stratallm.org/) (unofficial), [ithub.directory overview](https://www.ithub.directory/blog/strata-local-ai-inference-consumer-hardware)
- [InsiderLLM: Strata vs llama.cpp on an RTX 3060](https://insiderllm.com/guides/strata-vs-llama-cpp-rtx-3060/)
- [NInfer](https://github.com/Neroued/ninfer), [Splash](https://github.com/incoai/splash), [HyperQwen](https://github.com/syv-ai/HyperQwen)
- [PowerInfer](https://github.com/Tiiny-AI/PowerInfer), [paper 2312.12456](https://arxiv.org/abs/2312.12456)
- [llama.cpp MoE offload guide (HF blog)](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide), [--n-cpu-moe vs --moe-cache-mib](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)
- [KTransformers overview](https://www.noze.it/en/insights/ktransformers-hybrid-cpu-gpu-inference/), [KTransformers paper page](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/)
- [LLM in a Flash (Apple)](https://arxiv.org/abs/2312.11514)
- [mistral.rs distributed inference / device mapping](https://ericlbuehler.github.io/mistral.rs/guides/perf/distributed-inference/)
- MoE offload research: [DAOP 2501.10375](https://arxiv.org/abs/2501.10375), [DALI 2602.03495](https://arxiv.org/abs/2602.03495), [2512.16473](https://arxiv.org/abs/2512.16473)
- Windows sysmem fallback: [runaihome explainer](https://runaihome.com/blog/shared-gpu-memory-slow-local-ai-sysmem-fallback-fix-2026/)
