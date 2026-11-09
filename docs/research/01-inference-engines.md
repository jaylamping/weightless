# Track 1: Rust GPU inference engines (first pass)

*Status: desk research plus a Mac benchmark, 9 Nov 2026. Part 1 is the original desk research. Benchmark results and the updated recommendation are in "Benchmark results" below and in [RESULTS.md](../../bench/track1/RESULTS.md).*

## What the engine has to do

Weightless is not a chat server. Running prompts fast matters, but these five capabilities matter more, and most inference engines don't optimize for them:

1. **Hidden-state access.** Read the residual stream at any layer and step (trace view, cosine gate, rank-1 edits use `h`).
2. **Full logits per step.** Top-k, gap to runner-up, and per-token logits, not just a sampled token.
3. **In-place weight mutation on the GPU.** Apply and revert an edit to one tensor without reloading, at the dtype the model runs in.
4. **Batched trials on one loaded model.** Many candidate edits scored in one pass (per-row logit corrections for `lm_head`, multi-adapter style for deeper tensors). Joey's measurements showed extra copies are slower.
5. **Architecture coverage and size range.** Qwen3.5 first (hybrid Gated DeltaNet + attention), then Llama, Gemma, Mistral and Phi families, dense and MoE. No hard size cap, so quantized bases matter (track 2).

Cross-platform matters too: Metal (Mac), CUDA (Windows/Linux), and ideally Vulkan or ROCm for AMD.

## Important finding: Qwen3.5 is a hybrid model

Qwen3.5-9B is not a plain transformer. It has 32 layers in 8 blocks of (3 × Gated DeltaNet → FFN → 1 × gated attention → FFN). So 24 layers are linear attention with a fixed-size recurrent state, and only 8 have a normal KV cache. Hidden 4096, FFN 12288, 16 Q / 4 KV heads, head dim 256, vocab 248,320, plus one multi-token-prediction head. ([apxml](https://apxml.com/models/qwen35-9b), [HF blog](https://huggingface.co/blog/mlabonne/qwen35))

Consequences:

- Any engine needs a **Gated DeltaNet scan kernel**. Without a fused kernel it runs, but slowly. This is the main reason Qwen3.5 support is uneven across Rust engines.
- Hidden states for the 24 DeltaNet layers include a recurrent state, not just a residual. Editing anything above `lm_head` gets more complicated than in a plain transformer.
- Joey's kept edits were all on `lm_head`, so v1 only needs the final hidden state and logits. That works the same on any architecture.

## Candidates

### candle (Hugging Face)

- Pure Rust tensor library with CPU, CUDA and Metal backends. You write the model's forward pass yourself, in Rust.
- **Fit for editing: best.** Because we own the forward pass, capturing hidden states, returning full logits, swapping a weight tensor, and adding per-row logit corrections are all ordinary code.
- **Qwen3.5 status: not merged.** Model support is in open PR [#3396](https://github.com/huggingface/candle/pull/3396) (dense 0.8B–27B, BF16 and GGUF). It's tested on Metal only with 0.8B, with reported repetition problems that a side PR may fix. A chunked GatedDeltaNet scan is in a separate open PR, [#3461](https://github.com/huggingface/candle/pull/3461), last active 31 Jul 2026. That PR mentions no fused Metal or CUDA kernel. The tracking issue is [#3393](https://github.com/huggingface/candle/issues/3393).
- Community numbers on an RTX 4090 (unverified): 0.8B BF16 about 73 tok/s, 4B Q4_K_M about 44 tok/s. On an M4 Pro, 0.8B Q4 ran at 63–133 tok/s after two dtype fixes.
- One downstream project calls candle on Metal a "known-broken dead end" ([finch #924](https://github.com/darwin-finch/finch/issues/924)). I couldn't load the page to see why, so this needs checking.

### mistral.rs

- An inference server built on candle, with broad model support, in-situ quantization (ISQ), and Metal and CUDA backends. Recent releases claim strong CUDA speed. ([v0.8.2 coverage](https://aiweekly.co/node/2280))
- **Qwen3.5: supported** for 27B, 35B-A3B and 397B-A17B as vision-language models ([docs](https://ericlbuehler.github.io/mistral.rs/QWEN3_5.html)). The docs don't list 9B, but the 27B architecture is the same.
- **Fit for editing: medium.** It's a serving engine, so hidden states and weight mutation mean working against its internals. It is still the best place to **borrow working candle model code**, including the GDN kernels, rather than writing them ourselves.

### mlx-rs (Rust bindings to Apple MLX)

- v0.34.0 (4 Nov 2026), wrapping MLX 0.32.2. Unofficial, maintained by two owners. It ships only Mistral and MNIST examples. ([docs.rs](https://docs.rs/mlx-rs/latest))
- **Mac only.** It's the same runtime Joey already proved on Qwen3.5-9B via mlx-lm, so performance on a Mac is a known quantity.
- **Fit for editing: good.** It has array-level control like candle. We'd have to port the Qwen3.5 model code from Python mlx-lm to Rust ourselves.
- Best used as a Mac performance backend behind the same engine interface, or as the reference to check candle's numbers against.

### burn (Tracel)

- A full Rust deep-learning framework. Its CubeCL backend targets CUDA, ROCm, Metal, Vulkan and wgpu from one kernel language. v0.22 shipped 6 Oct 2026. Burn-LM, an LLM engine, is alpha. ([Tracel blog](https://tracel.ai/blog/?product=burn))
- **Fit for editing: good in principle**, and it has the widest hardware reach, including AMD and Vulkan.
- I found no Qwen3.5 implementation, and LLM inference is the least mature of the options. Worth a closer look as a long-term portability path, not the first prototype.

### llama.cpp through Rust bindings (llama-cpp-2, llama-cpp-4)

- **Widest architecture and quantization coverage**, with Qwen3.5 GDN on Metal and CUDA, and it's the reference most projects test against.
- **Fit for editing: poor.** Weights are GGUF, often quantized, and mmapped. Hidden states are reachable only through eval callbacks or tensor capture ([llama-cpp-4 tensor_capture](https://docs.rs/llama-cpp-4/0.2.24/llama_cpp_4/context/tensor_capture)). Changing a weight in place means poking ggml backend buffers.
- Useful as a **parity oracle** (do our logits match?) and possibly for running large quantized bases, but not as the editing engine.

### Others noted

- **rlx-qwen35**: a Rust Qwen3.5/3.6 implementation with GGUF, MTP and MoE. Some GPU paths still run the GDN recurrence on the CPU. Too early. ([docs.rs](https://docs.rs/crate/rlx-qwen35))
- **tch-rs** (libtorch) and **ort** (ONNX Runtime): mature, but libtorch is heavy to ship in a desktop app, and ONNX graphs make weight mutation and hidden-state taps awkward. Not shortlisted. Track 9 can revisit them if packaging changes the picture.

## Comparison

| | candle | mistral.rs | mlx-rs | burn | llama.cpp |
|---|---|---|---|---|---|
| Hidden states / full logits | yes (we own it) | internals | yes | yes | callbacks |
| In-place weight edits | yes | internals | yes | yes | awkward |
| Batched trials | we build it | has batching | we build it | we build it | has batching |
| Qwen3.5 today | open PRs | yes | port from mlx-lm | no | yes |
| Metal / CUDA / AMD | ✓ / ✓ / – | ✓ / ✓ / – | ✓ / – / – | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ |
| Desktop packaging | easy | easy | Mac only | easy | C++ build |

## Benchmark results (9 Nov 2026)

See [bench/track1/RESULTS.md](../../bench/track1/RESULTS.md). In short: on single-stream generation mistral.rs is in the same range as mlx-lm (~32 vs ~31 tok/s, though the two timers measure different work; see RESULTS), but its batched throughput is ~34% lower (26 vs 39.5 prompts/s, HTTP and concurrency untuned). It reproduces all four snapshots on the probes, but its control outputs differ from mlx (see RESULTS). candle's Qwen3.5 PR runs at 0.35 tok/s on Metal because the GDN scan is unfused. Kept edits flip between decode paths in the same framework, so the parity harness must pin the decode path.

## Recommendation (updated after benchmark)

The primary backend should build on **mistral.rs's model code and kernels** (candle-based, so the `Backend` contract below is unchanged), not candle's in-tree Qwen3.5. mlx is the speed and fidelity baseline. Plain candle stays the place where we write our own kernels, the fused GDN scan first.

### Original provisional recommendation

1. **Build the engine on candle**, behind our own `Backend` trait (load, forward with taps, logits, apply/revert, batch trials). Use mistral.rs and the open candle PRs as the source for Qwen3.5 and GDN model code instead of writing it from scratch.
2. **Keep mlx-rs as a second backend option for Macs**, chosen only if candle on Metal is clearly slower than mlx-lm on Joey's machine.
3. **Use llama.cpp as a parity and reference oracle**, and revisit it in track 2 for very large quantized bases.
4. **Watch burn** for AMD and Vulkan reach later.

*Superseded by the benchmark: candle's Qwen3.5 PR was not usable on Metal (0.35 tok/s).*

## Benchmark spike on Joey's Mac (done 9 Nov; open items: logit parity, apply/revert timing, LoRA on `lm_head`)

This has to run on Joey's MacBook (128 GB), since that's where the baseline lives. Measure, for Qwen3.5-9B at the pinned revision in bf16, thinking off, with the same 5 probes and 2 controls:

- tok/s and prefill time: mlx-lm (baseline) vs candle (PR #3396 branch) vs mistral.rs
- logits parity with mlx-lm at each probe's first 50 steps (max abs diff, top-1 agreement)
- time to apply and revert one rank-1 `lm_head` edit in place
- throughput for 100 short prompts in one process (Joey's baseline: 38.4 prompts/s)
- whether candle reproduces edit-brief exactly (same completion lengths: rectangle 427 → 369)

## Open questions

- Why does finch call candle on Metal a dead end? (Page didn't load.)
- Has candle PR #3396 been merged since July? Was the repetition bug fixed?
- Does mistral.rs expose a supported hook for hidden states or custom logits processors that we could use instead of forking?
- MoE edits: are router and expert tensors in scope? That's for track 3.

---

# Part 2: Universal engine (decision 9 Nov 2026)

Joey decided the goal is a **universal engine**: every viable runtime inside one Weightless API, with backends shipped in waves. This part drafts the contract and checks what each backend can support for each hook.

## Key finding: edits as low-rank deltas

A rank-1 edit `W += u vᵀ` is exactly a **rank-1 LoRA adapter** (A = vᵀ, B = u, scale 1). Most engines already support LoRA, so representing every Weightless edit as a low-rank delta, not a mutated tensor, gives us:

- **Portability.** A kept edit exports as a standard LoRA (safetensors for PEFT/MLX/mistral.rs, GGUF for llama.cpp). Users can run their edited model anywhere, not just in Weightless.
- **Concurrent workers for free on some backends.** mistral.rs keeps several adapters loaded on one base model and serves requests with different adapters side by side, with runtime load, replace and unload ([docs](https://ericlbuehler.github.io/mistral.rs/guides/customize/lora-adapters/)). That's the "1000 workers sharing one model" pattern, as long as limits allow (default 16 adapters, rank 256, configurable).
- **Cheap lineage.** A snapshot becomes base hash + an ordered list of small deltas, instead of a 19 GB copy per kept edit. Merging into a full checkpoint becomes an explicit export step (track 2).
- **Quantized bases.** The delta stays in full precision while the base can be quantized. That's the "90B on a laptop" requirement.

Caveats to verify:

- llama.cpp sets adapters per **context**, not per sequence, through `llama_set_adapters_lora(ctx, adapters, n, scales)`. Adapters must be loaded before the context is created, but they can be switched and rescaled without reloading ([guide](https://instagit.com/ggml-org/llama.cpp/how-to-use-lora-adapters-with-gguf-models)). For batched trials it needs one context per worker group, or our own patch.
- Whether each engine lets LoRA target `lm_head` (the output projection) is unconfirmed. Many LoRA loaders only map attention and MLP projections.
- mistral.rs says some GGUF architectures, such as Llama and Mistral 3, can't take adapters.
- Exploration still needs a fast path. For `lm_head` rank-1 trials, the effect is just `logits[k] += (δ/‖h‖²)(h·h_t)` per candidate (only the edited token's logit moves), computed on top of a shared forward pass. No adapter machinery needed.

## Draft adapter contract

```rust
/// What a backend can do. The UI and agent tools read this and hide what isn't supported.
pub struct Capabilities {
    pub run: bool,              // generate text
    pub full_logits: bool,      // full vocab logits per step
    pub hidden_taps: TapLevel,  // None | FinalOnly | EveryLayer
    pub weight_read: bool,      // read any named tensor
    pub delta_edits: DeltaSupport, // None | LmHeadOnly | AnyLinear
    pub per_request_deltas: bool,  // different edits per concurrent request
    pub batched_trials: bool,   // many candidate edits scored in one pass
    pub dtypes: Vec<DType>,     // bf16, f16, f32, q4_k, ...
    pub devices: Vec<Device>,   // Metal, CUDA, ROCm, Vulkan, CPU
}

/// One request. Deltas are activated per request, so concurrent workers can each carry different edits
/// without mutating shared state. `apply`/`revert` below are for the single-user, in-place path.
pub struct TraceRequest {
    pub prompt: Prompt,              // messages + chat-template options (e.g. thinking off)
    pub deltas: Vec<DeltaHandle>,    // per-request edits, empty = base model
    pub decode: DecodeSpec,          // greedy/sampled, max tokens, decode path (batch vs stream), seed
    pub taps: TapSpec,               // which layers/positions to return hidden states for
    pub top_k: usize,                // logits returned per step
}

pub trait Backend {
    fn capabilities(&self) -> &Capabilities;
    fn load(&mut self, model: ModelRef /* hf id + revision or local path */) -> Result<ModelInfo>; // ModelInfo includes tied_embeddings
    fn tensor_index(&self) -> Vec<TensorInfo>;              // names, shapes, dtypes
    fn read_tensor(&self, name: &str, slice: Slice) -> Result<HostTensor>;
    fn trace(&self, req: TraceRequest) -> Result<Trace>;     // generation + tokens, top-k logits, gaps, taps (covers `run`)
    fn apply(&mut self, delta: Delta) -> Result<DeltaHandle>; // low-rank delta on a named tensor
    fn revert(&mut self, h: DeltaHandle) -> Result<()>;
    fn score_trials(&self, base: TraceRequest, trials: &[Delta]) -> Result<Vec<TrialScore>>;
    fn export(&self, deltas: &[DeltaHandle], fmt: ExportFormat) -> Result<PathBuf>;
}
```

Everything above this trait (gates, keep rule, ledger, lineage, the agent API) is backend-agnostic. A **parity harness** runs the same probes and the same delta on every backend, and records logit and top-1 agreement in the ledger, so rows from different backends stay comparable.

## Capability matrix (expected, unverified)

| Hook | candle (own code) | mistral.rs | llama.cpp | MLX (mlx-rs) | burn |
|---|---|---|---|---|---|
| run | ✓ | ✓ | ✓ | ✓ | ✓ |
| full_logits | ✓ | likely | ✓ | ✓ | ✓ |
| hidden_taps | EveryLayer | patch needed | EveryLayer via eval callback | EveryLayer | EveryLayer |
| weight_read | ✓ | patch | ✓ (ggml tensors, may be quantized) | ✓ | ✓ |
| delta_edits | AnyLinear | AnyLinear via LoRA (lm_head?) | LoRA (lm_head?) | AnyLinear | AnyLinear |
| per_request_deltas | we build | ✓ | per context | we build | we build |
| batched_trials | we build | partial | – | we build | we build |
| Qwen3.5 | open PRs | ✓ | ✓ | port from mlx-lm | – |

## Wave order (proposed)

1. **Primary (full feature):** candle-based, reusing mistral.rs model code. Every hook, Metal and CUDA.
2. **Breadth:** llama.cpp adapter for very large and quantized models and the widest architecture list. Run, trace and LoRA deltas.
3. **Mac speed:** MLX adapter. The benchmark shows mlx still leads on batched throughput (39.5 vs 26 prompts/s).
4. **Reach:** burn for AMD, Vulkan and anything without CUDA or Metal.

## Updated next steps

- ~~Benchmark spike on Joey's Mac~~ (done). Still open: does a rank-1 LoRA on `lm_head` reproduce edit-brief on each engine that accepts it?
- Read mistral.rs internals for a hidden-state and logits hook. Upstreaming one may be cheaper than forking.
- Check llama.cpp's eval-callback cost at every layer, and whether `output.weight` accepts LoRA.

## Baseline vs our own engine (decision 9 Nov 2026)

candle and MLX are the **baseline**. Our own engine is developed alongside them, with its own kernels where it pays off, such as the GDN scan, batched `lm_head` trials and fused taps, and it must match or beat them. The benchmark is therefore a **permanent scoreboard**, not a one-time spike. It uses the same model, probe suite, controls and reference edit (edit-brief), and every engine gets scored on:

- prefill and decode tok/s, and prompts/s for batched short prompts
- logit parity vs the reference (max abs diff, top-1 agreement over the first 50 steps per probe)
- edit fidelity (does the reference delta reproduce the recorded completion lengths?)
- trials/s for batched candidate edits
- peak memory
