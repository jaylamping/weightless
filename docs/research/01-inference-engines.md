# Track 1: Rust GPU inference engines (first pass)

*Status: desk research, 9 Nov 2026. No code has been benchmarked yet; see "Next step".*

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

## Recommendation (provisional)

1. **Build the engine on candle**, behind our own `Engine` trait (load, forward with taps, logits, apply/revert, batch trials). Use mistral.rs and the open candle PRs as the source for Qwen3.5 and GDN model code instead of writing it from scratch.
2. **Keep mlx-rs as a second backend option for Macs**, chosen only if candle on Metal is clearly slower than mlx-lm on Joey's machine.
3. **Use llama.cpp as a parity and reference oracle**, and revisit it in track 2 for very large quantized bases.
4. **Watch burn** for AMD and Vulkan reach later.

This holds only if the benchmark below shows candle on Metal is usable for Qwen3.5-9B.

## Next step: benchmark spike on Joey's Mac

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
