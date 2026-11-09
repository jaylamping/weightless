# Track 1 baseline benchmark: Joey's MacBook, 9 Nov 2026

Machine: MacBook Pro, Apple silicon, 128 GB unified memory. Model: Qwen/Qwen3.5-9B @ `c2022362`, bf16, chat template with thinking off, greedy. Snapshots from `weight-space/qwen-harness/weights`. Joey's four resident servers were not running during the test, and memory was 93% free at the start.

Controls used here: "What is 8 × 7?" and "A square has side length 4. What is its area?". The session's exact control wording wasn't on disk, so square-control lengths aren't comparable to the 189 from the handoff doc.

## Speed

| Engine | Version | Decode tok/s (single) | 100 short prompts | Notes |
|---|---|---|---|---|
| mlx-lm (Python) | mlx 0.32.4, mlx-lm 0.32.2 | ~30–31 | **39.5 prompts/s** (`batch_generate`) | peak memory 20.0 GB |
| mistral.rs | 0.9.5 prebuilt Metal binary | ~32 | 26.0 prompts/s (100 concurrent HTTP requests) | paged KV cache, prefix caching |
| candle | PR #3396 branch (`e9e4929f`) + `--features metal` | **0.35** | not run | on Metal but CPU-bound: sampled stacks show the time in buffer alloc/free and kernel launches (unfused GDN recurrence) |

## Fidelity: do Joey's kept edits reproduce? (five-probe chars, max 256 tokens)

| Snapshot | Session record | mlx `stream_generate` | mlx `batch_generate` | mistral.rs |
|---|---|---|---|---|
| original | 1056 | **1056** | 1056 | **1056** (427, 623 match) |
| edit-brief | 998 | **998** | 994 | **998** |
| edit-gcd | 833 | **833** | 829 | not run |
| edit-template | 727 | **727** | 727 | **727** |

The candle example's chat template differs (it adds `/no_think` and the model still opens `<think>`), so its outputs aren't comparable until we template ourselves.

## Findings

1. **mistral.rs matches mlx on both speed and outputs**, including reproducing edit-brief and edit-template exactly from the saved snapshots. It's the strongest Rust option today for Qwen3.5 on Metal.
2. **candle's Qwen3.5 PR is about 85× slower than mlx on Metal** for 9B. The model runs, but the Gated DeltaNet recurrence is unfused, so the GPU waits on CPU-side kernel launches. That's exactly the kernel an engine of our own would need to write.
3. **Kept edits are fragile to the numeric path.** In the same framework on the same weights, `batch_generate` gives 994/829 where `stream_generate` gives 998/833. On the square control, edit-brief gives 377 chars via batch and 187 via stream. A hand-written mlx decode loop gave rectangle 402 instead of 369 for edit-brief. Near-tie edits flip on bf16 reduction-order differences. **Consequence:** the parity harness and the keep rule must pin the exact decode path, and an edit should be re-validated on every backend and decode path it will ship on. A "robustness margin" (how far past the tie the edit pushes) belongs in the ledger.
4. The kept edit-brief changes exactly **one row** of `lm_head` (token 248046), with max |Δ| = 0.019. That confirms the rank-1 / LoRA-delta representation.

## Not yet measured

- Logit parity numbers across engines (mlx top-64 traces for 50 steps per prompt are saved on the Mac in `~/code/weightless-bench/out/`). mistral.rs logprobs weren't collected.
- LoRA-delta application of edit-brief on mistral.rs (does `lm_head` accept an adapter?).
- mlx-rs. It should behave like mlx-lm, but we haven't confirmed that.

## Reproduce

Scripts are in this folder. They expect the weight-space checkout at `~/code/weight-space` and write to `~/code/weightless-bench/out/`. mistral.rs used the v0.9.5 `mistralrs-metal-aarch64-apple-darwin` release binary (`mistralrs serve --port 18800 -m <snapshot> --dtype bf16`). Building from source needs the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`), which isn't installed on the Mac.
