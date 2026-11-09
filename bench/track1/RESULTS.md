# Track 1 baseline benchmark: Joey's MacBook, 9 Nov 2026

Machine: MacBook Pro, Apple silicon, 128 GB unified memory. Model: Qwen/Qwen3.5-9B @ `c2022362`, bf16, chat template with thinking off, greedy. Snapshots from `weight-space/qwen-harness/weights`. Joey's four resident servers were not running during the test, and memory was 93% free at the start.

Controls: the first run used made-up wording ("What is 8 × 7?", "A square has side length 4. What is its area?"). The joe-review found the real wording in `qwen_harness/trial_resident.py` (`SCORE_EXTRA`: "What is 8 times 7?", "A square has side 4. What is its area?"), and the scripts now use it. The fidelity table below comes from the re-run with the real controls.

## Speed

| Engine | Version | Single-stream tok/s (see note) | 100 short prompts | Notes |
|---|---|---|---|---|
| mlx-lm (Python) | mlx 0.32.4, mlx-lm 0.32.2 | ~30–31 | **39.5 prompts/s** (`batch_generate`) | peak memory 20.0 GB |
| mistral.rs | 0.9.5 prebuilt Metal binary | ~32 | 26.0 prompts/s (100 concurrent HTTP requests) | paged KV cache, prefix caching |
| candle | PR #3396 branch (`e9e4929f`) + `--features metal` | **0.35** | not run | on Metal but CPU-bound: sampled stacks show the time in buffer alloc/free and kernel launches (unfused GDN recurrence) |

**Timing note:** the two speed columns aren't like-for-like. The mlx figure is decode-only (timed after prefill, in the hand-written loop in `mlx_baseline.py`, which also includes top-64 trace collection for the first 50 steps). The mistral.rs figure is end-to-end HTTP: completion tokens / request wall time, including prefill and HTTP. So mistral.rs's true decode rate is, if anything, understated. The batched numbers differ too: mlx runs one in-process `batch_generate`, while mistral.rs gets 100 concurrent HTTP requests with no batch warmup (re-run: 24.6–25.5 prompts/s).

## Fidelity: do Joey's kept edits reproduce? (max 256 tokens, real controls; re-run 9 Nov after review)

Probes are the five-probe character total. Controls are characters for 8 times 7 / square side 4.

| Snapshot | Session record | mlx stream | mlx batch | mistral.rs |
|---|---|---|---|---|
| original | 1056 · square n/a | **1056** · 155 / 239 | 1056 · 155 / 239 | **1056** · 128 / 239 |
| edit-brief | 998 · square 189 | **998** · 20 / 189 | 994 · 102 / 189 | **998** · 102 / 239 |
| edit-gcd | 833 | **833** · 20 / 189 | 829 · 100 / 189 | **833** · 100 / 189 |
| edit-template | 727 · square 189 | **727** · 20 / 189 | 727 · 100 / 189 | **727** · 100 / 189 |

- **Precision caveat (checked after the run):** the edited snapshots store `lm_head.weight` as **F32** (4.07 GB), the original as BF16; every other tensor is unchanged. mlx ran the edited snapshots with an F32 `lm_head`, while mistral.rs `--dtype bf16` cast it to bf16. So on the edited rows, mlx vs mistral.rs differences mix precision with engine behaviour. The original-weights row is like for like (and its controls already differ, 155 vs 128). The edited snapshots' `model.safetensors.index.json` `total_size` is also a stale copy of the original's.
- **The probes reproduce on every engine and snapshot** (mlx stream and mistral.rs).
- **The controls don't.** On the original weights, 8 times 7 is 155 chars on mlx and 128 on mistral.rs. After edit-brief, the square drops to 189 on mlx (matching the session) but stays 239 on mistral.rs until edit-gcd. And 8 times 7 drops to 20 chars on mlx stream after edit-brief but 100–102 elsewhere (whether those 20 chars still contain 56 was not captured).
- So the controls, which are exactly the prompts that sit near an edit's hidden state, are where engines and decode paths disagree.

The candle example's chat template differs (it adds `/no_think` and the model still opens `<think>`), so its outputs aren't comparable until we template ourselves.

## Findings

1. **mistral.rs reproduces all four snapshots on the probes, and matches mlx on single-stream speed** (~32 vs ~31 tok/s, different timers). Its batched throughput is ~34% lower (26 vs 39.5 prompts/s, HTTP and concurrency untuned), and its control outputs differ from mlx even on the original weights. It's the strongest Rust option today for Qwen3.5 on Metal, but not a drop-in for mlx's numbers.
2. **candle's Qwen3.5 PR is about 85× slower than mlx on Metal** for 9B. The model runs, but the Gated DeltaNet recurrence is unfused, so the GPU waits on CPU-side kernel launches. That's exactly the kernel an engine of our own would need to write.
3. **Kept edits are fragile to the numeric path.** In the same framework on the same weights, `batch_generate` gives 994/829 where `stream_generate` gives 998/833. On the 8×7 control, edit-brief gives 102 chars via batch and 20 via stream (table above). A hand-written mlx decode loop (`mlx_baseline.py`, which caps at 512 tokens rather than 256) gave rectangle 402 instead of 369 for edit-brief. Near-tie edits flip on numeric-path differences (reduction order, and on the batch path also padding and masking; not isolated). **Consequence:** the parity harness and the keep rule must pin the exact decode path, and an edit should be re-validated on every backend and decode path it will ship on. A "robustness margin" (how far past the tie the edit pushes) belongs in the ledger.
4. The kept edit-brief changes exactly **one row** of `lm_head` (token 248046), with max |Δ| = 0.019. That confirms the rank-1 / LoRA-delta representation.

## Not yet measured

- An A/A run (the same decode path twice) to measure the noise floor.
- A committed candle script (the 0.35 tok/s row came from the example binary: `cargo +stable build --release --example qwen --features metal`, run with `--model 3.5-9b --weight-path <snapshot> --repeat-penalty 1.0 --sample-len 8`).
- Whether mistral.rs honours top-level `enable_thinking=False` (outputs show no `<think>`, so probably).

- Logit parity numbers across engines (mlx top-64 traces for 50 steps per prompt are saved on the Mac in `~/code/weightless-bench/out/`). mistral.rs logprobs weren't collected.
- LoRA-delta application of edit-brief on mistral.rs (does `lm_head` accept an adapter?).
- mlx-rs. It should behave like mlx-lm, but we haven't confirmed that.

## Reproduce

Scripts are in this folder. They expect the weight-space checkout at `~/code/weight-space` and write JSON reports to `~/code/weightless-bench/out/` (`mlx_paths.py` was changed to save its completions after the run in the table; that run only printed lengths). mistral.rs used the v0.9.5 `mistralrs-metal-aarch64-apple-darwin` release binary (`mistralrs serve --port 18800 -m <snapshot> --dtype bf16`). Building from source needs the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`), which isn't installed on the Mac.
