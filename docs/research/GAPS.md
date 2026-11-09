# Research gaps and open questions (after the joe-review, 9 Nov 2026)

On 9 Nov a joe-review ran over PR #1 with six reviewers (sol, muse, grok46-fast, argon, deepseek, opus) and an Opus judge, using the `adversarial` and `comments` lenses. Those lenses are built for code, so the judge's 13 confirmed findings were mostly about accuracy: overclaims, unlike-for-like timing, made-up control prompts and a wrong formula. They're fixed in this PR. Section 2 is a separate pass for things the research track **didn't look at**, which no code lens would catch.

## 1. Fixed from the review

| # | Finding | Fix |
|---|---|---|
| J1 | `mrs_client.py` wrote to a relative `out/` that nothing created | absolute `OUT` + `mkdir` in both scripts, per-snapshot filenames |
| J2 | the speed column mixed mlx decode-only with mistral.rs end-to-end HTTP | relabelled, with a timing note |
| J3 | "mistral.rs matches mlx" ignored its ~34% lower batched throughput | claim scoped |
| J4 | "reproduces Joey's kept edits" when edit-gcd wasn't run | **re-ran:** all four snapshots reproduce on the probes |
| J5 | the controls were made up even though the real wording was in `trial_resident.py` | **re-ran with the real controls:** they reproduce the session's square 189 on mlx, but **mistral.rs and mlx disagree on the controls** (new finding, in RESULTS) |
| J6 | stale "nothing benchmarked" status lines | updated |
| J7 | `logits += …` added a scalar to every logit (a no-op) | `logits[k] += …` |
| J8 | "ε = 2.3 means 90% top-1" holds only against one competitor | defined against the log-sum-exp of the competitors |
| J9, J11 | trace collection inside the mlx decode timer, and a 512 vs 256 token cap | documented in RESULTS |
| J10 | the cached-hidden-state shortcut assumes `lm_head` isn't tied to the embeddings | precondition stated, `tied_embeddings` added to `ModelInfo` |
| J12 | `Engine` vs `Backend` naming | unified on `Backend` |
| J13 | dead code | removed |
| J14 | apply/revert are global, so per-request deltas weren't possible | `TraceRequest` defined with per-request `deltas` and decode path |

## 2. What the research track didn't look at

Ranked by how much each could change a decision.

1. **Licensing and provider terms. Could block features.**
   - **Model licenses.** Llama, Gemma and others carry use policies, and some restrict redistributing modified weights or certain uses. Exporting edited weights or LoRAs needs a per-license check.
   - **Subscription OAuth.** Consumer subscriptions (Claude, ChatGPT and others) may restrict use inside a third-party harness. T3 Code's approach was assumed and never checked.
   - **Engine licenses.** We believe mistral.rs, candle, llama.cpp and MLX are MIT or Apache, and Heretic is AGPL-3.0, which would matter if we borrowed from it. None of this was verified.
2. **Responsible use.** Direct weight editing makes it easy to strip a model's safety training ("abliteration"). Before shipping we need a position on refusal and safety regressions: detect them, warn, or block. The track 6 gate has a safety slice, but no policy decision exists yet.
3. **Prior art and competitors were not researched.** There's no landscape of what already exists or how Weightless differs:
   - interpretability: TransformerLens, nnsight, Neuronpedia, Goodfire
   - editing: EasyEdit, ROME/MEMIT tooling, mergekit
   - abliteration: Heretic
   - weight-reading models: Kabra et al., *Weight oracles* ([2610.07334](https://arxiv.org/abs/2610.07334)), the closest prior art to the navigator model
   - Takehi et al.'s weight-space programming post, and the paper's own code ([weightspace-rt](https://github.com/hnusrat1/weightspace-rt))
4. **Only one model, one edit type and one platform were tested.** Everything rests on Qwen3.5-9B, rank-1 `lm_head` edits and an Apple-silicon Mac.
   - **Windows and CUDA weren't benchmarked**, even though Joey-PC is available.
   - **AMD was never touched.**
   - **The cached-hidden-state shortcut** only covers `lm_head`. Deeper-layer edits, MoE router and expert tensors, and tied-embedding models all fall back to decoding.
5. **There's no noise floor.** No A/A run (the same decode path twice) was done, so the batch vs stream differences aren't yet separated from run-to-run noise.
6. **Thinking mode was always off.** Real users will run thinking models. Edits validated with thinking off may not hold with it on, and traces get much longer.
7. **Export path.** llama.cpp's LoRA converter rejects embedding adapters, and its comments say it ignores `lm_head` ones (seen in source, not fully confirmed). So Joey's style of edit may not export to GGUF as a LoRA. A merged-weights export might be needed instead. That's track 2.
8. **Resource footprint of multiple backends.** mistral.rs grabbed ~17 GB for its KV cache by default. Running several backends in one process, plus a navigator model, needs a memory budget design.
9. **Unverified sources.** Several 2026 papers in track 6 were read from abstracts only, the sample-size table is our own calculation, and the ε thresholds are suggestions.

## 3. Suggested additions to the plan

- **Track 0, licensing and policy:** model licenses, provider terms for subscription OAuth, engine licenses, and the safety-regression policy. Do this before track 7.
- **Landscape and prior art:** add to track 3.
- **Windows/CUDA benchmark on Joey-PC:** add to track 1.
- **A/A noise floor and a thinking-on run:** add to the track 6 scoreboard.

## 4. Joey's decisions (9 Nov 2026)

1. **Licensing:** respect licenses where practical. It isn't a blocker, so there's no track 0 gate.
2. **Safety stance:** none for now. It's a personal project with a single user.
3. **Competitors:** not needed as market research (personal, non-commercial). Prior art is still worth mining for techniques in track 3.
4. **Platforms:** proper testing on both **Mac and Windows** (Joey-PC) is required. A Windows/CUDA benchmark is added to track 1.
5. **A/A noise floor and thinking-on runs:** added to the track 6 scoreboard.
