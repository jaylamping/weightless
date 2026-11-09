# Track 8: The built-in navigator model

*Status: desk research and a plan, 9 Nov 2026. Nothing trained or benchmarked yet. Model facts come from model cards and release coverage; items marked (unverified) were not checked against the weights or run on our hardware. Memory and speed figures are our own estimates.*

## Summary

- **Division of labour (Joey, 9 Nov 2026):** a reasonably strong model is the **driver**. When Joey says "I wish this model wasn't so chatty", the driver breaks that into a precise **dial request** (what behaviour, which direction, on which prompts, what must not change). The navigator takes the dial request and knows **where and how to turn the dials**: which tensors, rows and positions, and by how much. The engine and gate then try and verify. **No distillation for now:** the navigator does not learn to imitate the driver. It learns from what happened to edits, meaning gate outcomes. Distilling the driver is a possible later phase.
- **Base model: Qwen3.5-2B** (Apache 2.0, March 2026), with **Qwen3.5-0.8B** as the lite tier and **Qwen3.5-4B** as the step-up if 2B misses the eval bar. The deciding reason is that Qwen3.5 is the architecture our primary engine already runs (track 1), so the navigator needs no extra runtime, and it shares a tokenizer and family with the most likely target models.
- **The navigator is three parts, not one model.** A tiny **scorer** (not an LLM) ranks thousands of candidate edits per second. The **navigator LM** turns the driver's dial request into concrete candidate edits and explains gate results. A later **activation reader** lets it read the target's hidden states directly.
- **The ledger is the training set, and the gate is the label.** Every candidate gets a verdict from the track 6 gate, so labels are free and objective. That also gives us a verifiable reward for RL later.
- **Bootstrap without users:** run automated edit campaigns on small target models with heuristic proposers to fill the ledger with millions of labeled candidates, import Joey's weight-space campaign, and add real sessions where the frontier driver works through the navigator.
- **Train on the Mac first** (128 GB, mlx-lm LoRA). Joey-PC only with Joey's go-ahead. Rent a cloud GPU for the RL stage if needed.
- **Ship the navigator as a LoRA on a stock base**, the same delta-on-base idea the app uses for edits.

## 1. What the navigator has to do

The flow, using Joey's example:

1. **Joey → driver:** "I wish this model wasn't so chatty."
2. **Driver → navigator:** a dial request (1a below), e.g. "shorten answers to simple factual and arithmetic questions; keep every required answer and the final number; don't change code or long-form tasks; probes: these 20 prompts; controls: these 8".
3. **Navigator → engine:** traces the probes, finds where the behaviour is decided (e.g. near-tie positions where an end token loses to another sentence, as in Joey's concision campaign), and proposes ranked candidate edits with sizes.
4. **Engine + gate:** run the trials and keep or reject. Results go back to the navigator to refine, and a summary goes to the driver.

The driver is the user's agent (frontier, local or a router). It reasons about intent; the navigator is the specialist for the target model's internals. Their jobs:

| Job | Input | Output | Volume |
|---|---|---|---|
| **Score** candidate edits before running them | candidate features (gap, cosines, predicted Δlogit on protect sets, tensor, layer, norms) | P(keep), expected margin, likely failure reason | thousands per second, to feed "1000+ workers" |
| **Propose** candidates | goal, trace (logits, gaps, top-k at each position), recent ledger | a list of edits (tensor, row, target token, δ) with a short rationale | tens per step |
| **Localize** a dial request | the driver's dial request, traces of its probes | where to edit (tensor, layer, rows, positions) and how (direction, size), as a ranked plan | one per request, revised as results come in |
| **Explain** results | gate output, ledger history | a short diagnosis ("spills onto control 8×7, cosine 0.987") | per verdict |
| **Summarize** state for the agent | session ledger | a compact status, so the frontier agent spends fewer tokens | per agent turn |

Two lessons from earlier work shape this:

- **Don't make it read raw weights.** The paper's own related work notes that reading raw weights failed at 270M parameters (Takehi et al.), and Nusrat & Nusrat only did it on MLPs with a few thousand parameters. The navigator reads **traces and gate features**, which the engine already computes.
- **The gate stays the authority.** The navigator ranks and proposes. It never decides what is kept. That keeps a weak or wrong navigator from damaging a model, and it means navigator quality can be measured as "how fast does the gate say yes".

### 1a. The dial request (driver → navigator)

This is the contract between the two, so it's typed, not free text, and every one is stored in the ledger next to the outcomes it led to.

```
dial_request {
  behaviour:   "verbosity" | "fact" | "format" | "tone" | "refusal" | "loop" | "custom",
  direction:   e.g. "shorter", "say X instead of Y", "use JSON",
  description: free text from the driver, for the record and for the navigator LM,
  probes[]:    prompts where the change must show, with what counts as success
               (required answers, max length, target string),
  controls[]:  prompts that must not change at all,
  protect[]:   wider sets that must stay close (KL / top-1 thresholds),
  budget:      max trials, max edits, max size per edit
}
```

The driver writes it (with Joey's words as context); the navigator can reply with typed questions back ("probes cover arithmetic only; include code?") before it starts.

## 2. Architecture: scorer, navigator LM, activation reader

**2a. Scorer (stage 0, no LLM).** A gradient-boosted tree model or a small MLP over numeric candidate features, predicting the gate verdict and margin. It trains in seconds, runs in microseconds per candidate on CPU, and can be retrained after every session. Joey's rules ("only near-ties under ~2 logits", "reject whole-row adds", "cosine above 0.98 to a control spills") are exactly the kind of thing it learns, and its feature importances show whether it learned them. This exists so that screening thousands of worker trials never waits on a language model.

The scorer is our own local **decision model**: typed questions in ("keep?", "which candidate?", "how risky for controls?"), probabilities out, no text. Hosted decision models like Jev work this way for general judging (track 6); ours is trained on gate labels, runs offline and costs nothing per call. The navigator LM can expose the same typed interface to the agent. Open decision models such as Kev, NanoJev and Nimble show how: a small Qwen-family LM plus a LoRA trained on answer tokens, or readout heads, gives calibrated typed answers. Tev1 and Mica are already built on Qwen3.5-4B. So stage 1 can add decision heads to the same Qwen3.5-2B navigator, and the scorer and navigator share one model when the numeric features alone aren't enough. Track 6 lists the current decision models.

**2b. Navigator LM (stage 1).** A small chat model with tool calling, fine-tuned on ledger-derived tasks (section 5). It speaks to the user's agent over the same MCP-style tool API the engine exposes (track 7), and it calls the scorer and the engine as tools itself. The agent can always bypass it and call the engine directly, which is also how we A/B it.

**2c. Activation reader (stage 3, research).** Activation Oracles ([Karvonen et al., 2512.15674](https://arxiv.org/abs/2512.15674), ICML 2026; [code](https://github.com/adamkarvonen/activation_oracles)) train a LoRA so a model answers questions about activations injected at placeholder tokens: the oracle adds a norm-matched copy of the target activation to the residual stream at layer 1 (`h' = h + ‖h‖·v/‖v‖`). They used Qwen3-8B, Gemma-2-9B and Llama-3.3-70B, about 1M training examples, and about 10 H100-hours for the 8B. Two ways to use this:

- **Navigator reads the target via a projector.** A learned linear map from the target's hidden size to the navigator's. Works across target families but needs training per target hidden size (or per family).
- **Target + navigator LoRA.** The Activation Oracle setup: the oracle is the target model itself with a LoRA. In our app the target is already loaded and the primary engine supports per-request adapters (track 1), so this costs almost no extra memory. The catch is that it needs a LoRA per base model.

We start with neither, and decide after stage 1 shows where the LM is blind.

## 3. Base model candidates

Requirements: an open license (Joey: respect where practical, not a blocker), good tool calling and structured output, long context for traces and ledgers, small enough to sit **next to** the target model, and fine-tunable on the Mac with mlx-lm. Running on our own engine is a big plus, since a second runtime means more code and more memory.

| Model | Size | License | Notes | Fit |
|---|---|---|---|---|
| **Qwen3.5-2B** | 2B dense | Apache 2.0 | Released 2 Mar 2026. 262K native context, built for tool calling, thinking mode. Hybrid Gated DeltaNet + attention, the same architecture as Qwen3.5-9B that mistral.rs and mlx-lm already ran in track 1 (small sizes untested on mistral.rs). | **Recommended default** |
| **Qwen3.5-0.8B** | 0.8B dense | Apache 2.0 | Same family. Lite tier for Joey-PC and for fast iteration. Also a good **target** model for synthetic campaigns. | **Lite tier** |
| **Qwen3.5-4B** | 4B dense | Apache 2.0 | Same family. Step-up if 2B fails the eval bar. | **Step-up** |
| Gemma 4 E2B / E4B | 2.3B / 4.5B effective (5.1B / 8B total with per-layer embeddings) | Apache 2.0 (first Gemma under Apache) | Released Apr 2026, 128K context, native function calling. Per-layer embeddings, 512-token sliding window and audio/vision towers would be new engine work. Memory is set by the total size, not the effective one. | Strong alternative for an A/B, not the default |
| SmolLM3-3B | 3B dense | Apache 2.0, open data | Llama-style, easy to run anywhere, 64K (128K with YaRN). Weaker at tool calling than the above (unverified). | Fallback |
| Granite 4.1 3B, Ministral 3 3B, Phi-4-mini 3.8B | 3–3.8B | Apache / Apache / MIT (per coverage, unverified) | All viable, none with a reason to beat Qwen3.5-2B for us. | Not shortlisted |
| Llama 3.2 1B/3B | 1–3B | Llama license | Sep 2024, outclassed, licence with a use policy. | No |

No Qwen3.6 or Qwen3.8 model exists below 27B as of Nov 2026 ([QwenLM/Qwen3.8](https://github.com/QwenLM/Qwen3.8)), so Qwen3.5 is the newest small Qwen.

**Why the Qwen3.5 family over Gemma 4.** Gemma 4 E2B may well be the stronger model per byte. But the Qwen3.5 small models reuse the architecture our primary engine is being built around, so the navigator ships on the engine we already have and improves when that engine does. A shared family with the most likely targets also helps the activation-reader stage, where hidden spaces of the same family are the easiest to bridge. The plan keeps Gemma 4 E2B as the A/B challenger in section 6.

### Memory budget (estimates)

| Navigator | bf16 weights | ~4.5-bit weights |
|---|---|---|
| Qwen3.5-0.8B | ~1.6 GB | ~0.5 GB |
| Qwen3.5-2B | ~4 GB | ~1.2 GB |
| Qwen3.5-4B | ~8 GB | ~2.3 GB |

- **Mac (128 GB):** any of these at bf16 next to a 9B bf16 target, with room to spare.
- **Joey-PC (RTX 5080, ~6 GB free):** the target model needs that VRAM. Run the navigator **on CPU** (i9-14900K, 64 GB RAM) at 4-bit: the 2B should give tens of tokens per second, which is enough for planning and explanation (unverified, to measure). The scorer is CPU-only anyway. A 0.8B on the GPU is the fallback if CPU latency is too high.
- The navigator's KV cache needs a cap. mistral.rs reserved ~17 GB of KV by default in track 1 (GAPS §2.8).

## 4. Ledger schema for training

Track 6 §5 lists per-trial fields and track 2 §4 defines the `Delta`. The navigator needs four more things on top: the **goal**, the **proposer**, the **agent conversation**, and **what happened later**. Proposed record types (JSONL, one file per session, schema-versioned; big vectors go to a sidecar safetensors file by hash):

```
session   { schema_version, session_id, target: { model_id, revision, base_hash, family, hidden_size, tied_embeddings },
            backend, dtype, quantization, decode_path, hardware,
            goal_text,                         // the user's/agent's goal in plain words
            spec: { prompts[], required_answers[], controls[], protect_sets[] } }

dial_req  { request_id, session_id, user_words, dial_request }   // section 1a; candidates link to it

trace     { trace_id, session_id, prompt_id, position,
            topk: [{ token, logit }], gap, chosen_token,
            h_ref,                              // sidecar hash of the hidden state (fp16), optional
            cos_to_protect: { max, argmax_prompt } }

candidate { candidate_id, session_id, request_id, parent_snapshot, trace_id,
            delta,                              // track 2 Delta, without gate fields
            proposer: "heuristic:<name>" | "random" | "agent:<model>" | "navigator:<version>" | "human",
            rationale_text,                     // free text from the proposer, if any
            features: { gap, delta, h_norm, pred_dlogit_protect_max, pred_dlogit_controls[], layer, tensor_kind, rank },
            scorer_pred: { p_keep, margin, version } }

verdict   { candidate_id, gate_steps: [{ step, pass, metrics }],   // track 6 gate, in order
            verdict: "keep" | "reject" | "revert",
            reason_codes[],                     // e.g. SPILL_CONTROL, MARGIN_LOW, PPL_COLLAPSE, BACKEND_FLIP
            cost: { trials, wall_ms } }

followup  { candidate_id, event: "reverted_later" | "stack_regression" | "export_flip" | "reconfirmed", detail }

turn      { session_id, turn, role: "agent" | "navigator" | "engine",
            content, tool_calls[], tool_results_ref }
```

Rules:

- **Log every reject.** Rejects are most of the signal (Joey's lesson from weight-space).
- **Log the scorer's and navigator's prediction before the gate runs,** so every row is also an eval of the navigator version that made it.
- **Record the decode path and backend on every row.** Track 1 found kept edits that flip between mlx batch and stream decoding; labels from different paths must not be mixed silently.
- **Store hidden states by reference and only for edit points and protect sets,** since a 4096-wide fp16 vector is 8 KB and a campaign has millions of positions.

## 5. Training tasks derived from the ledger

| # | Task | Model | Source rows | Label |
|---|---|---|---|---|
| T1 | Keep prediction | scorer (and LM, as text) | candidate + verdict | verdict, margin, reason codes |
| T2 | Candidate proposal | LM | trace + goal + recent ledger | kept candidates as targets; rejected ones as negatives for preference training |
| T3 | Diagnosis | LM | verdict + gate metrics | reason codes rendered as text from templates |
| T4 | Localization | LM (+ decision heads) | dial request + probe traces | the edit sites and sizes that the gate kept for that request; rejected sites as negatives |
| T5 | Summaries for the driver | LM | session ledger | templated from ledger fields |
| T6 | Activation QA (stage 3) | LM + reader | hidden-state refs | questions answerable from the ledger ("will this edit spill onto prompt X?") |

Format for the LM is the base model's own chat template with tool calls, so the fine-tune teaches content, not syntax. Long ledgers are summarized by the engine into a fixed "state block" so prompts stay under ~16K tokens even though the base handles 262K.

## 6. Bootstrapping before real ledger data

Real use will produce a few hundred candidates a day at best. That's not enough, so the first ledger is synthetic, but labeled by the real gate.

1. **Synthetic campaigns (the main source).** A headless runner loads small targets (Qwen3.5-0.8B and 2B first, then a Gemma, a Llama-style and an MoE model for diversity), picks goals from templates (concision, change a fact, change a format, persona words, stop a loop, change a refusal), and proposes candidates with heuristics: Joey's near-tie rank-1 rule, random rows, perturbed deltas, deliberately bad whole-row adds. The cached-hidden-state shortcut (track 6) scores an `lm_head` rank-1 candidate with a matrix multiply, so a Mac can label **millions** of candidates overnight. That covers T1, T2 and T3. Edits deeper than `lm_head` need real decoding and are slower, so they get a smaller, targeted share.
2. **Import Joey's weight-space campaign.** The Qwen3.5-9B concision run (8 Nov) has real keeps and rejects with Joey's keep rule. Convert it into the schema as a seed set and as a held-out sanity check ("does the scorer agree with Joey's verdicts?").
3. **Public editing datasets** for fact-edit goals: CounterFact and zsRE ([2202.05262](https://arxiv.org/abs/2202.05262)), MQuAKE for multi-hop ([2305.14795](https://arxiv.org/abs/2305.14795)).
4. **Dial requests from synthetic goals and the driver.** Synthetic campaigns generate dial requests from goal templates, so T4 has labels from day one. Real sessions add requests written by the frontier driver from Joey's own words. Either way the label is what the gate kept, not what the driver said. **No distillation for now (Joey, 9 Nov 2026):** we don't train the navigator to copy the driver's reasoning or tool calls. That stays a possible later phase.
5. **Only then, real sessions.** Every real session appends to the ledger, and the navigator is retrained on a schedule.

## 7. Fine-tuning method, in stages

| Stage | Method | Data | Where | Gate to move on |
|---|---|---|---|---|
| 0 | Train the **scorer** (LightGBM or a small MLP) | synthetic T1 | Mac or PC CPU, minutes | AUC and calibration beat Joey's hand rules on the held-out sets |
| 1 | **LoRA SFT** of Qwen3.5-2B (rank 16–64, attention + MLP), with decision heads | T2–T5 from synthetic campaigns and driver sessions | Mac, mlx-lm LoRA | tool-call validity ≥ 99%, beats the heuristic proposer (section 8) |
| 2 | **Preference training** (DPO) on kept vs rejected proposals for the same trace | T2 pairs | Mac | higher keep rate at equal trials |
| 3 | **Online RL** (GRPO) with the engine as the environment. Reward = gate verdict with margin, minus trials spent; full-tier checks on held-out goals guard against gaming the fast gate | live synthetic campaigns | rented GPU or Mac | trials-to-first-keep down vs stage 2 |
| 4 | **Activation reader** (section 2c) | T6 | Mac, or rented GPU | answers spill/locality questions better than the scorer alone |

Notes:

- **Why LoRA:** the adapter is tens of MB, the base stays a stock download, base upgrades mean retraining an adapter rather than a model, and it is the same "delta on a stock base" idea the app uses for edits. A full fine-tune of 0.8B is cheap enough on the Mac to try as a comparison.
- **Compute (estimates):** stage 1 on 2B with ~100K examples is hours on the Mac, not days. Stage 3 is where the cost is, because each rollout runs the engine. A rented H100 for a day or two is the likely path; Joey-PC only with Joey's go-ahead and when he isn't using it.
- **Thinking mode:** train with thinking off for scoring and tool calls (latency), and keep a thinking-on variant for planning. Both get evaluated (standing rule: every scoreboard has a thinking-on run).

## 8. Evaluation

**Offline, on held-out data.** Hold out whole target **families** and whole goal types, not random rows, so we measure generalization to new models.

| Metric | Measures |
|---|---|
| Scorer AUC, calibration (ECE), precision@k | Does P(keep) rank and mean what it says |
| Keep rate of the top-k proposals | Proposal quality |
| **Trials to first keep** and **wall-clock to first keep** | The headline number: how much faster the gate says yes |
| Tool-call validity, plan success on the task suite | Is it a usable interface |
| Explanation accuracy vs reason codes | Does it tell the truth about failures |
| Latency and memory on Mac and Joey-PC | Does it fit next to the target |

Baselines: random proposals, Joey's hand rules, the frontier agent alone, and Gemma 4 E2B with the same fine-tune (the A/B challenger from section 3).

**Online, in the app.** Paired A/B per task (track 6 §3): the same goal with the navigator on vs off. Report success rate, trials and wall-clock to the first keep, and **frontier tokens spent**, because cutting the agent's token bill is a large part of the navigator's value. An A/A run (navigator on vs on) sets the noise floor.

**Guardrails.** The gate decides keeps, so a bad navigator wastes time but can't damage the model. The RL stage can learn to game the fast gate, so held-out goals are checked at the medium and full tiers, and any rise in later reverts (`followup` rows) is a regression.

## 9. Integration

- **Serving:** the navigator runs on the primary engine (mistral.rs-based) in the same process as the target on the Mac. On Joey-PC it runs on CPU at 4-bit, through the same engine or llama.cpp if CPU speed is better there (to measure).
- **API:** the agent sees tools such as `navigator.plan(goal)`, `navigator.propose(trace_id, n)`, `navigator.score(candidates)`, `navigator.explain(candidate_id)` and `navigator.status()`, next to the raw engine tools. The navigator's own tool calls go through the same API, so its actions land in the ledger like anyone else's.
- **Versioning:** each navigator is `base + adapter + scorer`, with a version id stamped on every prediction row, so A/B between navigator versions is just a ledger query.

## 10. Prior art to mine (techniques only)

- **Activation Oracles** ([2512.15674](https://arxiv.org/abs/2512.15674)) and **LatentQA** ([2412.08686](https://arxiv.org/abs/2412.08686)): training models to answer questions about another model's activations. The basis for stage 4.
- **Kabra et al., "Weight oracles"** (2610.07334, listed in GAPS as the closest prior art): not found in a quick search; needs a proper read before stage 4 (unverified).
- **Nusrat & Nusrat** ([2611.05341](https://arxiv.org/abs/2611.05341)): test every edit against the spec before keeping it, verify in the run dtype, and distrust forced verdicts. All of that lives in the gate, which is why the gate and not the navigator owns keeps.
- **Editing methods** (ROME/MEMIT/AlphaEdit, track 3): their locality and stacking results define what "a good proposal" means for edits deeper than `lm_head`.

## 11. Decisions for Joey

1. **Base model:** Qwen3.5-2B as default, with 0.8B lite and 4B step-up (recommended), or Gemma 4 E2B as the default instead.
2. **Driver vs navigator (decided by Joey, 9 Nov 2026):** a strong model is the driver and writes dial requests; the navigator localizes and sizes edits and learns only from gate outcomes. No distillation of the driver for now.
3. **Training hardware:** Mac for stages 0–2 (recommended), a rented GPU for stage 3, and Joey-PC only when cleared.

## 12. Next steps

1. Fix the ledger schema with tracks 2 and 6, and write it into the engine's first milestone so the very first trials are training data.
2. Build the headless synthetic-campaign runner on Qwen3.5-0.8B with the cached-hidden-state scorer.
3. Convert Joey's weight-space campaign to the schema.
4. Train the stage 0 scorer and compare it with Joey's hand rules.
5. Measure Qwen3.5-2B and Gemma 4 E2B speed and memory on both machines (CPU on Joey-PC, with his go-ahead).

## Sources

- Qwen3.5 small models and release dates: [QwenLM/Qwen3.8 repo news](https://github.com/QwenLM/Qwen3.8); [Gemma 4 vs Qwen 3.5 comparison](https://codersera.com/blog/gemma-4-vs-qwen-3-5-comparison-2026/); [Qwen 3.8 lineup](https://codersera.com/blog/qwen-3-8-model-lineup-2026/)
- Gemma 4 E2B/E4B sizes and license: [WaveSpeed overview](https://wavespeed.ai/blog/posts/what-is-google-gemma-4/); [SiliconANGLE](https://siliconangle.com/?p=737125)
- Small-model landscape: [Turing Post SLM list](https://www.turingpost.com/p/slmslist); [KDnuggets: small models for tool calling](https://www.kdnuggets.com/5-small-language-models-for-agentic-tool-calling)
- Activation Oracles: [arXiv 2512.15674](https://arxiv.org/abs/2512.15674); LatentQA: [arXiv 2412.08686](https://arxiv.org/abs/2412.08686)
