# Track 6: Testing and A/B suite (state of the art, Nov 2026)

*Status: desk research by three parallel research agents, synthesized 9 Nov 2026. Items marked (unverified) were seen only in abstracts or summaries, or are our own calculations.*

## The big idea: score edits from cached hidden states

A rank-1 edit to `lm_head` leaves every hidden state unchanged. If we cache the final hidden state `h_x` for every prompt and position in a test set once, a candidate edit `W[k,:] += (δ/‖h‖²)·h` changes the logits at any cached `x` by

`Δlogit_k(x) = δ · (h·h_x)/‖h‖² = δ · cos(h, h_x) · ‖h_x‖/‖h‖`.

That gives us three things:

- **Locality is predictable before running anything.** A prompt at 0.98 cosine to the edit point gets about 98% of the push, scaled by the norm ratio. That's exactly Joey's "Family D" spill-over.
- **Every logprob-scored metric becomes a matrix multiply.** KL vs base, top-1 agreement, multiple-choice accuracy, flips and perplexity on teacher-forced text all qualify. Thousands of items per candidate take milliseconds, so 1000 workers can screen candidates without decoding.
- **Protect sets can be enforced by construction.** As in AlphaEdit ([2410.02355](https://arxiv.org/html/2410.02355v4)), project `h` orthogonal to the span of hidden states we must not move, then pick δ.

Only free-form generation (length, format, answers that come out after the edit point) needs real decoding. Deeper-layer edits lose the shortcut, since hidden states downstream change, but they keep everything else in this document.

## 1. Per-edit gate (what the editing literature measures)

Standard edit metrics:

- **efficacy:** the edit fires on its prompt
- **generalization:** it fires on paraphrases
- **locality/specificity:** neighbours and unrelated prompts don't change
- **portability:** downstream or multi-hop consequences
- **fluency**

Sources: CounterFact/ROME ([2202.05262](https://arxiv.org/pdf/2202.05262v1)), KnowEdit/EasyEdit ([2308.07269](https://arxiv.org/html/2308.07269v3)), MQuAKE ([2305.14795](https://arxiv.org/abs/2305.14795)), WikiBigEdit ([2503.05683](https://arxiv.org/html/2503.05683v2)).

Lessons that apply directly to us:

- **Score by free generation, not teacher forcing.** "The Mirage of Model Editing" found that single-edit success falls from 96.8% to 38.5% under realistic generation ([2502.11177](https://arxiv.org/abs/2502.11177v4)).
- **Report the worst case, not the mean.** A single edit can collapse a model while averaged metrics miss it. A cheap screen is perplexity on about 50 fixed sentences, with collapse at a perplexity above 1000 ([2402.09656](https://arxiv.org/pdf/2402.09656)).
- **Require a margin.** "Keys to Robust Edits" requires the target to win by a logit gap ε (about 2.3 means 90% top-1) on every equivalent input ([2410.09338](https://arxiv.org/html/2410.09338v2)). Our benchmark showed why: Joey's edits flip between mlx batch and stream decoding.
- **Edits stack badly.** General ability survives only a few dozen sequential ROME/MEMIT edits ([2410.18785](https://arxiv.org/abs/2410.18785)). AlphaEdit degrades past about 5k edits, and Llama3-8B MMLU falls from about 0.6 to 0.1 by 10k edits ([2606.26783](https://arxiv.org/pdf/2606.26783)). So we check the **cumulative stack**, not just each edit.
- **Steering has side effects and "anti-steerable" inputs.** Close to half of inputs move the wrong way ([2407.12404](https://arxiv.org/pdf/2407.12404)). Steering also bleeds into behaviours it never targeted, which SteeringControl calls "entanglement" ([2509.13450](https://arxiv.org/html/2509.13450v1)).
- **Quantization erodes edits,** most steeply from 4 to 2 bits ([2407.06483](https://arxiv.org/pdf/2407.06483)). Every edit must be re-checked in each format we ship.

### Proposed keep/reject gate, in order

1. **Pre-flight (from the cache, before running anything):** the predicted Δlogit on every protect-set and near-miss state stays under a threshold. Reject a whole-row add, which is Family A. Gap at the edit point ≤ ~2 logits, which is Joey's rule.
2. **Efficacy with margin:** the target wins at the edit point by ε ≥ 2–3 logits, or several times the measured bf16 jitter. That threshold is our suggestion, not a published standard.
3. **Generalization:** the edit fires on 10–20 paraphrases or contexts, scored by free generation.
4. **Locality:** the near-miss set, protect set and unrelated prompts keep identical greedy outputs. KL and top-1 change stay under thresholds. Report the worst case.
5. **Collapse screen:** perplexity on a fixed corpus within a few percent of base, plus n-gram entropy and a repetition/loop check.
6. **Numeric robustness:** re-run under batch, stream and every shipped quantization. Reject on any decision flip.
7. **Keep rule:** Joey's per-prompt rule, where every required answer is present, no prompt or control gets longer, and the total gets shorter.
8. **Capability spot-check:** every N kept edits, on the cumulative stack (tiers below).

## 2. Regression tiers (what labs and fine-tuners run)

After quantization, merging or abliteration, people judge a modified model mainly by **distance from its base model** (KL, top-1 agreement, flips), and absolute accuracy comes second:

- Compression research found accuracy can stay flat while many answers flip, and KL tracks flips at Spearman 0.981 ([2407.09141](https://arxiv.org/html/2407.09141v1)).
- llama.cpp's `llama-perplexity --kl-divergence` reports KL, "same top p" and the shift in correct-token probability ([README](https://raw.githubusercontent.com/ggml-org/llama.cpp/HEAD/tools/perplexity/README.md)).
- Heretic, an abliteration tool, optimizes first-token KL on harmless prompts ([repo](https://github.com/p-e-w/heretic)).
- Red Hat's quantization study reports scores as a percentage of the baseline, plus text similarity to the bf16 outputs ([2411.02355](https://arxiv.org/html/2411.02355v3)).

| Tier | When | Contents | Cost |
|---|---|---|---|
| **Fast gate** | every candidate | ~10 fixed generations (length, format, loops). From cache: KL and top-1 agreement vs base over ~2k tokens of chat, plus flips on tinyMMLU (100) and tinyARC (100) ([tinyBenchmarks](https://arxiv.org/html/2402.14992v2)) | seconds |
| **Medium** | candidates that pass | Cached: MMLU-Pro stratified 1–2k, ARC-C 1.2k, HellaSwag 1k, metabench 858 ([2407.12844](https://arxiv.org/html/2407.12844v2)), tracking accuracy **and** flips. Generative, batched: IFEval 541, IFBench 300 ([2507.02833](https://arxiv.org/html/2507.02833v3)), GSM8K subset 250, BBEH Mini 460 ([2502.19187](https://arxiv.org/html/2502.19187v2)), answer-length distribution | minutes (generation-bound, unverified) |
| **Full** | before calling a snapshot done | full MMLU-Pro (cached), GPQA Diamond, MATH-500, LiveBench public set, full BBEH, LiveCodeBench v6, Arena-Hard-style judged chat, a multilingual spot check, safety and refusal slice (XSTest/ToxiGen), thinking mode **on** compared with the model card ([Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B)) | overnight |

Saturation matters less than people think. We check for regressions **against our own base model**, so even GSM8K and ARC catch damage. Paired per-item comparison with the base beats any absolute-score estimator.

## 3. A/B methodology

- **Pair per prompt.** Report the mean of per-prompt differences (B−A), its SE and a 95% CI, plus the correlation between A and B. Cluster SEs on a shared source or template, because naive SEs were too small by 1.1–3× ([Miller 2024, 2411.00640](https://arxiv.org/abs/2411.00640)). Two snapshots of one model are highly correlated, so pairing helps a lot.
- **For pass/fail data,** show the discordant counts (A-only vs B-only wins) and a McNemar p-value.
- **Verdict wording:**
  - "B better" only when the CI excludes 0.
  - Otherwise "no detectable difference (could detect ≥ X pp)".
  - Regression checks use non-inferiority ("B no worse than A by more than 2 pp").
- **Sample sizes** (our calculation from the standard paired formula, unverified):

  | Effect to detect | Prompts or comparisons needed |
  |---|---|
  | 10 pp | ~150 prompts |
  | 5 pp | ~460 prompts |
  | 3 pp | ~1,300 prompts |
  | 60/40 judge preference | ~200 non-tie comparisons |

- **Early stopping:** use anytime-valid confidence sequences, not repeated p-value peeking ([Hsu & Shekhar 2607.17409](https://www.alphaxiv.org/abs/2607.17409), [SERPANT, ICML 2026](https://icml.cc/virtual/2026/poster/66048), both unverified beyond the abstract).
- **Noise floor:** always run an **A/A** check (the same snapshot twice) to measure nondeterminism. Greedy decoding isn't reproducible across batch sizes without batch-invariant kernels ([Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [vLLM batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)). Log the backend, kernel mode, dtype and seed in every report.
- **Sampled decoding:** take K = 3–5 samples per prompt and average them per prompt before computing SEs. Where possible, score the probability of the correct answer instead.

### LLM-as-judge (the user's linked agent)

- **Never use a judge for checkable properties** (correctness, length, format). Judges are near random on hard objective pairs ([JudgeBench 2410.12784](https://arxiv.org/abs/2410.12784), [RewardBench 2 2506.01937](https://arxiv.org/abs/2506.01937)).
- **Judge pairwise in both orders,** and count a disagreement as a tie. GPT-4 kept its verdict after an order swap only 65% of the time ([MT-Bench 2306.05685](https://arxiv.org/abs/2306.05685)). One 2026 preprint reports 13.6% flips on identical repeats and a 72% first-position preference ([2606.13685](https://papers.cool/arxiv/2606.13685), unverified).
- **Use a rubric plus a reference answer.** Report tie and inconsistency rates.
- **Report a length-controlled win rate** next to the raw one ([LC-AlpacaEval 2404.04475](https://arxiv.org/abs/2404.04475)). That matters for us, because edits change verbosity.
- **Flag same-family judges** for self-preference ([2404.13076](https://arxiv.org/abs/2404.13076)). Calibrate against a small human-labelled set using prediction-powered inference ([2403.07008](https://arxiv.org/abs/2403.07008)).

### Ranking thousands of candidate edits

- Use Bradley–Terry with bootstrap CIs, not online Elo ([LMArena](https://news.lmarena.ai/extended-arena)). Judge-aware BT is an option ([2601.21817](https://arxiv.org/abs/2601.21817)).
- Beware the **winner's curse:** best-of-many inflates scores ([Leaderboard Illusion 2504.20879](https://arxiv.org/abs/2504.20879)). Screen widely, then **re-test the top-k on a fresh held-out prompt set** with FDR control before keeping.

## 4. Harnesses and plumbing

| Harness | Fit |
|---|---|
| [lm-evaluation-harness](https://pypi.org/project/lm-eval/0.5.0.dev1/) (MIT) | Logprob tasks. mlx-lm ships an `mlxlm` backend; whether it supports generation is unverified. Also `local-completions` against any OpenAI-style server. |
| [Inspect](https://inspect.aisi.org.uk/providers.html) (MIT) | Generative, judged and agentic evals, with a good log viewer. No MLX provider, so use an OpenAI-compatible server. |
| lighteval, HELM, OpenCompass, EvalScope, NeMo Evaluator | Viable, but heavier or overlapping. OpenAI simple-evals stopped being updated in July 2025. |

Plan:

- Our own **cached-hidden-state scorer** in the engine for the fast gate and all logprob metrics.
- The engine exposes an **OpenAI-compatible endpoint with logprobs**, so lm-eval and Inspect run unmodified for the medium and full tiers.
- Results normalize into the **ledger**.

## 5. Ledger fields per trial

These fields serve the gate, A/B reports and navigator-model training:

- **Setup:**
  - model and base hash, edit spec (tensor, row, δ, ‖h‖, hash of h)
  - backend, decode path, kernel/batch mode, dtype, quantization, seed
- **Edit point:** logits and top-5 before and after, gap, margin.
- **Per paraphrase, near-miss and protect prompt:**
  - cos(h, h_x), predicted vs observed Δlogit
  - whether the output changed
- **Aggregates:**
  - KL and top-1 agreement vs base, perplexity and entropy deltas
  - flip counts across backends and decode paths
  - tier results with CIs
- **Outcome:** cumulative edit count, gate verdict and reason, plus judge model and version where used.

## Open items

- Verify `mlx_lm.evaluate` generation support, and the lighteval, HELM and LiveBench licenses.
- Calibrate ε and the KL and top-1 thresholds against Joey's real keeps and rejects.
- Build the A/A noise floor for each backend: mlx batch vs stream, and mistral.rs.
