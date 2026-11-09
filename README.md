# Weightless

Desktop app for post-training open-weight models by editing their weights directly: trace a generation, pick
the step where the model nearly said something else, apply a rank-1 edit, and keep it only if every probe and
control still passes. Inspired by weight-space programming ([arXiv 2611.05341](https://arxiv.org/abs/2611.05341)).

Every trial, kept or rejected, is written to a ledger. That ledger is the training data for Weightless's own
small navigator model, so using the app is how the navigator gets trained.

## Layout

```
crates/weightless-core     backend-agnostic engine: adapter contract, deltas, spec, gates, ledger, workbench
crates/weightless-candle   primary backend (candle stack; Qwen3 dense today, Qwen3.5 via mistral.rs next)
crates/weightless-cli      `weightless` headless CLI: info, trace, explore, export
app/                       Tauri v2 shell (src-tauri) + React dashboard (src)
docs/research/             research tracks behind the design
```

## How it works

- **Adapter contract** (`weightless_core::backend::Backend`): every runtime implements load, trace with
  per-step logits and hidden states, apply/revert deltas, and advertises `Capabilities`. The UI hides what a
  backend can't do. Shape from docs/research/01-inference-engines.md.
- **Edits are f32 overlays** (`Delta`): a sum of rank-1 terms on a named tensor, never merged into the base
  weights. Today the candle backend edits `lm_head`, adding the delta to the logits from the captured final hidden
  state, so applying and reverting are exact and free. The nudge `W[k] += (δ/‖h‖²)·h` raises exactly one logit by δ at
  hidden state h.
- **Spec** (`Spec`): the goal in words, probes to improve, controls that must not change, and required answers.
- **Gates** (`gate`): a pre-flight screen predicts exactly which greedy step each probe would first flip (for
  single-row `lm_head` nudges), so control spills are rejected without decoding. Survivors are decoded on every
  probe and control and gated on controls identical, answers kept, site flipped, objective met, magnitude. Every
  gate result and reason code is logged.
- **Ledger** (`Ledger`, SQLite): sessions (goal, spec, model fingerprint, backend, dtype, quantization, device,
  decode path), trials (proposer, decision group, site features, predictions, per-probe results, gates, cost),
  snapshots (lineage), dial requests, follow-up events and agent turns. `export_jsonl` writes one trial per line
  for navigator training. Field list agreed with docs/research/08-navigator-model.md.
- **Explore** (`Workbench::explore`): a heuristic proposer that tries the runner-up at every near-tie step and any
  end-of-turn token, so the app generates labelled data on its own.

## Build and run

Requirements: Rust (stable), Node 22. On Linux, Tauri also needs
`libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev libsoup-3.0-dev`.

```sh
cargo test                          # engine tests on a tiny random Qwen3 (no download needed)

# Desktop app
cd app && npm install
npm run tauri dev                                  # CPU
npm run tauri dev -- --features metal              # Mac GPU
npm run tauri dev -- --features cuda               # NVIDIA GPU

# Headless
cargo run --release -p weightless-cli --features metal -- info  --model ~/models/Qwen3-0.6B
cargo run --release -p weightless-cli --features metal -- trace --model ~/models/Qwen3-0.6B --prompt "What is 8x7?"
cargo run --release -p weightless-cli --features metal -- explore --model ~/models/Qwen3-0.6B --budget 50
cargo run --release -p weightless-cli -- export --out trials.jsonl
```

Models load from a local folder (`config.json`, `tokenizer.json`, `*.safetensors`), for example
`huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B`. The app's ledger lives in the OS app
data folder (shown on the Model tab).

## Not yet

- Qwen3.5 (GDN hybrid) model code ported from mistral.rs, and edits on tensors other than `lm_head`.
- Other backends (llama.cpp, MLX, burn) and the cross-backend parity harness.
- Batched trials, per-request deltas for concurrent workers, quantized bases.
- Export of kept edits (PEFT LoRA, merged safetensors, GGUF) and serve mode.
- The agent API and the navigator model itself.
