# Track 2: Model formats, size limits, and storing edits as deltas

*Status: desk research plus arithmetic, 9 Nov 2026. Joey's decisions: no hard size cap (warn, don't block), and edits are stored as low-rank deltas (track 1).*

## 1. Formats

| Format | What it is | Editing implications |
|---|---|---|
| **safetensors** (HF) | 8-byte header length, a JSON header (name → dtype, shape, offsets), then one packed little-endian buffer. Supports lazy and partial loading by tensor name, mmap on CPU, BF16 and FP8, and a `__metadata__` string map. 100 MB header limit. ([repo](https://github.com/huggingface/safetensors)) | **The native editing format.** Weights are stored at full precision, a single tensor can be read without loading the rest, and an edited tensor can be written back to just its shard. Our `Backend::read_tensor` maps straight onto it. |
| **MLX quantized** (safetensors container) | Affine quantization per group (typically 64 elements): packed ints plus a per-group `scales` and `biases` tensor | Editing a quantized tensor means dequantizing, adding the delta, and requantizing the touched groups. That's lossy (see §3). mlx-lm usually keeps sensitive tensors at higher bits (this varies by recipe; check each model's `quantization` config). |
| **GGUF** (llama.cpp) | "GGUF" magic, version 3, then KV metadata, tensor descriptors (name, dims, ggml type, offset) and data aligned to 32 bytes. Types: F32/F16/BF16, legacy Q4_0–Q8_0, K-quants Q2_K–Q8_K, I-quants, TQ and MXFP4. ([llama.cpp docs](https://www.mintlify.com/ggml-org/llama.cpp/concepts/gguf-format)) | Block-quantized, so the same dequantize/requantize issue applies. Tensor names differ from HF (`output.weight`, `token_embd.weight`), so we need a name map. llama.cpp's LoRA converter rejects embedding adapters, and its source says it ignores `lm_head` ones (track 1). That means a GGUF export of Joey-style edits must **patch `output.weight` directly**, not ship as a LoRA. |
| **Others** (GPTQ/AWQ/EXL3, FP8 checkpoints, UQFF for mistral.rs) | Mostly kernel-specific quantized layouts | Read-only sources at best. Support comes later, through "dequantize the tensors we touch". |

**Rule:** we edit at full precision. If the base model is quantized, the tensors we touch get a **full-precision overlay**; the rest stays quantized.

## 2. Size limits (no cap; the app warns)

Weights only, at 2 bytes/param for bf16 and about 0.56 bytes/param at roughly 4.5-bit including scales. These are our own calculations. KV cache, activations, a navigator model and OS headroom come on top. On Macs, the GPU can by default use only part of unified memory (about 65–75%; the wired limit can be raised).

| Model | Params (total / active) | bf16 weights | ~4.5-bit weights |
|---|---|---|---|
| Qwen3.5-9B | 9B / 9B | 18 GB (measured: 18 GB snapshot, 20 GB peak) | ~5 GB |
| Qwen3.5-27B | 27B / 27B | 54 GB | ~15 GB |
| Qwen3.5-35B-A3B (MoE) | 35B / 3B | 70 GB | ~20 GB |
| 70B dense | 70B / 70B | 140 GB | ~39 GB |
| 90B dense (Joey's "turn knobs for a couple of days" case) | 90B | 180 GB | ~50 GB |
| Qwen3.5-397B-A17B (MoE) | 397B / 17B | 794 GB | ~222 GB |

| Hardware | Fits in bf16 | Fits with a quantized base + overlay |
|---|---|---|
| 16 GB GPU (**Joey-PC: RTX 5080**, ~10 GB already used by desktop apps) | 9B bf16 does **not** fit; needs CPU offload or a quantized base | ≤ ~20B |
| 24 GB GPU (e.g. 4090) | ≤ ~9B (tight) | ≤ ~35B |
| 32 GB GPU (e.g. 5090) | ~9–12B | ≤ ~50B |
| 128 GB Mac (Joey's) | ≤ ~50B dense, 35B-A3B | ≤ ~180B dense (slow) |
| 192–512 GB Mac Studio | ≤ ~90–200B | 397B-A17B at 4-bit on 512 GB |

**Trial speed** scales with *active* parameters per token: about 30 tok/s at 9B on Joey's Mac (track 1), so roughly 3–4 tok/s for a 70B dense model. MoE models with few active parameters (35B-A3B) should trial *faster* than a dense 9B while needing more memory. The app's pre-load check should show "fits / doesn't fit" and "≈ N trials per minute" from these two numbers.

**Overlay cost for `lm_head`:** Qwen3.5-9B's `lm_head` is 248,320 × 4,096 = 1.02B params, which is 2.03 GB in bf16 or 4.07 GB in f32. A full-precision overlay of just the edited **rows** is tiny (one row is 16 KB in f32). Rows or low-rank factors are the unit to store, not whole tensors.

## 3. Precision: where edits get lost

- **Merging rounds the edit.** Joey's edit-brief delta has max |Δ| = 0.019 on a single row (measured). Adding it into bf16 weights rounds every element to bf16's ~3 significant digits. That rounding is itself a small perturbation, and track 1 showed near-tie edits flip on perturbations that small. **So the merged snapshot isn't guaranteed to behave like the live trial that was kept.** Keep the delta in f32 and apply it at runtime as an overlay. When merging for export, re-run the gate on the merged artifact.
- **Merging into a quantized base is worse.** Folding a 16-bit adapter into 4-bit weights loses information unpredictably, and the community advice is not to merge into 4-bit at all ([Kaitchup](https://kaitchup.substack.com/p/dont-merge-your-lora-adapter-into?open=false)). Editing research shows success, generalization and locality all fall as bit-width drops, most steeply below 4 bits ([2407.06483](https://arxiv.org/pdf/2407.06483)).
- **Consequence for the gate (track 6):** "numeric robustness" must include *the exact artifact being exported*, such as a merged bf16 file, a GGUF Q4_K_M or an MLX 4-bit, not just the live overlay.
- **Observation:** Joey's kept snapshots are ~20 GB against 18 GB for the original. The save path likely wrote some tensors in a wider dtype, or with different sharding. This is unverified, and worth checking: a dtype change on save is one more silent numeric change.

## 4. Edit storage and lineage

**Unit: a `Delta`.**

```
Delta {
  id, parent_snapshot,            // lineage
  base: { model_id, revision, shard_sha256[] },
  tensor: "model.language_model.lm_head.weight",
  kind: RowSparse { rows: [248046], values: f32[rows × 4096] }
      | LowRank  { u: f32[out × r], v: f32[r × in], scale }
      | Dense    { values: f32[...] }        // discouraged; rejected by default for lm_head (Family A)
  applied_dtype: bf16,            // dtype the gate validated against
  gate: { verdict, margin, ledger_row_ids[] },
  created_at, author (user | agent | navigator)
}
```

A **snapshot** is `base + ordered [Delta]`, which is a few KB to MB instead of a 19 GB copy. Lineage is a DAG, so branching and A/B between siblings come for free.

**Exports**, chosen by the user:

1. **PEFT LoRA** (`adapter_model.safetensors` + `adapter_config.json`, with `target_modules` including `lm_head`). Rank-1 and row-sparse deltas convert exactly to LoRA, since a row-sparse delta is rank ≤ number of rows. Works with transformers, vLLM, mistral.rs and mlx-lm where they accept `lm_head` targets (to verify per engine). ([PEFT checkpoint format](https://huggingface.co/docs/peft/en/developer_guides/checkpoint.md))
2. **Merged safetensors:** rewrite only the shards that contain edited tensors and copy or link the rest. For Qwen3.5-9B, editing `lm_head` touches one shard, not all four.
3. **GGUF:** patch `output.weight` in an existing GGUF in place if it's F16 or BF16. If it's quantized, requantize just that tensor and re-gate (§3).
4. **MLX:** merged safetensors plus mlx-lm's own conversion, or an MLX adapter where supported.

## 5. Open items

- Verify each engine accepts a LoRA on `lm_head` (mistral.rs, mlx-lm, vLLM). llama.cpp's converter is known not to.
- Check why Joey's kept snapshots are 2 GB larger than the original (dtype or sharding on save).
- Measure: does merging edit-brief into bf16 reproduce the live overlay's outputs on all probes and controls?
- Tied-embedding models: an edit to `lm_head` is also an input-embedding edit. Decide whether to untie on load.
- MoE: are router and expert tensors in scope for editing? Expert sparsity changes the "active params" speed model.
