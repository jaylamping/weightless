"""Weightless track 1 baseline: mlx-lm on Qwen3.5-9B (pinned), greedy, thinking off."""
import json, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load, batch_generate
from mlx_lm.models.cache import make_prompt_cache

W = Path.home() / "code/weight-space/qwen-harness/weights"
OUT = Path.home() / "code/weightless-bench/out"
PROBES = [l.strip() for l in (Path.home()/"code/weight-space/qwen-harness/probes.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
CONTROLS = ["What is 8 × 7?", "A square has side length 4. What is its area?"]
TOPK, TRACE_STEPS, MAX_TOK = 64, 50, 512

def fmt(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

def run_one(model, tok, text):
    ids = tok.encode(fmt(tok, text), add_special_tokens=False)
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(mx.array([ids]), cache=cache)[:, -1, :]
    mx.eval(logits); t_prefill = time.perf_counter() - t0
    out, tops, eos = [], [], set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
    t1 = time.perf_counter()
    for step in range(MAX_TOK):
        l32 = logits.astype(mx.float32)[0]
        nxt = int(mx.argmax(l32).item())
        if step < TRACE_STEPS:
            idx = mx.argsort(-l32)[:TOPK]
            tops.append((np.array(idx), np.array(l32[idx])))
        if nxt in eos: break
        out.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)[:, -1, :]
    mx.eval(logits); t_dec = time.perf_counter() - t1
    comp = tok.decode(out).strip()
    return dict(prompt=text, prompt_tokens=len(ids), completion=comp, chars=len(comp), gen_tokens=len(out),
                prefill_s=t_prefill, decode_s=t_dec, decode_tok_s=len(out)/t_dec if t_dec else None), tops

def suite(tag, path):
    t = time.perf_counter(); model, tok = load(str(path)); load_s = time.perf_counter() - t
    res, traces = [], {}
    run_one(model, tok, "Hi")  # warmup
    for i, p in enumerate(PROBES + CONTROLS):
        r, tops = run_one(model, tok, p); res.append(r)
        traces[f"ids_{i}"] = np.stack([a for a, _ in tops]); traces[f"vals_{i}"] = np.stack([b for _, b in tops])
        print(tag, i, r["chars"], round(r["decode_tok_s"] or 0, 1), repr(r["completion"][:60]), flush=True)
    np.savez(OUT / f"mlx_{tag}_trace.npz", **traces)
    return model, tok, dict(load_s=load_s, results=res, five_probe_chars=sum(r["chars"] for r in res[:5]))

def main():
    report = {"mlx": mx.__version__, "peak_mem_note": "mx.get_peak_memory"}
    model, tok, report["original"] = suite("original", W / "Qwen3.5-9B-c2022362")
    # batched throughput, Joey's baseline 38.4 prompts/s for 100 prompts x 16 tokens
    prompts = [tok.encode(fmt(tok, f"Reply with only the number {i}."), add_special_tokens=False) for i in range(100)]
    batch_generate(model, tok, prompts[:4], max_tokens=16, verbose=False)
    t = time.perf_counter(); batch_generate(model, tok, prompts, max_tokens=16, verbose=False); bt = time.perf_counter() - t
    report["batch100"] = dict(seconds=bt, prompts_per_s=100 / bt)
    print("batch100", report["batch100"], flush=True)
    report["peak_mem_gb"] = mx.get_peak_memory() / 1e9
    lm0 = np.array(model.language_model.lm_head.weight.astype(mx.float32)) if hasattr(model, "language_model") else None
    del model; mx.clear_cache()
    m2, tok2, report["edit_brief"] = suite("edit_brief", W / "Qwen3.5-9B-edit-brief")
    if lm0 is not None:
        lm1 = np.array(m2.language_model.lm_head.weight.astype(mx.float32))
        D = lm1 - lm0; rows = np.nonzero(np.abs(D).sum(1))[0]
        report["edit_brief_delta"] = dict(changed_rows=rows.tolist()[:20], n_changed_rows=int(len(rows)), max_abs=float(np.abs(D).max()))
        np.savez(OUT / "edit_brief_delta.npz", rows=rows, delta_rows=D[rows])
    (OUT / "mlx_report.json").write_text(json.dumps(report, indent=2))
    print("DONE", json.dumps({k: v for k, v in report.items() if k not in ("original", "edit_brief")}), flush=True)

main()
