import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.home()/"code/weight-space/qwen-harness"))
from mlx_lm import load
import mlx.core as mx
from qwen_harness.chat_generate import generate_chat_batch, generate_chat_completion
OUT = Path.home()/"code/weightless-bench/out"; OUT.mkdir(parents=True, exist_ok=True)
W = Path.home()/"code/weight-space/qwen-harness/weights"
P = [l.strip() for l in (Path.home()/"code/weight-space/qwen-harness/probes.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
P += ["What is 8 times 7?", "A square has side 4. What is its area?"]  # SCORE_EXTRA from qwen_harness/trial_resident.py
for snap in ["c2022362", "edit-brief", "edit-gcd", "edit-template"]:
    m, t = load(str(W/f"Qwen3.5-9B-{snap}"))
    br = generate_chat_batch(m, t, P, max_tokens=256)
    sr = [generate_chat_completion(m, t, p, max_tokens=256) for p in P]
    b = [r["char_count"] for r in br]; s = [r["char_count"] for r in sr]
    (OUT/f"mlx_paths_{snap}.json").write_text(json.dumps(dict(prompts=P, batch=br, stream=sr), indent=2, default=str))
    print(snap, "batch", b, sum(b[:5]), "stream", s, sum(s[:5]), flush=True)
    del m; mx.clear_cache()
