import json, time, urllib.request, concurrent.futures as cf
import sys
from pathlib import Path
OUT = Path.home() / "code/weightless-bench/out"
OUT.mkdir(parents=True, exist_ok=True)
TAG = sys.argv[1] if len(sys.argv) > 1 else "original"
U = "http://127.0.0.1:18800/v1/chat/completions"
P = [l.strip() for l in (Path.home()/"code/weight-space/qwen-harness/probes.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
P += ["What is 8 times 7?", "A square has side 4. What is its area?"]  # SCORE_EXTRA from qwen_harness/trial_resident.py
def ask(p, n=256):
    body = dict(model="default", messages=[{"role":"user","content":p}], temperature=0, max_tokens=n, enable_thinking=False)
    r = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type":"application/json"})
    t = time.perf_counter(); d = json.load(urllib.request.urlopen(r, timeout=600)); dt = time.perf_counter()-t
    return d["choices"][0]["message"]["content"].strip(), d.get("usage", {}), dt
ask("Hi", 8)
res = []
for p in P:
    c, u, dt = ask(p); res.append(dict(prompt=p, completion=c, chars=len(c), usage=u, seconds=dt))
    print(len(c), u.get("completion_tokens"), round(u.get("completion_tokens",0)/dt,1), repr(c[:50]), flush=True)
qs = [f"Reply with only the number {i}." for i in range(100)]
t = time.perf_counter()
with cf.ThreadPoolExecutor(100) as ex: list(ex.map(lambda q: ask(q, 16), qs))
bt = time.perf_counter()-t
print("batch100", bt, 100/bt)
(OUT / f"mistralrs_{TAG}_report.json").write_text(json.dumps(dict(results=res, batch100_s=bt, prompts_per_s=100/bt, five=sum(r["chars"] for r in res[:5])), indent=2))
