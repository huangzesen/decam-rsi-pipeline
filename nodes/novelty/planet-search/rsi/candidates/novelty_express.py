#!/usr/bin/env python3
"""Run novelty_check.py now for chosen inbox items instead of waiting for novelty_watch.py (2026-11-17: the watcher takes a
snapshot of the inbox and works through all of it, ~45 items an hour, so seen-again leads queued behind 300 others wait
hours). Each item is claimed first by moving it out of the inbox (os.replace), so the watcher skips it and nothing runs
twice; results, log, RESULTS.md line and the spec go where the watcher puts them. No hand-off to the confirm queue (the
all-archive checks now go to the TRKSUB01 session). Usage: ~/anaconda3/bin/python -I novelty_express.py ID ..."""
import json, os, subprocess, sys, time
C = os.path.dirname(os.path.abspath(__file__)); Q = C + "/novelty_queue"; PY = sys.executable
STAGE = Q + "/express"; os.makedirs(STAGE, exist_ok=True)
mpc = sorted(d for d in os.listdir(os.path.join(os.path.dirname(C), "data", "mpc")) if d.startswith("20"))[-1]
for name in sys.argv[1:]:
    spec = f"{Q}/inbox/{name}.json"; mine = f"{STAGE}/{name}.json"
    try:
        os.replace(spec, mine)
    except OSError:
        print(name, "not in the inbox (the watcher has it, or it is done)", flush=True); continue
    out = f"{Q}/done/{name}.result.json"; t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        p = subprocess.run([PY, "-I", C + "/novelty_check.py", mine, out], capture_output=True, text=True, timeout=2400, env=env, cwd=C)
        log = (p.stdout or "")[-3000:] + (p.stderr or "")[-1500:]
    except subprocess.TimeoutExpired:
        log = "TIMEOUT after 40 min"
    open(f"{Q}/done/{name}.log", "w").write(log)
    try:
        R = json.load(open(out)); verdict, nf, sid = R["verdict"], len(R.get("flagged", [])), R.get("id") or name
    except Exception:
        verdict, nf, sid = "ERROR (see done/" + name + ".log)", "-", name
    with open(Q + "/RESULTS.md", "a") as fh:
        fh.write(f"| {time.strftime('%F %H:%M')} | {sid} | {verdict} | {nf} | {mpc} | done/{name}.result.json (express) |\n")
    os.replace(mine, f"{Q}/done/{name}.json")
    print(time.strftime("%F %T"), sid, "->", verdict, f"({round(time.time() - t0)} s)", flush=True)
