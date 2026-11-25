#!/usr/bin/env python3
"""search38: run the novelty check on the 5 % search's leads with a few parallel workers instead of the one watcher.

The watcher (planet-search/rsi/candidates/novelty_watch.py, owned by the <coordinator-session> session) checks one spec at a time,
~2.7 min each, so the 303 leads of search38 v28 would take ~13 h. This takes the specs whose name starts with PREFIX
out of novelty_queue/inbox/ into novelty_queue/held_s38/ (the watcher never sees them) and runs W workers, each doing
for one spec exactly what the watcher does for a spec without "confirm": novelty_check.py SPEC done/NAME.result.json
(single-threaded, 40 min limit), done/NAME.log, the spec moved to done/, one line appended to RESULTS.md.
Usage: python3 -I novelty_par.py PREFIX [WORKERS=4]   (e.g. dbt-s38-v28-gen019-; 2026-11-14)
A spec already in done/ is skipped; each spec is claimed by a lock directory held_s38/locks/NAME, so runners can be added.
"""
import glob, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

C = os.path.expanduser("$ALE_HOME/planet-search/rsi/candidates"); Q = C + "/novelty_queue"
HOLD = Q + "/held_s38"; PY = os.path.expanduser("~/anaconda3/bin/python")
prefix = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(HOLD, exist_ok=True)
busy = subprocess.run(["ps", "-eo", "command"], capture_output=True, text=True).stdout   # the watcher's current spec stays
for f in sorted(glob.glob(f"{Q}/inbox/{prefix}*.json")):
    if os.path.basename(f) in busy:
        continue
    try:
        os.replace(f, f"{HOLD}/{os.path.basename(f)}")
    except OSError:
        pass                                          # the watcher took it meanwhile
specs = []
for f in sorted(glob.glob(f"{HOLD}/{prefix}*.json"), key=lambda f: int(f.split("-rank")[1].split("-")[0])):
    if os.path.exists(f"{Q}/done/{os.path.basename(f)}"):          # already checked (by the watcher or an earlier run)
        os.remove(f); continue
    specs.append(f)
os.makedirs(HOLD + "/locks", exist_ok=True)
mpc = json.load(open(Q + "/WATCHER.json")).get("mpc_files", "?")
print(time.strftime("%F %T"), len(specs), "specs,", W, "workers, MPC files", mpc, flush=True)


def one(spec):
    name = os.path.basename(spec)[:-5]; out = f"{Q}/done/{name}.result.json"; t0 = time.time()
    try:
        os.mkdir(f"{HOLD}/locks/{name}")                            # one runner per spec, so several runners can share HOLD
    except FileExistsError:
        return
    if not os.path.exists(spec):
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        p = subprocess.run(["nice", "-n", "5", PY, "-I", C + "/novelty_check.py", spec, out], capture_output=True, text=True,
                           timeout=2400, env=env, cwd=C)
        log = (p.stdout or "")[-3000:] + (p.stderr or "")[-1500:]
    except subprocess.TimeoutExpired:
        log = "TIMEOUT after 40 min"
    open(f"{Q}/done/{name}.log", "w").write(log)
    try:
        R = json.load(open(out)); verdict, nf, sid = R["verdict"], len(R.get("flagged", [])), R.get("id") or name
    except Exception:
        verdict, nf, sid = "ERROR (see done/" + name + ".log)", "-", name
    os.replace(spec, f"{Q}/done/{name}.json")
    with open(Q + "/RESULTS.md", "a") as fh:
        fh.write(f"| {time.strftime('%F %H:%M')} | {sid} | {verdict} | {nf} | {mpc} | done/{name}.result.json |\n")
    print(time.strftime("%F %T"), sid, "->", verdict, f"({round(time.time() - t0)} s)", flush=True)


with ThreadPoolExecutor(W) as ex:
    list(ex.map(one, specs))
print(time.strftime("%F %T"), "all done", flush=True)
