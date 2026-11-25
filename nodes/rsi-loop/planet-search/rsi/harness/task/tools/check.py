"""Check that submission/ runs the way the grader runs it and that its output is valid. Gives no score.

For each development scenario: copy submission/ to a fresh folder, run
    python -I finder.py --scenario SCEN --out OUT.json --budget-s B
from that folder with at most 4 CPU threads, kill it B + 30 s after start, then validate OUT.json: a dict with a
"candidates" list of at most 200 entries, each with a finite epoch_mjd, a 6-number finite state, p in [0, 1]; it also
reports states the grader would drop (unbound orbits, or closer than 25 au to the Sun).

Usage: python tools/check.py [--scenario NAME ...] [--budget-s 300]
"""
import argparse, glob, json, math, os, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
GM = 2.959122082855911e-4


def check(scen, budget):
    tmp = tempfile.mkdtemp(prefix="check_")
    sub = os.path.join(tmp, "sub")
    shutil.copytree(os.path.join(WS, "submission"), sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    out = os.path.join(tmp, "OUT.json")
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTORCH_ENABLE_MPS_FALLBACK="1")
    t = time.time()
    p = subprocess.Popen([sys.executable, "-I", "finder.py", "--scenario", scen, "--out", out, "--budget-s", str(budget)],
                         cwd=sub, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        log, _ = p.communicate(timeout=budget + 30)
        status = "exit 0" if p.returncode == 0 else f"exit {p.returncode}"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9); log, _ = p.communicate(); status = "KILLED (over budget + 30 s)"
    dt = time.time() - t
    msgs = []
    n = n_drop = 0
    if not os.path.exists(out):
        msgs.append("no OUT.json written")
    else:
        try:
            d = json.load(open(out))
            c = d["candidates"]
            n = len(c)
            if n > 200:
                msgs.append(f"{n} candidates (> 200: only the 200 with the highest p are used)")
            for k, x in enumerate(c):
                s = [float(v) for v in x["state"]]
                e = float(x["epoch_mjd"]); pp = float(x["p"])
                if len(s) != 6 or not all(map(math.isfinite, s + [e, pp])) or not 0 <= pp <= 1:
                    msgs.append(f"candidate {k}: bad state/epoch/p"); continue
                r = math.sqrt(sum(v * v for v in s[:3])); v2 = sum(v * v for v in s[3:])
                if r < 25 or v2 / 2 - GM / r >= 0:
                    n_drop += 1
        except Exception as ex:
            msgs.append(f"OUT.json invalid: {type(ex).__name__}: {ex}")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{os.path.basename(scen)}: {status}, {dt:.0f} s, {n} candidates, {n_drop} would be dropped (unbound or < 25 au)"
          + ("".join("\n   " + m for m in msgs[:10])))
    if status != "exit 0" or msgs:
        print("   last output:\n   " + "\n   ".join(log.strip().splitlines()[-15:]))
    return status == "exit 0" and not msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="*")
    ap.add_argument("--budget-s", type=float, default=float(os.environ.get("RSI_BUDGET_S", 300)))
    a = ap.parse_args()
    dev = os.environ.get("RSI_DEV_DIR")
    scens = sorted(glob.glob(os.path.join(dev, "*"))) if dev else []
    if a.scenario:
        scens = [s for s in scens if os.path.basename(s) in a.scenario] or a.scenario
    sub = os.path.join(WS, "submission")
    skip = (".pyc", ".npz", ".npy", ".fits", ".fits.fz")     # not copied with the submission
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(sub) for f in fs
               if "__pycache__" not in r and not f.endswith(skip))
    print(f"submission: {size / 1e6:.1f} MB (limit 500 MB)" + ("  OVER THE LIMIT: it would score 0" if size > 500e6 else ""))
    ok = all([check(s, a.budget_s) for s in scens]) and size <= 500e6
    print("ALL OK" if ok else "PROBLEMS FOUND")


if __name__ == "__main__":
    main()
