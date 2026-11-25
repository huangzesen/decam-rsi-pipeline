"""Check that submission/ runs the way the grader runs it and that its output is valid. Gives no score.

On the sample patch: copy submission/ to a fresh folder, run
    python finder.py --scen SAMPLE --out OUT --budget-s B
from that folder (scratch HOME and TMPDIR, no network in this container), kill it B + 30 s after start, then validate
OUT/candidates.json: a list; each entry with "det" (row numbers inside the patch) and a finite "p"; only the 2,000 with the
highest p are read.

Usage: python tools/check.py [--budget-s 600] [--scen DIR]
"""
import argparse, json, math, os, shutil, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
SKIP = (".pyc",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-s", type=float, default=float(os.environ.get("RSI_BUDGET_S", 600)))
    ap.add_argument("--scen", default=os.environ.get("RSI_SAMPLE", "/app/data/sample"))
    a = ap.parse_args()
    sub = os.path.join(WS, "submission")
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(sub) for f in fs if "__pycache__" not in r)
    print(f"submission: {size / 1e6:.1f} MB (limit 500 MB)" + ("  OVER THE LIMIT: it would score 0" if size > 500e6 else ""))
    tmp = tempfile.mkdtemp(prefix="check_")
    cp = os.path.join(tmp, "sub"); out = os.path.join(tmp, "out"); os.makedirs(out)
    shutil.copytree(sub, cp, ignore=shutil.ignore_patterns("__pycache__", *["*" + s for s in SKIP]))
    env = dict(os.environ, HOME=os.path.join(tmp, "home"), TMPDIR=os.path.join(tmp, "tmp"))
    os.makedirs(env["HOME"]); os.makedirs(env["TMPDIR"])
    t = time.time()
    p = subprocess.Popen([sys.executable, "finder.py", "--scen", a.scen, "--out", out, "--budget-s", str(a.budget_s)], cwd=cp, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        log, _ = p.communicate(timeout=a.budget_s + 30)
        status = "exit 0" if p.returncode == 0 else f"exit {p.returncode}"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9); log, _ = p.communicate()
        status = "KILLED at budget + 30 s (the grader scores whatever candidates.json holds then)"
    dt = time.time() - t
    nrow = len(np.load(os.path.join(a.scen, "det_exp.npy"), mmap_mode="r"))
    msgs, n = [], 0
    f = os.path.join(out, "candidates.json")
    if not os.path.exists(f):
        msgs.append("no candidates.json written")
    else:
        try:
            c = json.load(open(f)); assert isinstance(c, list), "not a list"; n = len(c)
            if n > 2000:
                msgs.append(f"{n} candidates (> 2,000: only the 2,000 with the highest p are read)")
            for k, x in enumerate(c):
                pp = float(x["p"]); det = [int(v) for v in x["det"]]
                if not math.isfinite(pp):
                    msgs.append(f"candidate {k}: p {pp} is not finite")
                if not det or min(det) < 0 or max(det) >= nrow:
                    msgs.append(f"candidate {k}: row numbers outside 0..{nrow - 1}")
        except Exception as ex:
            msgs.append(f"candidates.json invalid: {type(ex).__name__}: {ex}")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"sample: {status}, {dt:.0f} s, {n} candidates" + "".join("\n   " + m for m in msgs[:10]))
    if status != "exit 0" or msgs:
        print("   last output:\n   " + "\n   ".join(log.strip().splitlines()[-15:]))
    ok = status == "exit 0" and not msgs and size <= 500e6
    print("ALL OK" if ok else "PROBLEMS FOUND")


if __name__ == "__main__":
    main()
