#!/usr/bin/env python3
"""Grader checks: perfect candidates (every body's labelled rows, p 1) must score 1.0; perfect plus 1,000 random junk
candidates at p 0.5 must still score 1.0 (junk ranks below); junk at p 2 must push bodies out of the slots.
Usage: python3 -I oracle.py TRUTH WORK"""
import json, os, subprocess, sys
import numpy as np
TRUTH, WORK = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)
sids = sorted(f[:-4] for f in os.listdir(TRUTH) if f.endswith(".npz"))
for case in ("perfect", "junk_low", "junk_high"):
    out = os.path.join(WORK, case)
    for sid in sids:
        z = np.load(os.path.join(TRUTH, sid + ".npz")); b = z["det_body"]; nt = z["det_night"]
        C = [dict(det=np.flatnonzero(b == k).tolist(), p=1.0) for k in np.unique(b[(b >= 0) & (b < 100000)])
             if len(np.unique(nt[b == k])) >= 5]
        if case != "perfect":
            p = 0.5 if case == "junk_low" else 2.0
            C += [dict(det=rng.integers(0, len(b), 6).tolist(), p=p) for _ in range(1000)]
        os.makedirs(os.path.join(out, sid), exist_ok=True)
        json.dump(C, open(os.path.join(out, sid, "candidates.json"), "w"))
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "grade.py"), out, TRUTH, os.path.join(out, "grade")], capture_output=True, text=True)
    print(case, r.stdout.strip()[:400], r.stderr.strip()[-400:])
