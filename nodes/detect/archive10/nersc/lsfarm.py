#!/usr/bin/env python3
"""archive10: point detection at the DECam images NERSC already holds (Legacy Surveys staging, read-only, not copied).
2026-11-14. Reads ls10_nersc.txt ("STEM OOI_PATH",
in score order) and makes, for part h of N (every N-th exposure), symlinks ls_h/img/<YYMMDD>/<ooi,oow,ood> so detect.py
reads them as if downloaded. Exposures whose weight or mask file is missing are listed in ls_h/log/incomplete.txt.
Usage (from $A): python -I lsfarm.py ls10_nersc.txt N"""
import os, sys
A = os.path.dirname(os.path.abspath(__file__)); rows = [l.split() for l in open(sys.argv[1]) if l.strip()]; N = int(sys.argv[2])
for h in range(N):
    D = os.path.join(A, f"ls_{h}"); os.makedirs(D + "/log", exist_ok=True); os.makedirs(D + "/det", exist_ok=True)
    n = bad = 0; inc = open(D + "/log/incomplete.txt", "w")
    for stem, p in rows[h::N]:
        trio = [p, p.replace("_ooi_", "_oow_"), p.replace("_ooi_", "_ood_")]
        if not all(os.path.exists(x) for x in trio):
            inc.write(stem + "\n"); bad += 1; continue
        d = os.path.join(D, "img", stem[4:10]); os.makedirs(d, exist_ok=True)
        for x in trio:
            l = os.path.join(d, os.path.basename(x))
            if not os.path.lexists(l):
                os.symlink(x, l)
        n += 1
    open(D + "/log/FETCH_DONE", "w").close()
    print(f"ls_{h}: {n} exposures linked, {bad} incomplete")
