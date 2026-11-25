#!/usr/bin/env python3
"""archive10 step A (2026-11-16): the latest hill-climb program on the five regions of the second
5 % that were never searched and are on NERSC (R044 R057 R058 R059 R069; 4,782 exposures). Makes, for share k, the folder
sa_k/{img,det,log} with symlinks sa_k/img/<YYMMDD>/<ooi,oow,ood> to the files already here (the DESI mirror links in
ls_0, ls_1 and the downloads in a10n_0..3), so detect.py reads one share per CPU node. Nothing is copied.
Usage (from $A): python -I stepA_farm.py stepA/stems_0.txt stepA/stems_1.txt ..."""
import glob, os, sys
A = os.path.dirname(os.path.abspath(__file__))
SRC = [os.path.join(A, d, "img") for d in ("ls_0", "ls_1", "a10n_0", "a10n_1", "a10n_2", "a10n_3")]
for k, f in enumerate(sys.argv[1:]):
    D = os.path.join(A, f"sa_{k}")
    for d in ("img", "det", "log"):
        os.makedirs(os.path.join(D, d), exist_ok=True)
    n = bad = 0
    for stem in open(f).read().split():
        trio = []
        for s in SRC:
            trio = [p for p in glob.glob(os.path.join(s, stem[4:10], stem + "_oo[iwd]_*.fits.fz")) if os.path.getsize(p) > 0]
            if len(trio) == 3:
                break
        if len(trio) != 3:
            bad += 1; print("incomplete", stem); continue
        d = os.path.join(D, "img", stem[4:10]); os.makedirs(d, exist_ok=True)
        for p in trio:
            l = os.path.join(d, os.path.basename(p))
            if not os.path.lexists(l):
                os.symlink(os.path.realpath(p), l)
        n += 1
    print(f"sa_{k}: {n} exposures linked, {bad} incomplete")
