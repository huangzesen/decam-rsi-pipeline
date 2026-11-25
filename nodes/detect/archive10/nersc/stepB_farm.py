#!/usr/bin/env python3
"""Step B of the second 5 % (2026-11-16): link shares for the detection of the other 20 regions.
The same as stepA_farm.py with the shares named sb_<k>: for each stem list (stepB/stems_<k>.txt, one exposure stem per
line) make $A/sb_<k>/{img,det,log} and link the exposure's three files (ooi, oow, ood) into sb_<k>/img/<YYMMDD>/ from
wherever they are on scratch (the DESI-mirror links ls_0, ls_1 or the downloads a10n_0..3). Links only: nothing is
copied or downloaded. sb_0..3 hold the 19 smaller regions, sb_4..7 the one large region R061.
Usage, from $A on NERSC: python -I stepB_farm.py stepB/stems_0.txt ... stepB/stems_7.txt
"""
import glob, os, sys
A = os.path.dirname(os.path.abspath(__file__))
SRC = [os.path.join(A, d, "img") for d in ("ls_0", "ls_1", "a10n_0", "a10n_1", "a10n_2", "a10n_3")]
for k, f in enumerate(sys.argv[1:]):
    D = os.path.join(A, f"sb_{k}")
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
    print(f"sb_{k}: {n} exposures linked, {bad} incomplete")
