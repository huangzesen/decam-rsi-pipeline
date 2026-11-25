#!/usr/bin/env python3
"""conveyor unit (NERSC): link the unit's exposures that are in the DESI Legacy Surveys staging mirror into detection
shares, a copy of ldt5/l5_farm.py. Reads U/ls_nersc.txt ("stem path_of_ooi") and U/stems_<k>.txt; makes U/m<k>/{img,det,log}
and links ooi, oow, ood into img/<YYMMDD>/. Links only. Usage: python -I unit_farm.py U"""
import glob, os, sys
U = os.path.abspath(sys.argv[1])
LS = dict(l.split() for l in open(os.path.join(U, "ls_nersc.txt")) if l.strip())
for f in sorted(glob.glob(os.path.join(U, "stems_*.txt"))):
    k = os.path.basename(f)[6:-4]; D = os.path.join(U, f"m{k}")
    for d in ("img", "det", "log"):
        os.makedirs(os.path.join(D, d), exist_ok=True)
    n = bad = 0; inc = open(os.path.join(D, "log", "incomplete.txt"), "w")
    for stem in open(f).read().split():
        p = LS.get(stem); trio = [p, p.replace("_ooi_", "_oow_"), p.replace("_ooi_", "_ood_")] if p else []
        if not trio or not all(os.path.exists(x) for x in trio):
            bad += 1; inc.write(stem + "\n"); continue
        d = os.path.join(D, "img", stem[4:10]); os.makedirs(d, exist_ok=True)
        for p in trio:
            l = os.path.join(d, os.path.basename(p))
            if not os.path.lexists(l):
                os.symlink(os.path.realpath(p), l)
        n += 1
    print(f"m{k}: {n} exposures linked, {bad} incomplete", flush=True)
