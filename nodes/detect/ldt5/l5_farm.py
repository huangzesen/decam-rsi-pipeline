#!/usr/bin/env python3
"""ldt5 (fourth 5 %, LDT-weighted; user 2026-11-18): link the detection shares from the DESI Legacy Surveys staging mirror,
a copy of archive15/s15_farm.py without archive10's downloads. For each stem list (stems_<k>.txt) make
$L5/l5_<k>/{img,det,log} and link the exposure's ooi, oow, ood into img/<YYMMDD>/. Links only.
Usage, from $L5 on NERSC: python -I l5_farm.py stems_0.txt ... stems_7.txt"""
import os, sys
L5 = os.path.dirname(os.path.abspath(__file__))
LS = dict(l.split() for l in open(os.path.join(L5, "ls_nersc.txt")) if l.strip())
for k, f in enumerate(sys.argv[1:]):
    D = os.path.join(L5, f"l5_{k}")
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
    print(f"l5_{k}: {n} exposures linked, {bad} incomplete", flush=True)
