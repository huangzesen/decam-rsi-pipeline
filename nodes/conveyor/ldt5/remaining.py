#!/usr/bin/env python3
"""ldt5 (NERSC): exposures still to detect per share/part: complete ooi+oow+ood trios in DIR/img minus DIR/det lists.
Prints "share left total" lines, download parts first."""
import glob, os
L5 = os.path.dirname(os.path.abspath(__file__))
for d in [f"p{k}" for k in range(7)] + [f"l5_{k}" for k in range(8)]:
    D = os.path.join(L5, d)
    if not os.path.isdir(os.path.join(D, "img")):
        continue
    fs = set(os.path.basename(p) for p in glob.glob(f"{D}/img/*/c4d_*.fits.fz"))
    stems = {f[:17] for f in fs if "_ooi_" in f and f.replace("_ooi_", "_oow_") in fs and f.replace("_ooi_", "_ood_") in fs}
    done = {os.path.basename(p)[:17] for p in glob.glob(f"{D}/det/*/c4d_*.npz") if ".tmp." not in p}
    print(d, len(stems - done), len(stems))
