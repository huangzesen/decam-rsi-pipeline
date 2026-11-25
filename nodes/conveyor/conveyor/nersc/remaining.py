#!/usr/bin/env python3
"""conveyor unit (NERSC): exposures still to detect per share of a unit: complete ooi+oow+ood trios in <share>/img minus
<share>/det lists. Shares: m<k> (mirror links), p<k> (NOIRLab downloads), v (pulled back from Vista). A download share
counts as final when its log/SHARE_DONE exists (v: log/PULL_DONE). An exposure whose detection failed twice counts as
finished (unread). Prints JSON {share: [left, total, final]}.
Usage: python3 -I remaining.py U"""
import glob, json, os, sys
U = os.path.abspath(sys.argv[1]); out = {}
for D in sorted(glob.glob(os.path.join(U, "m[0-9]*")) + glob.glob(os.path.join(U, "p[0-9]*")) + glob.glob(os.path.join(U, "v"))):
    if not os.path.isdir(os.path.join(D, "img")):
        continue
    fs = set(os.path.basename(p) for p in glob.glob(f"{D}/img/*/c4d_*.fits.fz"))
    stems = {f[:17] for f in fs if "_ooi_" in f and f.replace("_ooi_", "_oow_") in fs and f.replace("_ooi_", "_ood_") in fs}
    done = {os.path.basename(p)[:17] for p in glob.glob(f"{D}/det/*/c4d_*.npz") if ".tmp." not in p}
    # an exposure that failed twice (e.g. a broken FITS header in the mirror, 2026-11-18) counts as finished, unread
    errs = {}
    lg = f"{D}/det/detect.log"
    if os.path.exists(lg):
        for l in open(lg):
            if " ERROR " in l:
                st = l.split()[2] if len(l.split()) > 2 else ""
                errs[st] = errs.get(st, 0) + 1
    done |= {st for st, n in errs.items() if n >= 2}
    s = os.path.basename(D)
    final = s.startswith("m") or os.path.exists(f"{D}/log/SHARE_DONE") or os.path.exists(f"{D}/log/PULL_DONE")
    out[s] = [len(stems - done), len(stems), final]
print(json.dumps(out))
