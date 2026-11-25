#!/usr/bin/env python3
"""Split one built search scenario into K date ranges, written as new scenarios (fresh sids) next to it.

2026-11-19 -> "yes please". gen053 returned nothing
on u05's R013a (scen09: 1,516 exposures, 229.9 M detections; a CUDA index assert after its static filter fell back to
the CPU), so the region is searched again as K parts.

Cuts fall at the longest gap between consecutive exposures within +-15 % of equal exposure counts (as conveyor/unit.py
does for regions over 2,000 exposures). Needs det_exp sorted and exposures.json in time order (checked), so each part
is one contiguous slice of every det_*.npy column. Prints a JSON list of the parts with their detection index ranges
[i0, i1), which split the truth labels the same way (split_truth on the Mac).

NOT USED for R013a in the end (2026-11-20): splitting it by date kept 21 of its 42 bodies with 5+ nights (the 2019
months alone hold none), and the whole region failed again with gen056 on an A100 80 GB (unit u05w), so it was thinned instead
(thin_region.py: <= 3 exposures per night and pointing, 38 of 42 kept; unit u05r). A date split only suits regions
with several long seasons.

Usage (NERSC, CPU job): python -I split_region.py SCEN_ROOT SID K"""
import json, os, secrets, sys
import numpy as np

root, sid, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
S = os.path.join(root, sid)
E = json.load(open(os.path.join(S, "exposures.json"))); R = json.load(open(os.path.join(S, "region.json")))
mjd = np.array([e["mjd_mid"] for e in E]); n = len(E)
assert np.all(np.diff(mjd) >= 0), "exposures.json is not in time order"
de = np.load(os.path.join(S, "det_exp.npy"), mmap_mode="r")
cuts = []
for j in range(1, K):
    lo, hi = int(n * (j / K - 0.15)), int(n * (j / K + 0.15))
    cuts.append(max(range(max(lo, 1), min(hi, n - 1)), key=lambda i: mjd[i] - mjd[i - 1]))
b = [0] + sorted(set(cuts)) + [n]
# detection index range of each part: det_exp is sorted, so searchsorted on exposure index bounds
ib = [int(np.searchsorted(de, e, side="left")) for e in b[:-1]] + [len(de)]
assert de[ib[1] - 1] < b[1] if K > 1 else True
cols = sorted(f for f in os.listdir(S) if f.startswith("det_") and f.endswith(".npy"))
parts = []
for p in range(len(b) - 1):
    ns = "s" + secrets.token_hex(4); D = os.path.join(root, ns); os.makedirs(D)
    parts.append(dict(sid=ns, part=p, exp=[b[p], b[p + 1]], det=[ib[p], ib[p + 1]], n_exp=b[p + 1] - b[p], n_det=ib[p + 1] - ib[p],
                      first_mjd=float(mjd[b[p]]), last_mjd=float(mjd[b[p + 1] - 1])))
for f in cols:                                   # one pass over each column, every part written from it
    a = np.load(os.path.join(S, f), mmap_mode="r")
    for q in parts:
        x = np.array(a[q["det"][0]:q["det"][1]])
        if f == "det_exp.npy":
            x = (x - q["exp"][0]).astype(a.dtype)
        np.save(os.path.join(root, q["sid"], f), x)
for q in parts:
    D = os.path.join(root, q["sid"])
    json.dump(E[q["exp"][0]:q["exp"][1]], open(os.path.join(D, "exposures.json"), "w"))
    json.dump(dict(R, sid=q["sid"], n_exposures=q["n_exp"], n_detections=q["n_det"], first_mjd=q["first_mjd"], last_mjd=q["last_mjd"],
                   part_of=sid, part=q["part"]), open(os.path.join(D, "region.json"), "w"))
print(json.dumps(parts))
