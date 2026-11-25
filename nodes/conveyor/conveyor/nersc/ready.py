#!/usr/bin/env python3
"""conveyor early build (2026-11-20): the regions of a unit that can be built now.
A region is ready when every one of its exposures has a detection list in one of the unit's shares (U/*/det/YYMMDD/
<stem>.npz). Prints the lines of LIST ("RID SID NEXP") whose region is ready and whose scenario SCEN/<sid> is not built yet.
Nothing is left out: a region waits until all its exposures are detected; the rest of the unit keeps going.
Usage: python -I ready.py U REGIONS_EXPOSURES.json LIST SCEN"""
import glob, json, os, sys
U, rx, lst, scen = sys.argv[1:5]
have = set()
for d in glob.glob(os.path.join(U, "*", "det")):
    for p in glob.glob(os.path.join(d, "*", "c4d_*.npz")):
        b = os.path.basename(p)
        if b.endswith(".npz") and ".tmp" not in b:
            have.add(b[:17])
rexp = json.load(open(rx))
for line in open(lst):
    f = line.split()
    if len(f) < 3:
        continue
    rid, sid = f[0], f[1]
    if os.path.exists(os.path.join(scen, sid, "exposures.json")):
        continue
    st = rexp.get(rid) or []
    if st and all(s in have for s in st):
        print(line.rstrip("\n"))
