#!/usr/bin/env python3
"""Design 2: add the known bodies fainter than H 8 to the truth (2026-11-14).

The H <= 8 truth was labelled when the regions were built (bin/build_scen.py: detections within 2" of the Horizons place,
the highest S/N one per exposure). This does the same for the fainter bodies of
planet-search/archive5/link/known_faint/<rid>.json (known_pos.py --hmin 8 --out known_faint), on the scenarios as they are
on NERSC scratch (read on the login node; nothing is written there), and writes NEW truth files; the old ones are not
touched:
    private/truth_r2/<sid>.npz               ->  private/truth_r2x/<sid>.npz        the four graded regions
    private/truth/, private/truth38/<sid>.npz ->  private/truth_openx/<sid>.npz      the open regions
A detection that already has a label keeps it. The new bodies are appended to the known table (name, H, weight, nights,
faint=True), so the body indices of the old truth do not change.
Usage: python3 -I harness/relabel_faint.py [--weight 1] [--rids REG_A,R037]
"""
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
os.environ["DBT_SITE"] = "nersc"
import lead_check as LC                                   # noqa: E402  (NERSC ssh)

ap = argparse.ArgumentParser()
ap.add_argument("--weight", type=int, default=1); ap.add_argument("--rids", default="")
a = ap.parse_args()
FAINT = os.path.join(HERE, "..", "..", "planet-search", "archive5", "link", "known_faint")
CFG = json.load(open(os.path.join(HERE, "runs", "r2", "config.json")))
GRADED = {rid: sid for sid, rid in CFG["regions"].items()}
reg = {r["rid"]: r["sid"] for r in json.load(open(os.path.join(HERE, "harness", "search38", "regions38.json")))}
reg["REG_C"] = "scen07"
OPEN_RIDS = ["REG_C", "R006", "R008", "R011", "R018", "R021", "R023", "R026", "R027", "R030", "R031", "R032", "R033", "R034",
             "R036", "R037", "R038", "R039", "R040"]
JOBS = []                                                 # (rid, sid, old truth, new truth folder)
for rid, sid in GRADED.items():
    JOBS.append((rid, sid, os.path.join(HERE, "private", "truth_r2", sid + ".npz"), os.path.join(HERE, "private", "truth_r2x")))
for rid in OPEN_RIDS:
    JOBS.append((rid, reg[rid], os.path.join(HERE, "private", "truth" if rid == "REG_C" else "truth38", reg[rid] + ".npz"),
                 os.path.join(HERE, "private", "truth_openx")))
if a.rids:
    JOBS = [j for j in JOBS if j[0] in a.rids.split(",")]

REMOTE = r'''
import json, sys, numpy as np
S = "$NERSC_SCRATCH/dbt/scen"; q = json.load(sys.stdin); out = {}
for sid, items in q.items():
    L = lambda k: np.load(f"{S}/{sid}/det_{k}.npy", mmap_mode="r")
    E = json.load(open(f"{S}/{sid}/exposures.json")); ix = {e["stem"]: i for i, e in enumerate(E)}
    ex = np.asarray(L("exp")); assert (np.diff(ex) >= 0).all(), "rows are not grouped by exposure in order"
    lo = np.searchsorted(ex, np.arange(len(E)), "left"); hi = np.searchsorted(ex, np.arange(len(E)), "right")
    ra, dec, snr = L("ra"), L("dec"), L("snr"); res = []
    for b, stem, r0, d0 in items:
        i = ix.get(stem)
        if i is None or hi[i] == lo[i]:
            continue
        r = np.asarray(ra[lo[i]:hi[i]]); d = np.asarray(dec[lo[i]:hi[i]])
        sep = np.hypot((r - r0) * np.cos(np.radians(d0)), d - d0) * 3600
        hit = np.flatnonzero(sep < 2.0)
        if len(hit):
            s = np.asarray(snr[lo[i]:hi[i]])[hit]; k = hit[int(np.argmax(s))]
            res.append([b, int(lo[i] + k), float(s.max())])
    out[sid] = res
json.dump(out, sys.stdout)
'''
q, faint = {}, {}
for rid, sid, old, _ in JOBS:
    f = os.path.join(FAINT, rid + ".json")
    B = json.load(open(f))["bodies"] if os.path.exists(f) else []
    have = {k["name"] for k in json.loads(str(np.load(old)["known"]))}
    B = [b for b in B if b["name"] not in have and b["H"] > 8.0]
    faint[sid] = B
    q[sid] = [[i, stem, float(p[0]), float(p[1])] for i, b in enumerate(B) for stem, p in b["pos"].items()]
print(f"{len(JOBS)} regions, {sum(len(v) for v in faint.values())} faint bodies, {sum(len(v) for v in q.values())} places", flush=True)
p = subprocess.run(LC.VSSH + [f"{LC.VPY} '{REMOTE}'"], input=json.dumps(q), capture_output=True, text=True, timeout=3600)
if p.returncode:
    sys.exit("NERSC read failed: " + p.stderr[-600:])
R = json.loads(p.stdout); tot = [0, 0, 0]
for rid, sid, old, new in JOBS:
    z = np.load(old); body = z["det_body"].copy(); night = z["det_night"]; K = json.loads(str(z["known"]))
    assert not json.loads(str(z["planted"]))
    n0 = len(K); per = {}
    for b, row, snr in R.get(sid, []):
        per.setdefault(b, []).append(row)
    taken = 0
    for i, b in enumerate(faint[sid]):
        rows = np.array(sorted(set(per.get(i, []))), dtype=np.int64)
        free = rows[body[rows] == -1] if len(rows) else rows
        taken += len(rows) - len(free)
        body[free] = n0 + i
        K.append(dict(name=b["name"], H=b["H"], weight=a.weight, nights=int(len(np.unique(night[free]))), faint=True))
    os.makedirs(new, exist_ok=True)
    np.savez_compressed(os.path.join(new, sid + ".npz"), det_body=body, det_night=night, known=json.dumps(K),
                        planted="[]", rid=rid)
    f0 = sum(1 for k in K[:n0] if k["nights"] >= 5); f1 = sum(1 for k in K[n0:] if k["nights"] >= 5)
    d1 = sum(1 for k in K[n0:] if k["nights"] >= 1)
    tot[0] += f0; tot[1] += f1; tot[2] += len(K) - n0
    print(f"{rid} {sid} {'graded' if rid in GRADED else 'open  '} H<=8 findable {f0:3d} | faint: listed {len(K) - n0:3d}, detected {d1:3d}, "
          f"findable (>= 5 nights) {f1:3d}" + (f", {taken} detections already labelled" if taken else ""), flush=True)
print(f"total: H<=8 findable {tot[0]}, faint listed {tot[2]}, faint findable {tot[1]}")
