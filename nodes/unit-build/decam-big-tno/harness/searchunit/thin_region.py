#!/usr/bin/env python3
"""Thin a search scenario's repeated exposures: at most MAXPER exposures per (night, pointing), spread through the night;
written as a new scenario (fresh sid) next to it. Every night stays, so multi-night arcs stay.

User, 2026-11-19: rerun u05's crowded region R013a -> "yes please". gen053 and gen056 both return nothing on R013a
(scen09, 229.9 M detections in a 7.8 deg circle, ~12x the density of regions that run): a CUDA index assert in the
program's sorted source index. ~170 M of its detections come from nine nights of a deep 2019 campaign (50-200 exposures
a night on 1-8 pointings, VR 120 s). A date split lost half the findable bodies (split_region.py); thinning keeps every night.

Needs det_exp sorted (checked). Prints JSON: new sid, kept exposure indices, per-exposure detection counts of the original
(so the truth labels, in the same row order, can be cut on the Mac without det_exp).
Usage (NERSC, CPU job): python -I thin_region.py SCEN_ROOT SID MAXPER"""
import collections, json, os, secrets, sys
import numpy as np

root, sid, maxper = sys.argv[1], sys.argv[2], int(sys.argv[3])
S = os.path.join(root, sid)
E = json.load(open(os.path.join(S, "exposures.json"))); R = json.load(open(os.path.join(S, "region.json")))
de = np.load(os.path.join(S, "det_exp.npy"), mmap_mode="r")
cnt = np.bincount(np.asarray(de), minlength=len(E))
starts = np.concatenate([[0], np.cumsum(cnt)])
assert int(starts[-1]) == len(de)
groups = collections.defaultdict(list)
for i, e in enumerate(E):
    groups[(int(np.floor(e["mjd_mid"] - 0.3)), round(e["ra0"], 1), round(e["dec0"], 1))].append(i)
keep = []
for g, ix in groups.items():
    ix = sorted(ix, key=lambda i: E[i]["mjd_mid"])
    if len(ix) <= maxper:
        keep += ix
    else:
        keep += [ix[int(round(j))] for j in np.linspace(0, len(ix) - 1, maxper)]
keep = sorted(set(keep))
ns = "s" + secrets.token_hex(4); D = os.path.join(root, ns); os.makedirs(D)
remap = -np.ones(len(E), np.int64); remap[keep] = np.arange(len(keep))
sel = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in keep])
for f in sorted(x for x in os.listdir(S) if x.startswith("det_") and x.endswith(".npy")):
    a = np.load(os.path.join(S, f), mmap_mode="r")
    x = np.asarray(a)[sel]
    if f == "det_exp.npy":
        x = remap[x].astype(a.dtype)
    np.save(os.path.join(D, f), x)
json.dump([E[i] for i in keep], open(os.path.join(D, "exposures.json"), "w"))
mj = [E[i]["mjd_mid"] for i in keep]
json.dump(dict(R, sid=ns, n_exposures=len(keep), n_detections=int(len(sel)), first_mjd=min(mj), last_mjd=max(mj), thinned_from=sid,
               max_per_night_pointing=maxper), open(os.path.join(D, "region.json"), "w"))
print(json.dumps(dict(sid=ns, n_exp=len(keep), n_det=int(len(sel)), keep=keep, counts=cnt.tolist())))
