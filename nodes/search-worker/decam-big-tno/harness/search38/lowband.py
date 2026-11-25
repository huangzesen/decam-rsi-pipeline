#!/usr/bin/env python3
"""search38: lead lists BELOW the confidence cut, for the in-data repeat test (2026-11-15).

The grader scores only the top K = 2 x findable candidates, so only those became leads. This takes every valid candidate of a
scored search with LO <= p < HI that is lead-like (>= 80 % unlabelled detections, >= 5 nights), ranks it among all candidates,
and files it as the pseudo-generation runs/s38/gens/TAG/grade/grade.json ({"leads": [...]}), then adds the ones not already
in the ledger to runs/s38/leads_sent.json with harness/leads.py's own code. Their novelty specs go to
novelty_queue/held_low/ (not the inbox): the novelty check costs ~2.7 min a lead, so it is run only on leads the repeat test
sees again. Then: leadcheck_par.py start s38 TAG K.
Usage: ~/anaconda3/bin/python -I harness/search38/lowband.py SRC_GEN TAG LO HI     (e.g. r2b-gen014 r2blowA-gen014 0.2 0.390978)
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src, tag, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
G = os.path.join(HERE, "runs", "s38", "gens", src); TR = os.path.join(HERE, "private", "truth38x")
g = json.load(open(os.path.join(G, "grade", "grade.json")))
allc = []
for sid in g["per_scenario"]:
    z = np.load(os.path.join(TR, sid + ".npz")); n = len(z["det_body"])
    for c in json.load(open(os.path.join(G, "out", sid, "candidates.json"))):
        det = np.unique(np.asarray(c["det"], dtype=np.int64)); p = float(c["p"])
        if len(det) and det.min() >= 0 and det.max() < n and math.isfinite(p):
            allc.append((p, sid, det))
allc.sort(key=lambda x: -x[0])
T = {}; leads = []
for rank, (p, sid, det) in enumerate(allc):
    if not lo <= p < hi:
        continue
    if sid not in T:
        z = np.load(os.path.join(TR, sid + ".npz")); T[sid] = (z["det_body"], z["det_night"])
    lab = T[sid][0][det]; v, k = np.unique(lab, return_counts=True)
    if int(v[np.argmax(k)]) == -1 and k.max() / len(lab) >= 0.8:
        nn = len(np.unique(T[sid][1][det]))
        if nn >= 5:
            leads.append(dict(sid=sid, rank=rank, p=p, det=det.tolist(), nights=nn))
out = os.path.join(HERE, "runs", "s38", "gens", tag, "grade"); os.makedirs(out, exist_ok=True)
json.dump(dict(leads=leads, note=f"candidates of {src} with {lo} <= p < {hi}, below the scored cut {g['cut_p']}; made by lowband.py"),
          open(os.path.join(out, "grade.json.new"), "w")); os.replace(os.path.join(out, "grade.json.new"), os.path.join(out, "grade.json"))
print(tag, len(leads), "lead-like candidates with", lo, "<= p <", hi, flush=True)
os.environ["DBT_SITE"] = "nersc"
sys.path.insert(0, os.path.join(HERE, "harness")); sys.argv = ["leads.py", "s38", tag, "--max", "100000", "--no-check"]
import leads as L
L.INBOX = os.path.join(os.path.dirname(L.INBOX), "held_low")
L.main()
