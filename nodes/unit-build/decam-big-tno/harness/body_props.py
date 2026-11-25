#!/usr/bin/env python3
"""Properties of every findable known body in the sealed patches, for the agents' loss table (2026-11-14).

Per body: weight, nights, nights per observing season (nights 150+ days apart start a new season), arc, median and
max S/N of its detections, and a two-body orbit fitted to its detections (r at mid-arc, a, e, i) - no name, place or
date. Rows and labels come from private/truth (Mac); S/N, places and times from the sealed scenarios on NERSC (read on
the login node). Writes runs/<run>/loss/bodies.json, read by climb.py's loss table (computed once; the bodies don't change).
The run's config.json gives the sealed regions (sids) and, if not private/truth, the truth folder (truth).
Usage: python3 -I harness/body_props.py [RUN=r1] [--truth private/truth_r2x] [--add]
--truth: a truth folder other than the run's; --add: fit only the bodies that bodies.json does not hold yet.
"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
os.environ["DBT_SITE"] = "nersc"
import lead_check as LC                                   # noqa: E402  (fit_lead, NERSC ssh)

RUN = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "r1"
CFG = json.load(open(os.path.join(HERE, "runs", RUN, "config.json")))
opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
TRUTH = os.path.join(HERE, opt("--truth") or CFG.get("truth", "private/truth")); OUT = os.path.join(HERE, "runs", RUN, "loss"); os.makedirs(OUT, exist_ok=True)
ADD = "--add" in sys.argv                                 # only bodies not yet in bodies.json (their fits are kept)
OLD = json.load(open(os.path.join(OUT, "bodies.json"))) if ADD else {}
SIDS = CFG["sids"]
REMOTE = r'''
import json, sys, numpy as np
S = "$NERSC_SCRATCH/dbt/scen"; q = json.load(sys.stdin); out = {}
for sid, rows in q.items():
    L = lambda k: np.load(f"{S}/{sid}/det_{k}.npy", mmap_mode="r")
    E = json.load(open(f"{S}/{sid}/exposures.json")); r = np.array(rows, dtype=np.int64)
    ex = np.asarray(L("exp")[r])
    out[sid] = dict(rows=rows, snr=np.asarray(L("snr")[r], float).tolist(), ra=np.asarray(L("ra")[r], float).tolist(),
                    dec=np.asarray(L("dec")[r], float).tolist(), mjd=[E[int(e)]["mjd_mid"] for e in ex])
json.dump(out, sys.stdout)
'''
q, meta = {}, {}
for sid in SIDS:
    z = np.load(os.path.join(TRUTH, sid + ".npz")); K = json.loads(str(z["known"])); body, night = z["det_body"], z["det_night"]
    for b, k in enumerate(K):
        if k["nights"] < 5 or str(b) in OLD.get(sid, {}):
            continue
        rows = np.flatnonzero(body == b)
        ns = np.sort(np.unique(night[rows])); pat, cur = [], 1
        for x0, x1 in zip(ns[:-1], ns[1:]):
            if x1 - x0 < 150:
                cur += 1
            else:
                pat.append(cur); cur = 1
        pat.append(cur)
        meta[(sid, b)] = dict(weight=k["weight"], nights=int(len(ns)), seasons=pat, arc_yr=round(float(ns[-1] - ns[0]) / 365.25, 2))
        if CFG.get("design") == 2:                         # run r2: the agent gets everything, names and H too (2026-11-14)
            meta[(sid, b)].update(name=k["name"], H=k["H"])
        q.setdefault(sid, []).extend(int(r) for r in rows)
p = subprocess.run(LC.VSSH + [f"{LC.VPY} '{REMOTE}'"], input=json.dumps(q), capture_output=True, text=True, timeout=1800)
if p.returncode:
    sys.exit("NERSC read failed: " + p.stderr[-400:])
R = json.loads(p.stdout); res = {}
for sid in SIDS:
    if sid not in R:
        continue
    d = R[sid]; ix = {r: i for i, r in enumerate(d["rows"])}
    z = np.load(os.path.join(TRUTH, sid + ".npz")); body = z["det_body"]
    for (s, b), m in meta.items():
        if s != sid:
            continue
        k = [ix[int(r)] for r in np.flatnonzero(body == b)]
        snr = np.array([d["snr"][i] for i in k]); t = np.array([d["mjd"][i] for i in k])
        ra = np.array([d["ra"][i] for i in k]); dec = np.array([d["dec"][i] for i in k])
        m.update(snr_med=round(float(np.median(snr)), 1), snr_max=round(float(snr.max()), 1), n_det=len(k))
        try:
            best = LC.fit_lead(t, ra, dec, np.full(len(t), 0.15))
            rms, pv, cov, t0, el = best
            m.update(r_au=round(float(np.linalg.norm(pv[:3])), 1), a=round(float(el["a"]), 1), e=round(float(el["e"]), 2),
                     i=round(float(el["i"]), 1), fit_rms=round(rms, 2))
        except Exception as err:
            m.update(fit=f"failed: {type(err).__name__}")
        res.setdefault(sid, {})[str(b)] = m
for sid, v in OLD.items():
    res.setdefault(sid, {}).update(v)
res = {sid: {b: v[b] for b in sorted(v, key=int)} for sid, v in res.items()}
json.dump(res, open(os.path.join(OUT, "bodies.json.new"), "w"), indent=1)
os.replace(os.path.join(OUT, "bodies.json.new"), os.path.join(OUT, "bodies.json"))
print(sum(len(v) for v in res.values()), "findable bodies written to", os.path.join(OUT, "bodies.json"))
