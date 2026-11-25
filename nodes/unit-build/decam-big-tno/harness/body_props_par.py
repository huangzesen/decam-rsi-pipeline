#!/usr/bin/env python3
"""body_props.py --add with the orbit fits spread over worker processes (same fits, same fields).
Usage: ~/anaconda3/bin/python -I harness/body_props_par.py RUN [WORKERS=6]
Fits only the bodies runs/RUN/loss/bodies.json does not hold, and writes the merged table to
runs/RUN/loss/bodies_par.json (move it over bodies.json by hand; the serial script may be writing that file).
Made 2026-11-14 because the serial fits of the 105 bodies of the 19 non-graded open regions took over 18 minutes.
"""
import json, os, subprocess, sys
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
os.environ["DBT_SITE"] = "nersc"
import lead_check as LC                                   # noqa: E402

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


def fit(job):
    sid, b, m, snr, t, ra, dec = job
    snr, t, ra, dec = map(np.array, (snr, t, ra, dec))
    m.update(snr_med=round(float(np.median(snr)), 1), snr_max=round(float(snr.max()), 1), n_det=len(t))
    try:
        rms, pv, cov, t0, el = LC.fit_lead(t, ra, dec, np.full(len(t), 0.15))
        m.update(r_au=round(float(np.linalg.norm(pv[:3])), 1), a=round(float(el["a"]), 1), e=round(float(el["e"]), 2),
                 i=round(float(el["i"]), 1), fit_rms=round(rms, 2))
    except Exception as err:
        m.update(fit=f"failed: {type(err).__name__}")
    return sid, b, m


if __name__ == "__main__":
    RUN = sys.argv[1]; NW = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    CFG = json.load(open(os.path.join(HERE, "runs", RUN, "config.json")))
    TRUTH = os.path.join(HERE, CFG.get("truth", "private/truth")); OUT = os.path.join(HERE, "runs", RUN, "loss")
    OLD = json.load(open(os.path.join(OUT, "bodies.json")))
    q, meta, rowsof = {}, {}, {}
    for sid in CFG["sids"]:
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
            if CFG.get("design") == 2:
                meta[(sid, b)].update(name=k["name"], H=k["H"])
            rowsof[(sid, b)] = [int(r) for r in rows]
            q.setdefault(sid, []).extend(rowsof[(sid, b)])
    print(len(meta), "bodies to fit in", len(q), "regions", flush=True)
    p = subprocess.run(LC.VSSH + [f"{LC.VPY} '{REMOTE}'"], input=json.dumps(q), capture_output=True, text=True, timeout=1800)
    if p.returncode:
        sys.exit("NERSC read failed: " + p.stderr[-400:])
    R = json.loads(p.stdout); jobs = []
    for (sid, b), m in meta.items():
        d = R[sid]; ix = {r: i for i, r in enumerate(d["rows"])}; k = [ix[r] for r in rowsof[(sid, b)]]
        jobs.append((sid, b, m, [d["snr"][i] for i in k], [d["mjd"][i] for i in k], [d["ra"][i] for i in k], [d["dec"][i] for i in k]))
    jobs.sort(key=lambda j: -len(j[4]))                    # longest first
    res, n = {}, 0
    with Pool(NW) as pool:
        for sid, b, m in pool.imap_unordered(fit, jobs):
            res.setdefault(sid, {})[str(b)] = m; n += 1
            print(n, "of", len(jobs), sid, b, m.get("n_det"), m.get("fit", "ok"), flush=True)
    for sid, v in OLD.items():
        res.setdefault(sid, {}).update(v)
    res = {sid: {b: v[b] for b in sorted(v, key=int)} for sid, v in res.items()}
    json.dump(res, open(os.path.join(OUT, "bodies_par.json.new"), "w"), indent=1)
    os.replace(os.path.join(OUT, "bodies_par.json.new"), os.path.join(OUT, "bodies_par.json"))
    print(sum(len(v) for v in res.values()), "findable bodies written to", os.path.join(OUT, "bodies_par.json"), flush=True)
