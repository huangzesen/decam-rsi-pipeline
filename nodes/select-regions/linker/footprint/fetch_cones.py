"""Measured single-detection densities: for each patch centre (estimate.py's Fibonacci points that have patch-windows),
count NSC DR2 objects with ndet = 1 within `radius` deg, grouped by MJD (an ndet = 1 object's MJD is its exposure's).
Each group is matched to the DECam exposure with that start MJD (< 1e-4 d). Appends one JSON line per patch
(resumable): {"p", "ra", "dec", "radius", "counts": {exposure index: n}}, unmatched times summed separately.
Costliest patches first (by the estimate's brute cost), so a partial run covers most of the cost.

Usage: python -I footprint/fetch_cones.py EXPOSURES.npz EST.json OUT.jsonl [--radius 0.25]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nsc_fetch import query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps"); ap.add_argument("est"); ap.add_argument("out")
    ap.add_argument("--radius", type=float, default=0.25)
    a = ap.parse_args()
    X = np.load(a.exps)
    m = np.flatnonzero(X["inst"] == "c4d")
    # exposure start MJD = tmid - exptime / 2
    t0 = X["tmid"][m] - X["exptime"][m] / 2 / 86400
    o = np.argsort(t0); ts, ms = t0[o], m[o]
    E = json.load(open(a.est))
    cost = {}
    for w in E["windows"]:
        cost[w["p"]] = cost.get(w["p"], 0.0) + w["proxy"][2]
    order = sorted(cost, key=lambda p: -cost[p])
    first = {w["p"]: (w["ra"], w["dec"]) for w in E["windows"]}
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["p"] for l in open(a.out)}
    fh = open(a.out, "a")
    tt = time.time()
    for k, p in enumerate(order):
        if p in done:
            continue
        ra, dec = first[p]
        txt = query(f"SELECT ROUND(mjd::numeric, 5) AS t, COUNT(*) AS n FROM nsc_dr2.object WHERE "
                    f"q3c_radial_query(ra, dec, {ra:.5f}, {dec:.5f}, {a.radius}) AND ndet = 1 GROUP BY t")
        rows = [l.split(",") for l in txt.strip().splitlines()[1:]]
        tv = np.array([float(r[0]) for r in rows]); nv = np.array([int(r[1]) for r in rows])
        j = np.clip(np.searchsorted(ts, tv), 1, len(ts) - 1)
        j = np.where(np.abs(ts[j - 1] - tv) < np.abs(ts[j] - tv), j - 1, j)
        ok = np.abs(ts[j] - tv) < 1e-4
        counts = {}
        for e, n in zip(ms[j[ok]], nv[ok]):
            counts[int(e)] = counts.get(int(e), 0) + int(n)
        fh.write(json.dumps(dict(p=int(p), ra=ra, dec=dec, radius=a.radius, counts=counts,
                                 unmatched=int(nv[~ok].sum()), matched=int(nv[ok].sum()))) + "\n"); fh.flush()
        if k % 100 == 0:
            print(f"{k + 1}/{len(order)} patch {p} ({ra:.1f}, {dec:.1f}): {len(counts)} exposures, {nv[ok].sum()} matched, "
                  f"{nv[~ok].sum()} unmatched, {time.time() - tt:.0f}s", flush=True)
    print("done", len(order), "patches", f"{time.time() - tt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
