"""Check the predicted single-detection counts where the cost sits: draw exposures at random from the costliest
patch-windows (estimate.py output, ranked by predicted pairs at 60 au), count their n1 (Data Lab, as
fetch_exposures.py) and compare with n1_pred. Resumable (CSV appended). Prints per window: true / predicted sum and
its square (the pair-count correction).

Usage: python -I footprint/validate_n1.py EXPOSURES.npz EST.json OUT.csv [--windows 30] [--per 8] [--random NMIN NMAX]
"""
import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from scipy.spatial import cKDTree
from estimate import unit
from nsc_fetch import query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps"); ap.add_argument("est"); ap.add_argument("out")
    ap.add_argument("--windows", type=int, default=30); ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--random", nargs=2, type=int, default=None, metavar=("NMIN", "NMAX"),
                    help="draw the windows at random among those with NMIN-NMAX exposures instead of the costliest")
    a = ap.parse_args()
    X = np.load(a.exps)
    m = np.flatnonzero((X["inst"] == "c4d") & np.isfinite(X["ra"]) & np.isfinite(X["n1_pred"]) & (X["n1_pred"] > 0))
    E = json.load(open(a.est))
    W = E["windows"]
    k60 = E["nodes"].index(60) if 60 in E["nodes"] else 2
    top = np.argsort([-w["proxy"][k60] for w in W])[:a.windows]
    if a.random:
        ok = np.flatnonzero([(a.random[0] <= w["n_exp"] < a.random[1]) for w in W])
        top = np.random.default_rng(a.seed + 1).choice(ok, size=min(a.windows, len(ok)), replace=False)
    tree = cKDTree(unit(X["ra"][m], X["dec"][m]))
    reach = 2 * np.sin(np.radians(3.7) / 2)
    rng = np.random.default_rng(a.seed)
    done = {}
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done[(int(r["window"]), r["exposure"])] = float(r["n1"])
    new = not os.path.exists(a.out)
    fh = open(a.out, "a"); wr = csv.writer(fh)
    if new:
        wr.writerow(["window", "exposure", "n1", "n1_pred"]); fh.flush()
    res = []
    for wi in top:
        w = W[wi]
        idx = m[np.array(tree.query_ball_point(unit(np.array([w["ra"]]), np.array([w["dec"]]))[0], reach), int)]
        idx = idx[(X["tmid"][idx] >= w["t0"] - 1e-6) & (X["tmid"][idx] <= w["t1"] + 1e-6)]
        pick = rng.choice(idx, size=min(a.per, len(idx)), replace=False)
        tr, pr = [], []
        for i in pick:
            e = str(X["name"][i])
            if (wi, e) in done:
                n = done[(wi, e)]
            else:
                t0 = time.time()
                txt = query(f"SELECT COUNT(*) AS n FROM nsc_dr2.meas m JOIN nsc_dr2.object o ON m.objectid = o.id "
                            f"WHERE m.exposure = '{e}' AND o.ndet <= 1")
                n = float(txt.strip().splitlines()[-1])
                wr.writerow([wi, e, int(n), round(float(X["n1_pred"][i]), 1)]); fh.flush()
                print(f"  {e} n1 {n:.0f} pred {X['n1_pred'][i]:.0f} {time.time() - t0:.1f}s", flush=True)
            tr.append(n); pr.append(float(X["n1_pred"][i]))
        q = sum(tr) / max(sum(pr), 1)
        res.append((wi, w["ra"], w["dec"], w["n_exp"], q))
        print(f"window {wi} ({w['ra']:.1f}, {w['dec']:.1f}) n_exp {w['n_exp']}: true/pred sum {q:.2f}, pairs x{q * q:.2f}", flush=True)
    qs = np.array([r[4] for r in res])
    print(f"{len(res)} windows: true/pred median {np.median(qs):.2f}, mean of squares {np.mean(qs ** 2):.2f}")


if __name__ == "__main__":
    main()
