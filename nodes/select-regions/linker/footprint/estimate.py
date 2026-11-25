"""Whole-archive cost of the tracklet-less search, from NSC DR2 exposure metadata and this field's measured cost.

Patches: N points on a Fibonacci sphere, N = 1.209 x 41,253 / (pi 1.5^2) (the thinnest covering of the sky by the
1.5 deg cores); each keeps detections within 2.6 deg, as the OF201 field. Seasons: split at solar conjunction for
the patch's ecliptic longitude; windows: two consecutive seasons with exposures (a season with no neighbour alone);
windows with fewer than 5 distinct nights are skipped. For each patch-window the pair proxy (proxy.py) is evaluated
at gamma nodes and divided by the OF201 field's proxy; each band of the field's grid is scaled by that ratio at its
distance, so cost = sum over bands of the field's cost for the band x ratio. Cost per method = the field's GB200 time
per band divided by that method's measured speed-up (see BENCHMARK.md).

Usage: python -I footprint/estimate.py EXPOSURES.npz FIELD.npz OUT.json [--inst c4d] [--n1 n1_pred]
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.spatial import cKDTree
import proxy as P

NODES = np.array([50, 55, 60, 66, 75, 88, 105, 130, 170, 250, 400, 1000.0])     # au
# speed-ups per hypothesis against brute-force verify, GH200 (BENCHMARK.md, jobs <slurm-job-id> / 1109690 / 1109817)
SPEED = {"brute": (1.0, 1.0), "index": (6.7, 4.4), "index+veto": (12.6, 7.5), "index+veto+sharp": (19.7, 11.0),
         "index+veto+sharp+flags": (110, 56)}                          # (60-62 au, 84-89 au)


def unit(ra, dec):
    a, d = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)]


def ecl_lon(ra, dec):
    e = np.radians(23.4393)
    x, y, z = unit(ra, dec).T
    return np.degrees(np.arctan2(y * np.cos(e) + z * np.sin(e), x)) % 360


def windows(tmid, lam):
    """Index sets of the patch's windows: two consecutive seasons (split at conjunction)."""
    k = np.floor((280.460 + 0.9856474 * (tmid - 51544.5) - lam) / 360).astype(int)
    ks = np.unique(k)
    out = []
    for s in ks:
        if s + 1 in ks:
            out.append(np.flatnonzero((k == s) | (k == s + 1)))
        elif s - 1 not in ks:
            out.append(np.flatnonzero(k == s))
    return out


def field_bands(runs):
    """(r_mid, GB200 seconds) per band of the field's 50-1000 au search."""
    out = []
    for f in sorted(glob.glob(runs)):
        if "rank" in f:
            continue
        r = json.load(open(f))
        out.append((float(np.sqrt(r["args"]["rmin"] * r["args"]["rmax"])) if r["args"]["rmax"] < 200 else 250.0, r["gpu_s"]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps"); ap.add_argument("field"); ap.add_argument("out")
    ap.add_argument("--inst", default="c4d"); ap.add_argument("--n1", default="n1_pred")
    ap.add_argument("--cones", default=None, help="fetch_cones.py output: measured counts replace the predicted ones")
    ap.add_argument("--cone-scale", type=float, default=1 / 1.12,
                    help="cone count -> exposure count (check_cones.py: cones read 1.12x the exact counts, median of 138)")
    ap.add_argument("--runs", default="runs/vista/gb[45]_*.json")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--limit", type=int, default=0, help="N patches only (test)")
    a = ap.parse_args()
    tz = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "g_table.npz")); tab = (tz["s"], tz["R"], tz["G"])
    X = np.load(a.exps)
    m = X["inst"] == a.inst
    ra, dec, tmid, night, xe, n1 = X["ra"][m], X["dec"][m], X["tmid"][m], X["night"][m], X["xe"][m], X[a.n1][m]
    good = np.isfinite(ra) & np.isfinite(n1) & (n1 > 0)
    ra, dec, tmid, night, xe, n1 = ra[good], dec[good], tmid[good], night[good], xe[good], n1[good]
    meas = np.full(len(ra), np.nan)                     # measured n1 over the whole disk, from the cones
    if a.cones:
        full = np.flatnonzero(m)[good]                  # index into the exposure table
        pos = np.full(len(X["inst"]), -1); pos[full] = np.arange(len(full))
        tr = cKDTree(unit(ra, dec))
        s_, c_ = np.zeros(len(ra)), np.zeros(len(ra))
        for line in open(a.cones):
            q = json.loads(line)
            area = np.pi * q["radius"] ** 2
            hit = {int(k): v for k, v in q["counts"].items()}
            # exposures well inside whose disk the cone lies (centre within 0.85 deg) and no single detection in it: 0
            near = set(tr.query_ball_point(unit(np.array([q["ra"]]), np.array([q["dec"]]))[0], 2 * np.sin(np.radians(0.85) / 2)))
            for j in near:
                s_[j] += hit.get(int(full[j]), 0) / area * np.pi * P.R0 ** 2; c_[j] += 1
            # exposures farther out only partly cover the cone: left unmeasured here
        meas = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan) * a.cone_scale
        print(f"{np.isfinite(meas).sum()} of {len(ra)} exposures measured; measured / predicted sum over them "
              f"{np.nansum(meas) / n1[np.isfinite(meas)].sum():.3f}", flush=True)
    print(f"{len(ra)} {a.inst} exposures with a predicted count", flush=True)
    # the OF201 field, same model: its 100 window exposures, the same predicted counts
    F = json.load(open(a.field))
    pf = np.array(F["proxy_model"]) if "proxy_model" in F else None
    bands = field_bands(a.runs)
    N = int(round(1.209 * 41252.96 / (np.pi * 1.5 ** 2)))
    i = np.arange(N) + 0.5
    pdec = np.degrees(np.arcsin(1 - 2 * i / N)); pra = np.degrees(np.pi * (1 + 5 ** 0.5) * i) % 360
    tree = cKDTree(unit(ra, dec))
    reach = 2 * np.sin(np.radians(2.6 + P.R0) / 2)
    lam = ecl_lon(pra, pdec)
    res = []
    t0 = time.time()
    for p in range(a.start, N if not a.limit else min(N, a.start + a.limit)):
        idx = np.array(tree.query_ball_point(unit(pra[p:p + 1], pdec[p:p + 1])[0], reach), int)
        if len(idx) < 5:
            continue
        for w in windows(tmid[idx], lam[p]):
            e = idx[w]
            if len(np.unique(night[e])) < 5:
                continue
            ne = n1[e]
            if a.cones:
                mm = np.isfinite(meas[e])
                q = meas[e][mm].sum() / ne[mm].sum() if mm.sum() >= 3 else 1.0
                ne = np.where(mm, meas[e], ne * q)
            pr = P.window_pairs(tab, pra[p], pdec[p], ra[e], dec[e], night[e], ne, xe[e], 1 / NODES)
            res.append(dict(p=p, ra=float(pra[p]), dec=float(pdec[p]), n_exp=int(len(e)), nights=int(len(np.unique(night[e]))),
                            t0=float(tmid[e].min()), t1=float(tmid[e].max()), proxy=[float(x) for x in pr],
                            measured=int(np.isfinite(meas[e]).sum())))
        if p % 500 == 0:
            print(f"patch {p}/{N}: {len(res)} patch-windows, {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(inst=a.inst, n_patches=N, nodes=NODES.tolist(), bands=bands.tolist(), speed=SPEED, windows=res),
              open(a.out, "w"))
    print("wrote", a.out, len(res), "patch-windows", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
