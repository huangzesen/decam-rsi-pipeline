"""Score an OUT.json against injected truth.

python tools/score_inject.py TRUTH.json OUT.json EXPOSURES.csv [--tol 2] [--top 25] [-v]

A candidate hits body j when its predicted position (orbit.predict_dirs, light time) is within --tol arcsec of the
body's predicted position at >= 3 distinct nights among the exposures where the body was injected.
Each body counts once (first, i.e. highest-p, hit); later hits on the same body are duplicates.
"""
import argparse, csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "$RSI_SANDBOX/r1/gen151/workspace/submission/linker")
import orbit  # noqa: E402

ARC = math.pi / 180 / 3600


def night_of(mjd):
    return np.floor(np.asarray(mjd) + 0.3).astype(int)


def score(truth, cands, rows, tol=2.0, top=25):
    t = np.array([float(r["mjd_mid"]) for r in rows])
    obs = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows])
    nights = night_of(t)
    bodies = truth["bodies"]
    U = np.array([orbit.predict_dirs(b["state"], b["epoch_mjd"], t, obs) for b in bodies])  # (B,N,3)
    cands = sorted(cands, key=lambda c: -float(c.get("p", 0)))
    res = []
    for rank, c in enumerate(cands):
        try:
            d = orbit.predict_dirs(c["state"], float(c["epoch_mjd"]), t, obs)
        except Exception:
            res.append(None); continue
        sep = np.arccos(np.clip(np.einsum("bni,ni->bn", U, d), -1, 1)) / ARC   # (B,N)
        best = None
        for j, b in enumerate(bodies):
            ex = np.array(b["exps"], int)
            if len(ex) == 0:
                continue
            ok = ex[sep[j, ex] < tol]
            nn = len(set(nights[ok].tolist()))
            if nn >= 3 and (best is None or np.median(sep[j, ex]) < best[2]):
                best = (j, nn, float(np.median(sep[j, ex])))
        res.append(best)
    hit_rank = {}
    dup = 0
    for rank, h in enumerate(res):
        if h is None:
            continue
        if h[0] in hit_rank:
            dup += 1
        else:
            hit_rank[h[0]] = (rank, float(cands[rank]["p"]))
    return cands, res, hit_rank, dup


def closest(truth, cands, rows, j):
    """Closest candidate to body j: number of nights within 2", 10", 60" and median sep."""
    t = np.array([float(r["mjd_mid"]) for r in rows])
    obs = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows])
    nights = night_of(t)
    b = truth["bodies"][j]
    ex = np.array(b["exps"], int)
    u = orbit.predict_dirs(b["state"], b["epoch_mjd"], t, obs)
    best = None
    for rank, c in enumerate(cands):
        d = orbit.predict_dirs(c["state"], float(c["epoch_mjd"]), t, obs)
        sep = np.arccos(np.clip(np.sum(u[ex] * d[ex], -1), -1, 1)) / ARC
        m = float(np.median(sep))
        if best is None or m < best["med_sep"]:
            best = dict(rank=rank, p=float(c["p"]), med_sep=m,
                        n2=len(set(nights[ex[sep < 2]].tolist())), n10=len(set(nights[ex[sep < 10]].tolist())),
                        seps=[round(float(s), 1) for s in sep])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth"); ap.add_argument("out"); ap.add_argument("exposures")
    ap.add_argument("--tol", type=float, default=2.0); ap.add_argument("--top", type=int, default=25)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    truth = json.load(open(a.truth))
    try:
        cands = json.load(open(a.out))["candidates"]
    except Exception:
        cands = []
    rows = list(csv.DictReader(open(a.exposures)))
    cands, res, hit_rank, dup = score(truth, cands, rows, a.tol, a.top)
    bodies = truth["bodies"]
    print(f"{a.out}: {len(cands)} candidates, {len(hit_rank)} bodies hit, {dup} duplicate hits")
    for cls in sorted({b["cls"] for b in bodies}):
        idx = [j for j, b in enumerate(bodies) if b["cls"] == cls and len(b["nights"]) >= 3]
        h = [j for j in idx if j in hit_rank]
        ranks = sorted(hit_rank[j][0] for j in h)
        print(f"  class {cls}: {len(h)}/{len(idx)} recovered; ranks {ranks}")
    ntop = sum(1 for r in res[:a.top] if r is not None)
    print(f"  hits in top {a.top}: {ntop}; first-time hits in top {a.top}: "
          f"{sum(1 for j,(r,p) in hit_rank.items() if r < a.top)}")
    if a.v:
        for j, b in enumerate(bodies):
            if len(b["nights"]) < 3:
                continue
            s = f"  body {j:2d} {b['cls']} snr {b['snr']:.1f} r {b['r_au']:.0f} nights {len(b['nights'])} "
            if j in hit_rank:
                s += f"HIT rank {hit_rank[j][0]} p {hit_rank[j][1]:.3f}"
            else:
                c = closest(truth, cands, rows, j) if cands else None
                s += f"miss; closest {c}" if c else "miss"
            print(s)


if __name__ == "__main__":
    main()
