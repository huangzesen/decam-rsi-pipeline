"""Which known TNOs could a blind multi-season search of NSC DR2 DECam have found? Every TNO in JPL SBDB is placed on
every DECam exposure (two-body orbit from its osculating elements, geocentric observer, no light time); it counts in
an exposure when it is within --radius of the centre and brighter than the exposure's depth95 minus --margin.
A body is findable when some window of one or two consecutive seasons (detections split at gaps > 90 days) has
>= --nights distinct nights spanning >= --span days, the linker's minimum (>= 5 nights, pairs >= 10 nights apart).
Designation year >= 2021 marks bodies designated after NSC DR2 went public (archive finds keep the year of their first
observation, so they are not in that group).

Usage: python -I footprint/known_yield.py data_real/nsc_exposures/exposures.npz data_real/sbdb/tno_20261031.json OUT.json
       [--radius 1.0] [--margin 0.3] [--nights 5] [--span 10]
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from orbits import elements_to_rv, ecl_to_eq, GM

# TNO colours (V - band), mean g - r ~ 0.8; u is far fainter
VMINUS = {"u": -1.6, "g": -0.45, "r": 0.3, "i": 0.6, "z": 0.8, "Y": 0.85, "VR": 0.15}


def windows(nights, gap=90):
    """(first, last, n distinct nights) of every 1- and 2-season window."""
    n = np.unique(nights)
    if n.size == 0:
        return []
    cut = np.flatnonzero(np.diff(n) > gap) + 1
    seas = np.split(n, cut)
    out = [(s[0], s[-1], s.size) for s in seas]
    out += [(a[0], b[-1], a.size + b.size) for a, b in zip(seas[:-1], seas[1:])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exposures")
    ap.add_argument("sbdb")
    ap.add_argument("out")
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--nights", type=int, default=5)
    ap.add_argument("--span", type=float, default=10.0)
    a = ap.parse_args()

    X = np.load(a.exposures, allow_pickle=True)
    m = X["inst"] == "c4d"
    ra, dec, t, night = np.radians(X["ra"][m]), np.radians(X["dec"][m]), X["tmid"][m], X["night"][m]
    xe, filt, depth = X["xe"][m], X["filt"][m], X["depth95"][m]
    off = np.array([VMINUS.get(f, 9.0) for f in filt])
    lim = depth - a.margin + off                      # faintest V seen in each exposure
    lim[~np.isfinite(lim)] = -99
    ue = np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]

    S = json.load(open(a.sbdb))
    F = S["fields"]
    rows = [dict(zip(F, r)) for r in S["data"]]
    rows = [r for r in rows if all(r[k] not in (None, "") for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H"))
            and float(r["e"]) < 1 and float(r["a"]) > 0]
    nb = len(rows)
    el = {k: np.array([float(r[k]) for r in rows]) for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H")}
    mjd0 = el["epoch"] - 2400000.5
    n_mot = np.sqrt(GM / el["a"] ** 3)
    names = [r["full_name"].strip() for r in rows]
    year = [int(m_.group(1)) if (m_ := re.search(r"(\d{4}) [A-Z]{2}\d*", r["full_name"])) else 0 for r in rows]

    cosr = np.cos(np.radians(a.radius))
    hits = [[] for _ in range(nb)]                    # (night, heliocentric r) per detectable exposure
    nin = np.zeros(nb, int)
    order = np.argsort(night, kind="stable")
    un, start = np.unique(night[order], return_index=True)
    bounds = np.r_[start, order.size]
    for k, nt in enumerate(un):
        idx = order[bounds[k]:bounds[k + 1]]
        tm = t[idx].mean()
        M = np.radians(el["ma"]) + n_mot * (tm - mjd0)
        r, _ = elements_to_rv(el["a"], el["e"], np.radians(el["i"]), np.radians(el["om"]), np.radians(el["w"]), M)
        r = ecl_to_eq(r)
        g = r[:, None, :] - xe[idx][None, :, :]       # (nb, nexp, 3) geocentric
        dist = np.linalg.norm(g, axis=-1)
        cosang = np.einsum("bej,ej->be", g, ue[idx]) / dist
        inside = cosang > cosr
        if not inside.any():
            continue
        rh = np.linalg.norm(r, axis=-1)
        V = el["H"][:, None] + 5 * np.log10(rh[:, None] * dist)
        seen = inside & (V < lim[idx][None, :])
        nin += inside.any(1)
        for b in np.flatnonzero(seen.any(1)):
            hits[b].append((int(nt), float(rh[b])))

    res = []
    for b in range(nb):
        h = hits[b]
        best = (0, 0, 0)
        if h:
            nts = np.array([x[0] for x in h])
            for w in windows(nts):
                if w[1] - w[0] >= a.span and w[2] > best[2]:
                    best = w
        rmean = float(np.mean([x[1] for x in h])) if h else None
        res.append(dict(name=names[b], year=year[b], H=float(el["H"][b]), a=float(el["a"][b]), r=rmean,
                        nights_in=int(nin[b]), nights_seen=len({x[0] for x in h}), best_window=int(best[2]),
                        findable=bool(best[2] >= a.nights)))
    json.dump(dict(args=vars(a), bodies=res), open(a.out, "w"))

    def line(sel, label):
        s = [x for x in res if sel(x)]
        f = [x for x in s if x["findable"]]
        def bin_(lo, hi):
            return sum(1 for x in f if x["r"] is not None and lo <= x["r"] < hi)
        print(f"{label:34s} {len(s):5d} bodies, {sum(1 for x in s if x['nights_seen']):5d} seen at least once, "
              f"{len(f):5d} findable (r < 50: {bin_(0, 50)}, 50-100: {bin_(50, 100)}, >= 100 au: {bin_(100, 1e9)})")

    line(lambda x: True, "all TNOs")
    line(lambda x: x["year"] <= 2020, "designated <= 2020")
    line(lambda x: x["year"] >= 2021, "designated >= 2021")
    line(lambda x: x["year"] >= 2021 and x["H"] <= 5.5, "designated >= 2021, H <= 5.5")
    line(lambda x: x["year"] >= 2021 and x["H"] <= 7, "designated >= 2021, H <= 7")


if __name__ == "__main__":
    main()
