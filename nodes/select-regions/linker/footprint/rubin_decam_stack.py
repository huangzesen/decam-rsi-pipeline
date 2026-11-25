"""Rubin-era TNOs in the DECam archive: which are seen in single exposures, and which only in a stack of one night's
exposures? Ground-truth check for an RSI task (agents write a finder; hidden truth = TNOs first observed >= 2025).

Every TNO in JPL SBDB with first_obs >= --since is placed on every DECam exposure of the NOIRLab archive listing
(845,585 raw exposures 2012-2026, `fetch_archive_pointings.py`; >= --min-exp s, broad bands only), two-body orbit,
Earth-Moon barycentre observer, one epoch per night (caldat + 1.17 d, ~04 UT). A body is inside an exposure within
--radius of its centre. Single-exposure V limit = NSC DR2 median depth95 at 90 s for the band, + 1.25 log10(t / 90),
+ (V - band) for a mean TNO colour, - --margin. A night's stack limit = 1.25 log10 sum 10^(0.8 Vlim) over the exposures
of that night that contain the body (the same rule as survey/bin/model.py).
Classes per body: catalogue (>= --nights single-exposure nights in a window of <= 2 seasons spanning >= --span d),
stack-linkable (same, counting stack nights), one-night (a night whose stack reaches it, but too few nights to link).
Positions back-propagated from arcs of ~1 yr are uncertain far from that arc, so per-body results are reliable only
for DECam nights near the arc; the table splits by |DECam night - first_obs|.

Usage: python -I footprint/rubin_decam_stack.py data_real/noirlab_decam data_real/sbdb/tno_20261031.json OUT.json
"""
import argparse, glob, json, os, sys
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from orbits import elements_to_rv, ecl_to_eq, earth_eq, GM

DEPTH90 = {"g": 23.52, "r": 23.30, "i": 22.79, "z": 22.23, "Y": 21.30, "VR": 23.50}   # NSC DR2 c4d medians
VMINUS = {"g": -0.45, "r": 0.3, "i": 0.6, "z": 0.8, "Y": 0.85, "VR": 0.15}
MJD_1970 = 40587


def mjd(s):
    p = (s[:10].split("-") + ["", ""])[:3]
    y, m, d = (int(x) if x.isdigit() else dflt for x, dflt in zip(p, (2000, 7, 1)))
    return (date(y, m, d) - date(1970, 1, 1)).days + MJD_1970


def windows(n, gap=90):
    n = np.unique(n)
    if n.size == 0:
        return []
    seas = np.split(n, np.flatnonzero(np.diff(n) > gap) + 1)
    out = [(s[0], s[-1], s.size) for s in seas]
    return out + [(a[0], b[-1], a.size + b.size) for a, b in zip(seas[:-1], seas[1:])]


def best(n, span):
    w = [x for x in windows(n) if x[1] - x[0] >= span]
    return max((x[2] for x in w), default=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pointings")
    ap.add_argument("sbdb")
    ap.add_argument("out")
    ap.add_argument("--since", default="2025-01-01")
    ap.add_argument("--until", default="2099-01-01", help="last exposure date (e.g. the end of the 18-month proprietary period)")
    ap.add_argument("--min-exp", type=float, default=30)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--nights", type=int, default=5)
    ap.add_argument("--span", type=float, default=10)
    a = ap.parse_args()

    ra, dec, nt, lim = [], [], [], []
    for f in sorted(glob.glob(os.path.join(a.pointings, "decam_raw_*.json"))):
        for r in json.load(open(f))["rows"]:
            try:
                x, y, d, t, b = float(r[0]), float(r[1]), r[2], float(r[3]), r[4]
            except (TypeError, ValueError, IndexError):
                continue
            if b not in DEPTH90 or not (t >= a.min_exp) or not isinstance(d, str) or len(d) < 10 or d[:10] > a.until:
                continue
            ra.append(x); dec.append(y); nt.append(mjd(d))
            lim.append(DEPTH90[b] + 1.25 * np.log10(t / 90) + VMINUS[b] - a.margin)
    ra, dec, nt, lim = map(np.array, (ra, dec, nt, lim))
    ue = np.c_[np.cos(np.radians(dec)) * np.cos(np.radians(ra)), np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
               np.sin(np.radians(dec))]
    print(f"{nt.size} exposures, {np.unique(nt).size} nights", flush=True)

    S = json.load(open(a.sbdb))
    rows = [dict(zip(S["fields"], r)) for r in S["data"]]
    rows = [r for r in rows if (r["first_obs"] or "") >= a.since and all(r[k] not in (None, "") for k in
            ("a", "e", "i", "om", "w", "ma", "epoch", "H")) and float(r["e"]) < 1 and float(r["a"]) > 0]
    el = {k: np.array([float(r[k]) for r in rows]) for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H")}
    t0 = el["epoch"] - 2400000.5
    nmot = np.sqrt(GM / el["a"] ** 3)
    fo = np.array([mjd(r["first_obs"]) for r in rows])
    lo = np.array([mjd(r["last_obs"] or r["first_obs"]) for r in rows])
    nb = len(rows)
    print(f"{nb} bodies first observed since {a.since}", flush=True)

    cosr = np.cos(np.radians(a.radius))
    rec = [[] for _ in range(nb)]          # (night, n_exp inside, V, single lim, stack lim)
    order = np.argsort(nt, kind="stable")
    un, st = np.unique(nt[order], return_index=True)
    bd = np.r_[st, order.size]
    for k, n in enumerate(un):
        idx = order[bd[k]:bd[k + 1]]
        tm = n + 1.17
        r, _ = elements_to_rv(el["a"], el["e"], np.radians(el["i"]), np.radians(el["om"]), np.radians(el["w"]),
                              np.radians(el["ma"]) + nmot * (tm - t0))
        r = ecl_to_eq(r)
        g = r - earth_eq(np.array([tm]))[0]
        dist = np.linalg.norm(g, axis=1)
        inside = (g / dist[:, None]) @ ue[idx].T > cosr          # (nb, nexp)
        hit = np.flatnonzero(inside.any(1))
        if hit.size == 0:
            continue
        V = el["H"][hit] + 5 * np.log10(np.linalg.norm(r[hit], axis=1) * dist[hit])
        for j, b in enumerate(hit):
            L = lim[idx][inside[b]]
            rec[b].append((int(n), int(L.size), float(V[j]), float(L.max()),
                           float(1.25 * np.log10(np.sum(10 ** (0.8 * L))))))

    res = []
    for b in range(nb):
        h = rec[b]
        sn = [x[0] for x in h if x[2] < x[3]]
        kn = [x[0] for x in h if x[2] < x[4]]
        near = [x for x in h if abs(x[0] - fo[b]) <= 730]
        res.append(dict(name=rows[b]["full_name"].strip(), H=float(el["H"][b]), arc_d=int(lo[b] - fo[b]),
                        V=round(float(np.median([x[2] for x in h])), 2) if h else None,
                        nights_in=len(h), single_nights=len(set(sn)), stack_nights=len(set(kn)),
                        max_nexp=max((x[1] for x in h), default=0),
                        cat=best(sn, a.span) >= a.nights, stk=best(kn, a.span) >= a.nights, one=bool(set(kn) - set(sn)),
                        near_single=len({x[0] for x in near if x[2] < x[3]}),
                        near_stack=len({x[0] for x in near if x[2] < x[4]})))
    json.dump(dict(args=vars(a), bodies=res), open(a.out, "w"))

    def row(lbl, s):
        cat = sum(x["cat"] for x in s)
        stk = sum(x["stk"] and not x["cat"] for x in s)
        one = sum(x["one"] and not x["stk"] and not x["cat"] for x in s)
        print(f"{lbl:28s} {len(s):5d} | inside a DECam exposure {sum(x['nights_in'] > 0 for x in s):5d} | "
              f"catalogue-linkable {cat:4d} | stack-linkable only {stk:4d} | one stacked night only {one:4d}")
    row("all since " + a.since, res)
    row("  H <= 5.5", [x for x in res if x["H"] <= 5.5])
    row("  H 5.5-7", [x for x in res if 5.5 < x["H"] <= 7])
    row("  H > 7", [x for x in res if x["H"] > 7])
    near = [x for x in res if x["near_single"] or x["near_stack"]]
    print(f"within 2 yr of first_obs: {sum(x['near_single'] > 0 for x in res)} seen in a single exposure, "
          f"{sum(x['near_stack'] > 0 and x['near_single'] == 0 for x in res)} only in a night stack, "
          f"{sum(x['near_stack'] >= 3 for x in res)} with >= 3 stacked nights")
    ns = np.array([x["max_nexp"] for x in res if x["nights_in"]])
    print("max exposures on one night containing the body: median", np.median(ns), "90th pct", np.percentile(ns, 90))


if __name__ == "__main__":
    main()
