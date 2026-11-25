"""Step 1 re-costed for the target of 2026-11-12: bodies of about 200 km and up (H <= ~7).

Local metadata only, the same exposures and placements as season_inventory.py (its cache is reused). Two arms:
  near arm  30-50 au: the body shows in single exposures; source extraction per exposure, then linking over nights and years
  far arm   beyond 50 au: single exposures are too shallow; stacks, slow motions only
Sizes at albedo 0.1: 200 km is H 6.6, 1,000 km is H 3.1. An exposure's reach r5(H) is the distance at which a body of
that H gives S/N 5 in it (V = H + 5 log10(r (r - 1)) = the exposure's 5-sigma V limit, depth model of groups.py).

Prints and writes linker/footprint/bright_target.json:
  reach      share of exposures that reach a 200 km / 1,000 km body at given distances
  near arm   exposures, 3 deg patches, two-season patch windows of >= 5 nights
  far arm    1 deg cell x season windows (>= 3 nights) whose stack of all exposures reaches a 200 km body beyond 50 au
  known      TNOs with H <= --hmax: how many cross the data, how many are recoverable by each arm, by distance and H
  edges      the window-edge question for these bodies: tiers per window scheme, and >= 3 nights for bodies beyond 50 au

Recoverable by linking = the linker's minimum (known_yield.py): >= 5 nights with a single exposure at S/N >= 5, spanning
>= 10 days, inside one or two consecutive observing seasons, anywhere on the sky (no window, no cell).

Usage, from planet-search/:  ~/anaconda3/bin/python -I linker/footprint/bright_target.py [--cache DIR] [--hmax 7]
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import season_inventory as SI
import numpy as np
import pandas as pd

H200, H1000 = 6.6, 3.1
EXTRACT_S = (10, 15)                 # A100 seconds per exposure before linking (r2 gen020: 39-281 s for 5-20 exposures)


def reach(lim, H):
    return (1 + np.sqrt(1 + 4 * 10 ** ((lim - H) / 5))) / 2


def linkable(d, nmin=5, span=10):
    """Bodies with >= nmin nights spanning >= span days inside one or two consecutive seasons. d: b, bs, day (unique)."""
    g = d.groupby(["b", "bs"])["day"].agg(n="size", lo="min", hi="max").reset_index()
    one = g[(g["n"] >= nmin) & (g["hi"] - g["lo"] >= span)]["b"]
    nxt = g.assign(bs=g["bs"] - 1).rename(columns={"n": "n2", "lo": "lo2", "hi": "hi2"})
    two = g.merge(nxt, on=["b", "bs"])
    two = two[(two["n"] + two["n2"] >= nmin)]["b"]
    return set(one) | set(two)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=os.path.join(SI.LK, "data_real", "noirlab_decam_meta"))
    ap.add_argument("--instcal", default=os.path.join(SI.LK, "data_real", "noirlab_decam_instcal"))
    ap.add_argument("--sbdb", default=os.path.join(SI.LK, "data_real", "sbdb", "tno_20261031.json"))
    ap.add_argument("--fov", type=float, default=1.1)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--hmax", type=float, default=7.0)
    ap.add_argument("--out", default=os.path.join(HERE, "bright_target.json"))
    a = ap.parse_args()
    t_ = time.time()
    X, hb, he, hs, hbs, hr, Hmag, asof = SI.get_data(a)
    ra, dec, day, size, mid, lim = X["ra"], X["dec"], X["day"], X["size"], X["mid"], X["lim"]
    N = ra.size
    out = dict(asof=asof, hmax=a.hmax, exposures=int(N), tb=round(size.sum() / 1e12, 1))

    # ---- reach of single exposures
    r200, r1000 = reach(lim, H200), reach(lim, H1000)
    out["reach"] = dict(km200={str(d): round(float((r200 >= d).mean()), 3) for d in (30, 35, 40, 45, 50)}, km200_median=round(float(np.median(r200)), 1),
                        km1000={str(d): round(float((r1000 >= d).mean()), 3) for d in (60, 80, 100, 110, 130)}, km1000_median=round(float(np.median(r1000)), 1))

    # ---- near arm: exposures that reach a 200 km body at 30 au or more; patches and two-season windows
    c1, lam1 = SI.cell(ra, dec, 1)
    c3, lam3 = SI.cell(ra, dec, 3)
    seas = lambda lam: np.floor((SI.sun_lon(mid) - lam) / 360).astype(np.int64) + 10
    near = r200 >= 30
    ps = pd.DataFrame(dict(c=c3[near], s=seas(lam3)[near], day=day[near])).drop_duplicates().groupby(["c", "s"]).size().rename("n").reset_index()
    nx = ps.assign(s=ps["s"] - 1).rename(columns={"n": "n2"})
    two = ps.merge(nx, on=["c", "s"], how="left").fillna(0)
    two["tot"] = two["n"] + two["n2"]
    hrs = [N_ * s / 3600 for N_ in (int(near.sum()),) for s in EXTRACT_S]
    out["near_arm"] = dict(exposures=int(near.sum()), tb=round(size[near].sum() / 1e12, 1), patches_3deg=int(ps["c"].nunique()),
                           patch_seasons=int(len(ps)), patch_seasons_3_nights=int((ps["n"] >= 3).sum()),
                           two_season_windows_5_nights=int((two["tot"] >= 5).sum()),
                           extraction_a100_h=[round(h) for h in hrs], extraction_node_h=[round(h / 4) for h in hrs])

    # ---- far arm: 1 deg cell x season, >= 3 nights, by how far the stack of all exposures reaches a 200 km body
    inv, nights, nexp, nbytes = SI.windows(c1 * 1000 + seas(lam1), day, size)
    stack = 1.25 * np.log10(np.bincount(inv, weights=10 ** (0.8 * lim), minlength=nights.size))
    best1 = np.full(nights.size, -99.0); np.maximum.at(best1, inv, lim)
    rs, r1 = reach(stack, H200), reach(best1, H200)
    rs1000 = reach(stack, H1000)
    ok = nights >= 3
    far = {}
    for d in (50, 55, 60, 70, 75):
        m = ok & (rs >= d)
        far[str(d)] = dict(windows=int(m.sum()), exposures=int(nexp[m].sum()), tb=round(nbytes[m].sum() / 1e12, 1),
                           reference_rate_a100_h=round(float(SI.cost(nexp[m]).sum())),
                           maps_only_a100_h=[round(float(nexp[m].sum()) * s / 3600) for s in EXTRACT_S])
    out["far_arm"] = dict(cell_seasons_3_nights=int(ok.sum()), by_stack_reach_200km=far,
                          median_stack_reach_200km=round(float(np.median(rs[ok])), 1), median_single_reach_200km=round(float(np.median(r1[ok])), 1),
                          stack_reach_1000km={str(d): int((ok & (rs1000 >= d)).sum()) for d in (110, 130, 150, 200)})

    # ---- known bodies of H <= hmax
    Hh = pd.DataFrame(dict(b=hb, e=he, snr=hs, s2=hs ** 2, bs=hbs, day=day[he], r=hr))
    rmed = Hh.groupby("b")["r"].median()

    def known(sel, label):
        bodies = np.flatnonzero(sel)
        d = Hh[np.isin(Hh["b"].values, bodies)]
        cross = set(d["b"])
        det = d[d["snr"] >= 5][["b", "bs", "day"]].drop_duplicates()
        det7 = d[d["snr"] >= 7][["b", "bs", "day"]].drop_duplicates()
        s3 = det.groupby(["b", "bs"]).size()
        three = set(s3[s3 >= 3].reset_index()["b"])
        link, link7 = linkable(det), linkable(det7)
        ceil = SI.best(d.assign(w=d["bs"]), ["b", "w"])
        stackonly = set(ceil[ceil["rank"] >= 2]["b"]) - link
        res = dict(in_sbdb=int(bodies.size), cross=len(cross), three_nights_one_season_single=len(three), linkable=len(link),
                   linkable_at_snr7=len(link7), stack_only=len(stackonly), either=len(link | stackonly))
        by = {}
        for lo, hi in ((0, 30), (30, 50), (50, 75), (75, 1e9)):
            bb = {b for b in cross if lo <= rmed[b] < hi}
            by[f"{lo}-{int(hi) if hi < 1e8 else 'up'} au"] = dict(cross=len(bb), linkable=len(bb & link), stack_only=len(bb & stackonly))
        res["by_distance"] = by
        print(f"\n{label}: {json.dumps(res)}")
        return res

    out["known"] = {f"H<={a.hmax:g}": known(Hmag <= a.hmax, f"known TNOs, H <= {a.hmax:g}"),
                    "H<=6.6": known(Hmag <= H200, "H <= 6.6 (200 km at albedo 0.1)"),
                    "H<=5.5": known(Hmag <= 5.5, "H <= 5.5"), "H<=4.5": known(Hmag <= 4.5, "H <= 4.5"),
                    "all": known(Hmag < 99, "all TNOs")}

    # ---- the window-edge question for these bodies
    b60 = day // 60
    schemes = {"now": c1 * 1000 + b60, "season": c1 * 1000 + seas(lam1), "3 deg": c3 * 1000 + b60, "3 deg, season": c3 * 1000 + seas(lam3)}

    def edges(Hsub):
        r = {}
        for name, wid in schemes.items():
            inv_, nights_, _, _ = SI.windows(wid, day, size)
            r[name] = SI.tally(SI.body_table(Hsub, inv_, nights_ >= 3))
        r["ceiling"] = SI.tally(SI.best(Hsub.assign(w=Hsub["bs"]), ["b", "w"]))
        return r

    bright = Hh[Hmag[Hh["b"].values] <= a.hmax]
    out["edges"] = {f"H<={a.hmax:g}": edges(bright), "beyond 50 au, any H": edges(Hh[Hh["r"] >= 50]),
                    f"beyond 50 au, H<={a.hmax:g}": edges(bright[bright["r"] >= 50])}
    json.dump(out, open(a.out, "w"), indent=1)

    print("\nreach:", json.dumps(out["reach"]))
    print("\nnear arm:", json.dumps(out["near_arm"]))
    print("\nfar arm:", json.dumps(out["far_arm"]))
    for k, v in out["edges"].items():
        print(f"\nedges, {k}")
        for name, s in v.items():
            print(f"  {name:14s} 3 nights {s['three_nights']:6d}  tier1 {s['tier1']:5d}  tier2 {s['tier2']:5d}  tier3 {s['tier3']:5d}  bodies findable {s['bodies_findable']:5d} (tier 1: {s['bodies_tier1']:5d})  bodies with 3 nights {s['bodies_three_nights']:5d}")
    print(f"\nwrote {a.out}  [{time.time() - t_:.0f} s]")


if __name__ == "__main__":
    main()
