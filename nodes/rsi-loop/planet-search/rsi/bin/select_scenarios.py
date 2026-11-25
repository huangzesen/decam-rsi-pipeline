"""Pick DECam field-windows for the planet-search RSI task: public broad-band exposures (2023-01 to --until), grouped
by pointing cell (--cell deg) and --window-day bins, ranked by the known TNOs each shows on >= 3 nights.

Truth = every TNO in the JPL SBDB snapshot (two-body orbits, as in linker/footprint/rubin_decam_stack.py); "Rubin era" =
first_obs >= 2025. A body counts in a field-window when it is inside >= 3 nights of its exposures and the night's stack
reaches it (NSC DR2 median depth95 at 90 s, + 1.25 log10(t/90), TNO colours, - 0.3 mag); "stack-only" = fewer than 3 of
those nights see it in a single exposure. Size = exposures x --gb-per-exp (InstCal image + weight + mask, measured
image 0.32 GB). Greedy pick by (known + --stack-weight x stack-only) per GB until --budget-gb. Rubin-era bodies count only in
windows within --near-days of their first observation (back-propagated short arcs are imprecise further away).

Usage: python -I select_scenarios.py POINTINGS_DIR SBDB.json OUT.json [--budget-gb 1000]
"""
import argparse, glob, json, os, sys
from collections import defaultdict
from datetime import date
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "linker"))
import numpy as np
from orbits import elements_to_rv, ecl_to_eq, earth_eq, GM

D90 = {"g": 23.52, "r": 23.30, "i": 22.79, "z": 22.23, "Y": 21.30, "VR": 23.50}
VM = {"g": -0.45, "r": 0.3, "i": 0.6, "z": 0.8, "Y": 0.85, "VR": 0.15}


def mjd(s):
    p = (s[:10].split("-") + ["", ""])[:3]
    y, m, d = (int(x) if x.isdigit() else v for x, v in zip(p, (2000, 7, 1)))
    return (date(y, m, d) - date(1970, 1, 1)).days + 40587


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pointings"); ap.add_argument("sbdb"); ap.add_argument("out")
    ap.add_argument("--since", default="2023-01-01"); ap.add_argument("--until", default="2025-04-30")
    ap.add_argument("--cell", type=float, default=1.0); ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--gb-per-exp", type=float, default=0.6); ap.add_argument("--budget-gb", type=float, default=1000)
    ap.add_argument("--min-nights", type=int, default=4)
    ap.add_argument("--near-days", type=int, default=730, help="Rubin-era bodies count only this close to first_obs")
    ap.add_argument("--stack-weight", type=float, default=2.0)
    a = ap.parse_args()

    E = []
    for f in sorted(glob.glob(os.path.join(a.pointings, "decam_raw_*.json"))):
        for r in json.load(open(f))["rows"]:
            try:
                x, y, d, t, b = float(r[0]), float(r[1]), r[2], float(r[3]), r[4]
            except (TypeError, ValueError, IndexError):
                continue
            if b in D90 and t >= 30 and isinstance(d, str) and a.since <= d[:10] <= a.until:
                E.append((x, y, mjd(d), t, b, D90[b] + 1.25 * np.log10(t / 90) + VM[b] - 0.3))
    ra, dec, nt, et, band, lim = (np.array([e[k] for e in E]) for k in range(6))
    ue = np.c_[np.cos(np.radians(dec)) * np.cos(np.radians(ra)), np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
               np.sin(np.radians(dec))]
    j = np.floor(dec / a.cell).astype(int)
    nra = np.maximum(np.round(360 * np.cos(np.radians((j + 0.5) * a.cell)) / a.cell).astype(int), 1)
    i = np.floor(ra / 360 * nra).astype(int) % nra
    win = (nt - mjd(a.since)) // a.window_days
    key = j * 100000 + i * 100 + win
    print(f"{nt.size} exposures, {np.unique(key).size} field-windows", flush=True)

    S = json.load(open(a.sbdb))
    rows = [dict(zip(S["fields"], r)) for r in S["data"]]
    rows = [r for r in rows if all(r[k] not in (None, "") for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H"))
            and float(r["e"]) < 1 and float(r["a"]) > 0]
    el = {k: np.array([float(r[k]) for r in rows]) for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H")}
    rubin = np.array([(r["first_obs"] or "") >= "2025-01-01" for r in rows])
    fo = np.array([mjd(r["first_obs"] or "2000-01-01") for r in rows])
    t0, nmot = el["epoch"] - 2400000.5, np.sqrt(GM / el["a"] ** 3)
    cosr = np.cos(np.radians(1.0))

    # (field-window, body) -> {night: (single seen, stack lim list)}
    seen = defaultdict(lambda: defaultdict(list))
    order = np.argsort(nt, kind="stable")
    un, st = np.unique(nt[order], return_index=True)
    bd = np.r_[st, order.size]
    for k, n in enumerate(un):
        idx = order[bd[k]:bd[k + 1]]
        tm = n + 1.17
        r, _ = elements_to_rv(el["a"], el["e"], np.radians(el["i"]), np.radians(el["om"]), np.radians(el["w"]),
                              np.radians(el["ma"]) + nmot * (tm - t0))
        g = ecl_to_eq(r) - earth_eq(np.array([tm]))[0]
        dist = np.linalg.norm(g, axis=1)
        inside = (g / dist[:, None]) @ ue[idx].T > cosr
        for b in np.flatnonzero(inside.any(1)):
            V = el["H"][b] + 5 * np.log10(np.linalg.norm(r[b]) * dist[b])
            for e in idx[inside[b]]:
                seen[(int(key[e]), int(b))][int(n)].append((V, lim[e]))

    fw = defaultdict(lambda: dict(known=0, rubin=0, stack_only=0, bodies=[]))
    for (kk, b), nights in seen.items():
        if rubin[b] and min(nights) < fo[b] - a.near_days:      # short Rubin arcs: only near the arc is the truth precise
            continue
        good = [n for n, L in nights.items() if L[0][0] < 1.25 * np.log10(np.sum(10 ** (0.8 * np.array([x[1] for x in L]))))]
        if len(good) < 3:
            continue
        single = [n for n in good if any(V < l for V, l in nights[n])]
        f = fw[kk]
        f["known"] += 1; f["rubin"] += int(rubin[b]); f["stack_only"] += int(len(single) < 3)
        f["bodies"].append(rows[b]["pdes"])
    out = []
    for kk in np.unique(key):
        m = key == kk
        nn = np.unique(nt[m]).size
        if nn < a.min_nights or kk not in fw:
            continue
        f = fw[kk]
        gb = m.sum() * a.gb_per_exp
        e0 = np.flatnonzero(m)[0]
        out.append(dict(key=int(kk), ra=float(np.median(ra[m])), dec=float(np.median(dec[m])),
                        ra_lo=float(i[e0] * 360 / nra[e0]), ra_hi=float((i[e0] + 1) * 360 / nra[e0]),
                        dec_lo=float(j[e0] * a.cell), dec_hi=float((j[e0] + 1) * a.cell),
                        mjd0=int(nt[m].min()), mjd1=int(nt[m].max()), n_exp=int(m.sum()), n_nights=int(nn),
                        bands="".join(sorted(set(band[m]))), gb=round(gb, 1), known=f["known"], rubin=f["rubin"],
                        stack_only=f["stack_only"], value=(f["known"] + a.stack_weight * f["stack_only"]) / gb, bodies=f["bodies"]))
    out.sort(key=lambda x: -x["value"])
    pick, tot = [], 0.0
    for x in out:
        if tot + x["gb"] > a.budget_gb:
            continue
        pick.append(x); tot += x["gb"]
    json.dump(dict(args=vars(a), candidates=out, picked=[x["key"] for x in pick]), open(a.out, "w"), indent=0)
    s = lambda k: sum(x[k] for x in pick)
    print(f"{len(out)} field-windows with >= {a.min_nights} nights and >= 1 known TNO on >= 3 stacked nights")
    print(f"picked {len(pick)}: {tot:.0f} GB, {s('n_exp')} exposures, known TNOs {s('known')} (Rubin era {s('rubin')}), "
          f"stack-only {s('stack_only')}")
    for x in pick[:15]:
        print(f"  ra {x['ra']:6.1f} dec {x['dec']:6.1f} mjd {x['mjd0']}-{x['mjd1']} exp {x['n_exp']:4d} nights {x['n_nights']:3d} "
              f"{x['bands']:6s} {x['gb']:6.1f} GB known {x['known']:3d} rubin {x['rubin']:3d} stack-only {x['stack_only']:3d}")


if __name__ == "__main__":
    main()
