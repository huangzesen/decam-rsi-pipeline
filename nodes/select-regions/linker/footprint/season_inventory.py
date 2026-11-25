"""Inventory of the DECam archive by sky patch x observing season (step 1 of the DECam-only plan, 2026-11-12).

Local metadata only: the archive's exposure list, the InstCal file list and JPL SBDB elements. No image, no download,
no cluster. Two questions:

(a) Window edges. DECAM_PLAN's window is the exposures of one 1 deg pointing cell in one fixed 60-day bin. A body can
    lose its third night at a bin edge (time) or to a neighbouring cell (space). Every known TNO is placed on every
    exposure (two-body orbit from its osculating elements, Earth-Moon barycentre observer, inside when within --fov
    of the pointing centre; V = H + 5 log10(r delta); S/N from the depth model of groups.py). Then the same exposures
    are cut into windows in several ways and each body's nights and predicted tier are counted per window:
      now        1 deg cell x fixed 60-day bin
      half-step  1 deg cell x 60-day bins at two offsets (0 and 30 days): every exposure is in two windows
      season     1 deg cell x observing season (conjunction to conjunction at the cell's ecliptic longitude)
      L deg      L deg cell (L = 2, 3) x fixed 60-day bin, and x season
      ceiling    everything a body crosses in one of its observing seasons, whatever the cell
    Unit of the body counts: a body-season = one body in one of its observing seasons, with its best window.
    Tier rule (known_bodies.py): 1 = >= 3 nights with a single exposure at S/N >= 5; 2 = >= 3 nights whose night
    stack reaches 5; 3 = >= 3 nights and the stack of all exposures reaches 5; 0 = >= 3 nights, stack below 5.
    "3n3" = >= 3 nights whose night stack reaches S/N 3 (what r2's missed bodies lacked).
(b) Dense / sparse split. Dense = a 1 deg cell has >= 3 nights within 15 days, each with >= 2 exposures; the dense
    window is the cell's exposures from the first to the last night of such a run (runs chained while the next
    multi-exposure night of a run is within 15 days). Sparse = every other exposure.

Only exposures that are usable (g r i z Y VR, >= 30 s), public on the grouping's date and have a complete InstCal set
are counted, as the slices do. Cost model as groups.py: 0.2276 A100-hours per window of <= 20 exposures, 0.0209 per
exposure above; it is untested for windows longer than 45 days or wider than one pointing.

Usage, from planet-search/:  ~/anaconda3/bin/python -I linker/footprint/season_inventory.py [--cache DIR]
Output: linker/footprint/season_inventory.json and a printed report.
"""
import argparse, datetime, glob, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")                              # the Mac is shared: a few threads only
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LK = os.path.dirname(HERE)
PS = os.path.dirname(LK)
sys.path.insert(0, LK)
sys.path.insert(0, os.path.join(PS, "groups", "kit"))
from orbits import elements_to_rv, ecl_to_eq, earth_eq, GM
import make_groups as M

BANDS = ("g", "r", "i", "z", "Y", "VR")
D90 = {"g": 23.52, "r": 23.30, "i": 22.79, "z": 22.23, "Y": 21.30, "VR": 23.50}
VM = {"g": -0.45, "r": 0.3, "i": 0.6, "z": 0.8, "Y": 0.85, "VR": 0.15}
D0 = datetime.date(2012, 9, 1)
MJD0 = 56171.0                       # 2012-09-01
EPS = np.radians(23.43928)
FLAT, SLOPE = 0.2276, 0.0209         # A100-hours: per window of <= 20 exposures, per exposure above


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def ecl_lon(u):
    """Ecliptic longitude (deg, 0-360) of equatorial unit vectors."""
    return np.degrees(np.arctan2(np.cos(EPS) * u[:, 1] + np.sin(EPS) * u[:, 2], u[:, 0])) % 360


def sun_lon(mjd):
    """Mean longitude of the Sun, degrees, not wrapped (within 2 deg of the true longitude)."""
    return 280.460 + 0.9856474 * (np.asarray(mjd, float) - 51544.5)


def load(meta, instcal, asof):
    e2s = {}
    for y in (2012, 2013):
        f = os.path.join(instcal, f"instcal_{y}.json")
        if os.path.exists(f):
            for row in json.load(open(f))["rows"]:
                if len(row) > 4 and row[4] is not None:
                    e2s[int(row[4])] = row[0][:17]
    ra, dec, day, expt, band, tsec, stamp = [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(meta, "decam_meta_*.json"))):
        for r in json.load(open(f))["rows"]:
            if r[0] is None or r[1] is None or not r[2] or r[4] not in BANDS or r[3] is None or float(r[3]) < 30:
                continue
            d = (datetime.date.fromisoformat(r[2][:10]) - D0).days
            if d < 0 or (r[9] or "9999")[:10] > asof:
                continue
            stamp.append(r[10] or e2s.get(r[7]))
            ra.append(float(r[0])); dec.append(float(r[1])); day.append(d); expt.append(float(r[3])); band.append(r[4])
            try:
                t = datetime.datetime.fromisoformat((r[6] or "")[:19])
                tsec.append((t - datetime.datetime(2012, 9, 1)).total_seconds())
            except ValueError:
                tsec.append(d * 86400.0 + 43200)
    ra, dec, day, expt, tsec = map(np.array, (ra, dec, day, expt, tsec))
    lim = np.array([D90[b] + VM[b] for b in band]) + 1.25 * np.log10(expt / 90) - 0.3
    inst = M.load_instcal(instcal, {s for s in stamp if s})
    size = np.array([sum(inst[s][p][2] for p in "iwd") if s in inst else 0 for s in stamp], float)
    return dict(ra=ra, dec=dec, day=day.astype(np.int64), expt=expt, tsec=tsec, lim=lim, size=size,
                mid=MJD0 + tsec / 86400 + expt / 2 / 86400)


def cell(ra, dec, L):
    """Equal-area cells L deg on a side, by pointing centre (L = 1 is groups.py's cell). Returns id and centre longitude."""
    j = np.floor(dec / L).astype(np.int64)
    nra = np.maximum(np.round(360 / L * np.cos(np.radians((j + 0.5) * L))).astype(np.int64), 1)
    i = np.floor(ra / 360 * nra).astype(np.int64) % nra
    lam = ecl_lon(unit((i + 0.5) / nra * 360, (j + 0.5) * L))
    return (j + 200) * 100000 + i, lam


def place(X, sbdb, fov):
    """Every TNO on every exposure: arrays b, e, snr, body-season."""
    S = json.load(open(sbdb))
    rows = [dict(zip(S["fields"], r)) for r in S["data"]]
    rows = [r for r in rows if all(r[k] not in (None, "") for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H"))
            and float(r["e"]) < 1 and float(r["a"]) > 0]
    el = {k: np.array([float(r[k]) for r in rows]) for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H")}
    t0, nmot = el["epoch"] - 2400000.5, np.sqrt(GM / el["a"] ** 3)
    cosf = np.cos(np.radians(fov))
    ue = unit(X["ra"], X["dec"])
    order = np.argsort(X["day"], kind="stable")
    un, start = np.unique(X["day"][order], return_index=True)
    bounds = np.r_[start, order.size]
    B, E, SN, BS, RH = [], [], [], [], []
    for k in range(un.size):
        ix = order[bounds[k]:bounds[k + 1]]
        tm = float(X["mid"][ix].mean())
        r, _ = elements_to_rv(el["a"], el["e"], np.radians(el["i"]), np.radians(el["om"]), np.radians(el["w"]),
                              np.radians(el["ma"]) + nmot * (tm - t0))
        r = ecl_to_eq(r)
        g = r - earth_eq(tm)
        dist = np.linalg.norm(g, axis=1)
        gu = g / dist[:, None]
        b, kk = np.nonzero((gu @ ue[ix].T) > cosf)
        if not b.size:
            continue
        rh = np.linalg.norm(r, axis=1)
        V = el["H"][b] + 5 * np.log10(rh[b] * dist[b])
        B.append(b); E.append(ix[kk]); SN.append(5 * 10 ** (0.4 * (X["lim"][ix[kk]] - V)))
        BS.append(np.floor((sun_lon(tm) - ecl_lon(gu[b])) / 360).astype(np.int64)); RH.append(rh[b])
    return (np.concatenate(B), np.concatenate(E), np.concatenate(SN), np.concatenate(BS), np.concatenate(RH),
            el["H"], [r["pdes"] for r in rows])


def windows(wid, day, size):
    """Per-exposure window index, and per window: nights, exposures, bytes."""
    uw, inv = np.unique(wid, return_inverse=True)
    nights = np.bincount(np.unique(inv * 100000 + day) // 100000, minlength=uw.size)
    return inv, nights, np.bincount(inv, minlength=uw.size), np.bincount(inv, weights=size, minlength=uw.size)


def cost(n):
    return np.where(n <= 20, FLAT, SLOPE * n)


RANK = {1: 4, 2: 3, 3: 2, 0: 1}


def body_table(H, inv, ok):
    """Best window of every body-season: rank (4 tier 1, 3 tier 2, 2 tier 3, 1 three nights but too faint), 3n3 flag."""
    d = H[ok[inv[H["e"].values]]].copy()
    d["w"] = inv[d["e"].values]
    return best(d, ["b", "w"])


def best(d, keys):
    n = d.groupby(keys + ["day"]).agg(mx=("snr", "max"), ss=("s2", "sum"), bs=("bs", "min")).reset_index()
    n["k1"] = n["mx"] >= 5; n["k2"] = n["ss"] >= 25; n["k3"] = n["ss"] >= 9
    w = n.groupby(keys).agg(nn=("mx", "size"), n1=("k1", "sum"), n2=("k2", "sum"), n3=("k3", "sum"), tot=("ss", "sum"),
                            bs=("bs", "min")).reset_index()
    w = w[w["nn"] >= 3]
    rank = np.where(w["n1"] >= 3, 4, np.where(w["n2"] >= 3, 3, np.where(w["tot"] >= 25, 2, 1)))
    w = w.assign(rank=rank, f3=(w["n3"] >= 3) & (rank >= 2))
    return w.groupby(["b", "bs"]).agg(rank=("rank", "max"), f3=("f3", "max"), nn=("nn", "max")).reset_index()


def tally(t):
    r = t["rank"].values
    return dict(three_nights=int(len(t)), tier1=int((r == 4).sum()), tier2=int((r == 3).sum()), tier3=int((r == 2).sum()),
                tier0=int((r == 1).sum()), findable=int((r >= 2).sum()), weight=int((r == 4).sum() + 2 * (r == 3).sum() + 4 * (r == 2).sum()),
                tier3_3n3=int(((r == 2) & t["f3"].values).sum()), bodies_findable=int(t[r >= 2]["b"].nunique()), bodies_tier1=int(t[r == 4]["b"].nunique()),
                bodies_three_nights=int(t["b"].nunique()))


def get_data(a):
    """Exposures with files and the placements of every TNO on them, from the cache when there is one."""
    asof = json.load(open(os.path.join(HERE, "groups.json")))["asof"]
    cf = a.cache and os.path.join(a.cache, "season_inventory_cache.npz")
    if cf and os.path.exists(cf):
        Z = np.load(cf, allow_pickle=True)
        X = {k[2:]: Z[k] for k in Z.files if k.startswith("x_")}
        hb, he, hs, hbs, hr, Hmag = (Z[k] for k in ("hb", "he", "hs", "hbs", "hr", "Hmag"))
    else:
        X = load(a.meta, a.instcal, asof)
        n_all = X["ra"].size
        keep = X["size"] > 0
        X = {k: v[keep] for k, v in X.items()}
        X["n_all"] = np.array(n_all)
        hb, he, hs, hbs, hr, Hmag, _ = place(X, a.sbdb, a.fov)
        if cf:
            os.makedirs(a.cache, exist_ok=True)
            np.savez(cf, hb=hb, he=he, hs=hs, hbs=hbs, hr=hr, Hmag=Hmag, **{"x_" + k: v for k, v in X.items()})
    return X, hb, he, hs, hbs, hr, Hmag, asof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=os.path.join(LK, "data_real", "noirlab_decam_meta"))
    ap.add_argument("--instcal", default=os.path.join(LK, "data_real", "noirlab_decam_instcal"))
    ap.add_argument("--sbdb", default=os.path.join(LK, "data_real", "sbdb", "tno_20261031.json"))
    ap.add_argument("--fov", type=float, default=1.1)
    ap.add_argument("--cache", default=None, help="folder for the loaded exposures and placements (reruns in seconds)")
    ap.add_argument("--out", default=os.path.join(HERE, "season_inventory.json"))
    a = ap.parse_args()
    t_ = time.time()
    X, hb, he, hs, hbs, hr, Hmag, asof = get_data(a)
    ra, dec, day, size, mid = X["ra"], X["dec"], X["day"], X["size"], X["mid"]
    N = ra.size
    print(f"{int(X['n_all'])} usable exposures public on {asof}; {N} with a complete InstCal set, {size.sum() / 1e12:.1f} TB, "
          f"{np.unique(day).size} nights; {hb.size} body-exposure placements of {np.unique(hb).size} TNOs  [{time.time() - t_:.0f} s]", flush=True)
    H = pd.DataFrame(dict(b=hb, e=he, snr=hs, s2=hs ** 2, bs=hbs, day=day[he]))
    out = dict(asof=asof, fov=a.fov, exposures=int(N), tb=round(size.sum() / 1e12, 1), placements=int(hb.size),
               bodies_placed=int(np.unique(hb).size), body_seasons=int(len(H.drop_duplicates(["b", "bs"]))))

    # ---- (a) window schemes
    c1, lam1 = cell(ra, dec, 1)
    seas = lambda lam: np.floor((sun_lon(mid) - lam) / 360).astype(np.int64) + 10
    b60, b60h = day // 60, (day + 30) // 60 + 500
    sets = {"now": [c1 * 1000 + b60], "half-step": [c1 * 1000 + b60, c1 * 1000 + b60h], "120-day": [c1 * 1000 + day // 120],
            "season": [c1 * 1000 + seas(lam1)]}
    for L in (2, 3):
        cL, lamL = cell(ra, dec, L)
        sets[f"{L} deg"] = [cL * 1000 + b60]
        sets[f"{L} deg, half-step"] = [cL * 1000 + b60, cL * 1000 + b60h]
        sets[f"{L} deg, season"] = [cL * 1000 + seas(lamL)]
    res, tabs, keepw = {}, {}, {}
    for name, wids in sets.items():
        tt, st = [], dict(windows=0, exposures_processed=0, cost=0.0, le20=0, n21_55=0, n56_1000=0, gt1000=0)
        inwin = np.zeros(N, bool)
        spans, nn_all = [], []
        for wid in wids:
            inv, nights, nexp, nbytes = windows(wid, day, size)
            ok = nights >= 3
            keepw.setdefault(name, []).append((inv, nights, ok))
            inwin |= ok[inv]
            st["windows"] += int(ok.sum()); st["exposures_processed"] += int(nexp[ok].sum()); st["cost"] += float(cost(nexp[ok]).sum())
            st["le20"] += int((nexp[ok] <= 20).sum()); st["n21_55"] += int(((nexp[ok] > 20) & (nexp[ok] <= 55)).sum())
            st["n56_1000"] += int(((nexp[ok] > 55) & (nexp[ok] <= 1000)).sum()); st["gt1000"] += int((nexp[ok] > 1000).sum())
            lo = np.full(nights.size, 10 ** 9); hi = np.zeros(nights.size, np.int64)
            np.minimum.at(lo, inv, day); np.maximum.at(hi, inv, day)
            spans.append((hi - lo)[ok]); nn_all.append(nights[ok])
            tt.append(body_table(H, inv, ok))
            if wid is not wids[0]:                  # offset windows that hold exposures of two fixed bins: the only new ones
                new = ok & (hi // 60 > lo // 60)
                st.update(extra_windows=int(new.sum()), extra_exposures_processed=int(nexp[new].sum()), extra_cost=round(float(cost(nexp[new]).sum())))
        t = pd.concat(tt).groupby(["b", "bs"]).agg(rank=("rank", "max"), f3=("f3", "max"), nn=("nn", "max")).reset_index()
        spans, nn_all = np.concatenate(spans), np.concatenate(nn_all)
        st.update(exposures=int(inwin.sum()), tb=round(size[inwin].sum() / 1e12, 1), cost=round(st["cost"]),
                  median_nights=float(np.median(nn_all)), median_span=float(np.median(spans)), p90_span=float(np.percentile(spans, 90)),
                  span_gt45=int((spans > 45).sum()), **tally(t))
        res[name], tabs[name] = st, t
    t = best(H.assign(w=H["bs"]), ["b", "w"])                   # ceiling: all of a body's exposures in one of its seasons
    res["ceiling"], tabs["ceiling"] = tally(t), t

    # how many of today's windows gain nights in each re-formed scheme
    invA, nightsA, okA = keepw["now"][0]
    for name in sets:
        if name == "now":
            continue
        bestn = np.zeros(nightsA.size, np.int64)
        for inv, nights, ok in keepw[name]:
            p = pd.DataFrame(dict(a=invA, t=inv)).drop_duplicates()
            m = p.assign(n=nights[p["t"].values]).groupby("a")["n"].max()
            bestn[m.index.values] = np.maximum(bestn[m.index.values], m.values)
        gain = (bestn - nightsA)[okA]
        res[name].update(windows_gain=int((gain > 0).sum()), windows_gain_share=round(float((gain > 0).mean()), 3),
                         median_gain=float(np.median(gain[gain > 0])) if (gain > 0).any() else 0.0)

    # where the lost third nights go: body-seasons with >= 3 nights at the ceiling and fewer in today's windows
    key = lambda t: set(zip(t["b"].values, t["bs"].values))
    ceil3, now3 = key(tabs["ceiling"]), key(tabs["now"])
    lost = ceil3 - now3
    time_fix, space_fix = key(tabs["season"]) & lost, key(tabs["3 deg"]) & lost
    out["lost"] = dict(ceiling_three_nights=len(ceil3), now_three_nights=len(now3), lost=len(lost),
                       back_with_season=len(time_fix), back_with_3deg=len(space_fix), back_with_either=len(time_fix | space_fix),
                       back_with_3deg_season=len(key(tabs["3 deg, season"]) & lost))
    out["schemes"] = res

    # ---- (b) dense / sparse
    def dense_split(min_gap_min):
        d = pd.DataFrame(dict(c=c1, day=day, t=X["tsec"]))
        g = d.groupby(["c", "day"])["t"].agg(["size", "min", "max"]).reset_index()
        g = g[(g["size"] >= 2) & ((g["max"] - g["min"]) >= min_gap_min * 60)].sort_values(["c", "day"])
        cc, dd = g["c"].values, g["day"].values
        dn = np.zeros(cc.size, bool)
        tri = (cc[2:] == cc[:-2]) & (dd[2:] - dd[:-2] <= 15)
        for s in range(3):
            dn[s:cc.size - 2 + s] |= tri
        cc, dd = cc[dn], dd[dn]
        new = np.r_[True, (cc[1:] != cc[:-1]) | (dd[1:] - dd[:-1] > 15)]
        ep = np.cumsum(new) - 1
        first = pd.Series(dd).groupby(ep).min().values; last = pd.Series(dd).groupby(ep).max().values
        ecell = pd.Series(cc).groupby(ep).first().values
        # exposures of the cell from the first to the last night of the run
        E = pd.DataFrame(dict(c=ecell, first=first, last=last, ep=np.arange(first.size)))
        x = pd.DataFrame(dict(c=c1, day=day, i=np.arange(N))).merge(E, on="c")
        x = x[(x["day"] >= x["first"]) & (x["day"] <= x["last"])]
        wid = np.full(N, -1, np.int64)
        wid[x["i"].values] = x["ep"].values
        return wid

    arms = {}
    for label, gap in (("dense", 0), ("dense, second look >= 15 min later", 15)):
        wid = dense_split(gap)
        m = wid >= 0
        inv, nights, nexp, nbytes = windows(wid[m], day[m], size[m])
        full = np.full(N, 0, np.int64); full[m] = inv + 1
        okf = np.r_[False, np.ones(nights.size, bool)]
        tb = body_table(H, full, okf)
        lo = np.full(nights.size, 10 ** 9); hi = np.zeros(nights.size, np.int64)
        np.minimum.at(lo, inv, day[m]); np.maximum.at(hi, inv, day[m])
        arms[label] = dict(windows=int(nights.size), exposures=int(m.sum()), tb=round(size[m].sum() / 1e12, 1), cost=round(float(cost(nexp).sum())),
                           median_exposures=float(np.median(nexp)), median_nights=float(np.median(nights)), median_span=float(np.median(hi - lo)),
                           le20=int((nexp <= 20).sum()), n21_55=int(((nexp > 20) & (nexp <= 55)).sum()), n56_1000=int(((nexp > 55) & (nexp <= 1000)).sum()),
                           gt1000=int((nexp > 1000).sum()), in_todays_windows=int((m & okA[invA]).sum()), **tally(tb))
        if gap == 0:
            sp = ~m
            cs = c1[sp] * 1000 + seas(lam1)[sp]
            inv, nights, nexp, nbytes = windows(cs, day[sp], size[sp])
            Hs = H[sp[H["e"].values]]
            single = Hs[Hs["snr"] >= 5]                          # what a single-detection catalogue holds
            per = single.groupby(["b", "bs"])["day"].nunique()
            per_all = H[H["snr"] >= 5].groupby(["b", "bs"])["day"].nunique()
            byb = single.groupby("b").agg(n=("day", "nunique"), s=("bs", "nunique"))
            bya = H[H["snr"] >= 5].groupby("b").agg(n=("day", "nunique"), s=("bs", "nunique"))
            invs, nts, nxs, _ = windows((c1 * 1000 + b60)[sp], day[sp], size[sp])          # the sparse exposures cut as today's windows
            oks = nts >= 3
            full = np.zeros(N, np.int64); full[sp] = invs + 1
            as_now = dict(windows=int(oks.sum()), exposures=int(nxs[oks].sum()), cost=round(float(cost(nxs[oks]).sum())),
                          **tally(body_table(H, full, np.r_[False, oks])))
            arms["sparse"] = dict(as_todays_windows=as_now, exposures=int(sp.sum()), tb=round(size[sp].sum() / 1e12, 1), cell_seasons=int(nights.size),
                                  cell_seasons_3_nights=int((nights >= 3).sum()), cell_seasons_5_nights=int((nights >= 5).sum()),
                                  in_todays_windows=int((sp & okA[invA]).sum()),
                                  body_seasons_3_nights_single=int((per >= 3).sum()), bodies_3_nights_single=int(per[per >= 3].reset_index()["b"].nunique()),
                                  bodies_5_nights_2_seasons=int(((byb["n"] >= 5) & (byb["s"] >= 2)).sum()),
                                  all_exposures_body_seasons_3_nights_single=int((per_all >= 3).sum()),
                                  all_exposures_bodies_3_nights_single=int(per_all[per_all >= 3].reset_index()["b"].nunique()),
                                  all_exposures_bodies_5_nights_2_seasons=int(((bya["n"] >= 5) & (bya["s"] >= 2)).sum()))
    # the dense runs with neighbouring cells merged: in each 3 deg cell, runs whose dates overlap become one window that
    # takes every exposure of the 3 deg cell between its first and last night
    wid = dense_split(0)
    m = wid >= 0
    c3, _ = cell(ra, dec, 3)
    iv = pd.DataFrame(dict(c=c3[m], ep=wid[m], day=day[m])).groupby(["c", "ep"])["day"].agg(["min", "max"]).reset_index().sort_values(["c", "min"])
    rows, cur = [], None
    for c, lo, hi in zip(iv["c"].values, iv["min"].values, iv["max"].values):
        if cur is not None and c == cur[0] and lo <= cur[2]:
            cur[2] = max(cur[2], hi)
        else:
            cur = [c, lo, hi]; rows.append(cur)
    E3 = pd.DataFrame(rows, columns=["c", "first", "last"]); E3["ep"] = np.arange(len(E3))
    x = pd.DataFrame(dict(c=c3, day=day, i=np.arange(N))).merge(E3, on="c")
    x = x[(x["day"] >= x["first"]) & (x["day"] <= x["last"])]
    full = np.zeros(N, np.int64); full[x["i"].values] = x["ep"].values + 1
    m3 = full > 0
    inv, nights, nexp, nbytes = windows(full[m3], day[m3], size[m3])
    span = (E3["last"] - E3["first"]).values
    arms["dense, 3 deg patches"] = dict(windows=int(nights.size), exposures=int(m3.sum()), tb=round(size[m3].sum() / 1e12, 1), cost=round(float(cost(nexp).sum())),
                                        median_exposures=float(np.median(nexp)), median_nights=float(np.median(nights)), median_span=float(np.median(span)),
                                        span_gt45=int((span > 45).sum()), le20=int((nexp <= 20).sum()), n21_55=int(((nexp > 20) & (nexp <= 55)).sum()),
                                        n56_1000=int(((nexp > 55) & (nexp <= 1000)).sum()), gt1000=int((nexp > 1000).sum()),
                                        **tally(body_table(H, full, np.r_[False, np.ones(nights.size, bool)])))
    out["arms"] = arms
    json.dump(out, open(a.out, "w"), indent=1)

    # ---- report
    print(f"\nbody-seasons with the body on >= 1 exposure: {out['body_seasons']} ({out['bodies_placed']} bodies)")
    hdr = f"{'scheme':17s} {'windows':>8s} {'exposures':>10s} {'TB':>6s} {'A100-h':>7s} {'nights':>6s} {'span':>5s} {'>45d':>6s} {'>55exp':>6s} | {'3 nights':>8s} {'tier1':>6s} {'tier2':>6s} {'tier3':>6s} {'(3n3)':>6s} {'weight':>7s} {'bodies':>6s} | {'gain':>6s}"
    print(hdr)
    for name, s in res.items():
        if name == "ceiling":
            print(f"{name:17s} {'':>8s} {'':>10s} {'':>6s} {'':>7s} {'':>6s} {'':>5s} {'':>6s} {'':>6s} | {s['three_nights']:8d} {s['tier1']:6d} {s['tier2']:6d} {s['tier3']:6d} {s['tier3_3n3']:6d} {s['weight']:7d} {s['bodies_findable']:6d} |")
            continue
        print(f"{name:17s} {s['windows']:8d} {s['exposures']:10d} {s['tb']:6.1f} {s['cost']:7d} {s['median_nights']:6.0f} {s['median_span']:5.0f} {s['span_gt45']:6d} {s['n56_1000'] + s['gt1000']:6d} | "
              f"{s['three_nights']:8d} {s['tier1']:6d} {s['tier2']:6d} {s['tier3']:6d} {s['tier3_3n3']:6d} {s['weight']:7d} {s['bodies_findable']:6d} | {s.get('windows_gain', 0):6d}"
              + (f"   new windows only: {s['extra_windows']} windows, {s['extra_exposures_processed']} exposures, {s['extra_cost']} A100-h" if "extra_cost" in s else ""))
    print("\nlost third nights:", json.dumps(out["lost"]))
    for k, v in arms.items():
        print(f"\n{k}: {json.dumps(v)}")
    print(f"\nwrote {a.out}  [{time.time() - t_:.0f} s]")


if __name__ == "__main__":
    main()
