"""Share of DECam's archive exposures that sit in a window the reference solver can search: >= 3 distinct nights on
the same sky within a few weeks (rsi/REFERENCE_SOLVER.md, "What to tailor", item 4).

Two measures, both from the archive pointing list (data_real/noirlab_decam, every raw science exposure):

1. Per exposure: among usable exposures whose centres lie within R deg of this one, the most distinct nights in any
   W-day span that contains this exposure's night, and the exposures in that span. Usable = g r i z Y VR, >= 30 s,
   with coordinates (the task's definition, rsi/bin/select_scenarios.py).
2. Field-windows as the task builds them (1 deg pointing cell x 60-day bin): how many have >= 3 / >= 4 nights, by
   size and by sky class.

Sky classes by the pointing centre: "tested" = within 30 deg of the ecliptic and >= 40 deg from the Galactic plane
(where the 10 sealed fields lie); "crowded" = |b| < 15 deg or in the Magellanic Clouds; "other" = the rest.

Usage: ~/anaconda3/bin/python -I footprint/window_share.py data_real/noirlab_decam [--json footprint/window_share.json]
"""
import argparse, datetime, glob, json, os
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("decam")
ap.add_argument("--json")
ap.add_argument("--radii", type=float, nargs="+", default=[0.25, 0.5, 1.0])
ap.add_argument("--spans", type=int, nargs="+", default=[15, 30, 60])
ap.add_argument("--snap", type=float, default=0.05, help="deg; centres closer than this are one pointing")
a = ap.parse_args()

BANDS = ("g", "r", "i", "z", "Y", "VR")
D0 = datetime.date(2012, 9, 1)
SIZES = ((1, 20, "<= 20 (tested)"), (21, 55, "21-55"), (56, 1000, "56-1000 (coadd mode)"), (1001, 10 ** 9, "> 1000 (deep)"))


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def sky_class(u):
    """0 tested, 1 crowded, 2 other, from unit vectors (equatorial J2000)."""
    eps = np.radians(23.43928)
    beta = np.degrees(np.arcsin(np.clip(-np.sin(eps) * u[:, 1] + np.cos(eps) * u[:, 2], -1, 1)))
    b = np.degrees(np.arcsin(np.clip(u @ unit(192.85948, 27.12825)[0], -1, 1)))
    lmc = u @ unit(80.89, -69.76)[0] > np.cos(np.radians(8.0))
    smc = u @ unit(13.19, -72.83)[0] > np.cos(np.radians(5.0))
    crowded = (np.abs(b) < 15) | lmc | smc
    tested = (np.abs(beta) <= 30) & (np.abs(b) >= 40) & ~crowded
    return np.where(tested, 0, np.where(crowded, 1, 2)), beta, b


CLS = ("tested sky (|ecl lat| <= 30, |gal lat| >= 40)", "crowded (|gal lat| < 15, Magellanic Clouds)", "other")


def sliding_best(key, W):
    """key[:, s] = value of the span starting at day s; out[:, d] = max over spans that contain day d (s in d-W+1..d)."""
    out = key.copy()
    for k in range(1, W):
        np.maximum(out[:, k:], key[:, :-k], out=out[:, k:])
    return out


# self-check of the span logic against brute force
_rng = np.random.default_rng(1)
_p = (_rng.random((3, 200)) < 0.08).astype(np.int64)
for _W in (5, 30):
    _pad = np.zeros((3, 200 + _W), np.int64); _pad[:, :200] = _p
    _cs = np.c_[np.zeros((3, 1), np.int64), np.cumsum(_pad, 1)]
    _nn = _cs[:, _W:] - _cs[:, :-_W]
    _best = sliding_best(_nn, _W)
    for _r in range(3):
        for _d in np.flatnonzero(_p[_r]):
            _ref = max(_p[_r, max(s, 0):s + _W].sum() for s in range(_d - _W + 1, _d + 1))
            assert _best[_r, _d] == _ref, (_W, _r, _d)

# ---- load
tot = dict(rows=0, no_coords=0, other_filter=0, short=0)
ra, dec, day, expt = [], [], [], []
for f in sorted(glob.glob(os.path.join(a.decam, "decam_raw_*.json"))):
    for r in json.load(open(f))["rows"]:
        tot["rows"] += 1
        if r[0] is None or r[1] is None or not r[2]:
            tot["no_coords"] += 1
            continue
        if r[4] not in BANDS:
            tot["other_filter"] += 1
            continue
        if r[3] is None or float(r[3]) < 30:
            tot["short"] += 1
            continue
        ra.append(float(r[0])); dec.append(float(r[1])); expt.append(float(r[3]))
        day.append((datetime.date.fromisoformat(r[2][:10]) - D0).days)
ra, dec, day, expt = np.array(ra), np.array(dec), np.array(day), np.array(expt)
ok = day >= 0
ra, dec, day, expt = ra[ok], dec[ok], day[ok], expt[ok]
tot["usable"] = int(ra.size)
print(tot, flush=True)
ND = int(day.max()) + 1

# ---- pointings (snapped centres) and their nights
jd = np.round(dec / a.snap).astype(np.int64)
ir = np.round(ra * np.cos(np.radians(dec)) / a.snap).astype(np.int64)
_, gid = np.unique(jd * 100000 + ir, return_inverse=True)
G = int(gid.max()) + 1
ue = unit(ra, dec)
gu = np.zeros((G, 3))
np.add.at(gu, gid, ue)
gu /= np.linalg.norm(gu, axis=1)[:, None]
M = sparse.coo_matrix((np.ones(ra.size, np.int32), (gid, day)), shape=(G, ND)).tocsr()   # exposures per pointing-night
M.sum_duplicates()
gcls, gbeta, gb = sky_class(gu)
print(f"{G} pointings, {M.nnz} pointing-nights, {ND} days", flush=True)

res = dict(totals=tot, per_exposure={}, field_windows={}, classes=CLS,
           note="usable = g r i z Y VR, >= 30 s, with coordinates; shares are of usable exposures")
tree = cKDTree(gu)
for R in a.radii:
    pairs = tree.query_pairs(2 * np.sin(np.radians(R) / 2), output_type="ndarray")
    ii = np.r_[pairs[:, 0], pairs[:, 1], np.arange(G)]
    jj = np.r_[pairs[:, 1], pairs[:, 0], np.arange(G)]
    A = sparse.coo_matrix((np.ones(ii.size, np.int32), (ii, jj)), shape=(G, G)).tocsr()
    N = (A @ M).tocsr()                                         # exposures per day within R of each pointing
    for W in a.spans:
        nights = np.zeros(M.nnz, np.int32); nexp = np.zeros(M.nnz, np.int64)
        for g0 in range(0, G, 1500):
            g1 = min(g0 + 1500, G)
            cnt = np.zeros((g1 - g0, ND + W), np.int64)
            cnt[:, :ND] = N[g0:g1].toarray()
            c1 = np.c_[np.zeros((g1 - g0, 1), np.int64), np.cumsum(cnt > 0, 1)]
            c2 = np.c_[np.zeros((g1 - g0, 1), np.int64), np.cumsum(cnt, 1)]
            key = (c1[:, W:] - c1[:, :-W]) * (1 << 24) + np.minimum(c2[:, W:] - c2[:, :-W], (1 << 24) - 1)
            best = sliding_best(key, W)
            sub = M[g0:g1].tocoo()
            v = best[sub.row, sub.col]
            lo, hi = M.indptr[g0], M.indptr[g1]
            # M[g0:g1].tocoo() keeps CSR order, so entries map onto M.data[lo:hi]
            nights[lo:hi] = v >> 24; nexp[lo:hi] = v & ((1 << 24) - 1)
        w = M.data.astype(np.int64)
        grow = np.repeat(np.arange(G), np.diff(M.indptr))
        year = 2012 + ((M.indices + 244) // 365.25).astype(int)          # D0 is day 244 of 2012
        k = f"R{R:g}_W{W}"
        d = dict(radius_deg=R, span_days=W)
        for n in (2, 3, 4, 5, 7, 10):
            d[f"share_ge{n}_nights"] = round(float(w[nights >= n].sum() / w.sum()), 4)
        m3 = nights >= 3
        d["exposures_ge3"] = int(w[m3].sum())
        d["by_class_ge3"] = {CLS[c]: dict(exposures=int(w[m3 & (gcls[grow] == c)].sum()),
                                          of_class=round(float(w[m3 & (gcls[grow] == c)].sum() / max(w[gcls[grow] == c].sum(), 1)), 4),
                                          class_total=int(w[gcls[grow] == c].sum())) for c in range(3)}
        d["by_size_ge3"] = {lab: int(w[m3 & (nexp >= lo_) & (nexp <= hi_)].sum()) for lo_, hi_, lab in SIZES}
        d["tested_sky_and_le55_ge3"] = int(w[m3 & (gcls[grow] == 0) & (nexp <= 55)].sum())
        d["by_year_ge3"] = {int(y): [int(w[m3 & (year == y)].sum()), int(w[year == y].sum())] for y in np.unique(year)}
        res["per_exposure"][k] = d
        print(k, {x: d[x] for x in d if x.startswith("share")}, "tested&<=55:", d["tested_sky_and_le55_ge3"], flush=True)

# ---- field-windows as the task builds them: 1 deg cell x 60-day bin
cell, wd = 1.0, 60
j = np.floor(dec / cell).astype(np.int64)
nra = np.maximum(np.round(360 * np.cos(np.radians((j + 0.5) * cell)) / cell).astype(np.int64), 1)
i = np.floor(ra / 360 * nra).astype(np.int64) % nra
key = (j + 100) * 10 ** 7 + i * 1000 + day // wd
uk, inv = np.unique(key, return_inverse=True)
n_exp = np.bincount(inv)
kn = np.unique(inv.astype(np.int64) * 100000 + day)
n_night = np.bincount((kn // 100000).astype(np.int64), minlength=uk.size)
cu = np.zeros((uk.size, 3)); np.add.at(cu, inv, ue); cu /= np.linalg.norm(cu, axis=1)[:, None]
ccls = sky_class(cu)[0]
span = np.zeros(uk.size, np.int64)
dmin = np.full(uk.size, 10 ** 9); dmax = np.full(uk.size, -1)
np.minimum.at(dmin, inv, day); np.maximum.at(dmax, inv, day)
span = dmax - dmin
fw = dict(cell_deg=cell, bin_days=wd, windows=int(uk.size), exposures=int(n_exp.sum()))
for n in (3, 4):
    m = n_night >= n
    t = dict(windows=int(m.sum()), exposures=int(n_exp[m].sum()), share_of_usable=round(float(n_exp[m].sum() / n_exp.sum()), 4),
             median_exposures=float(np.median(n_exp[m])), median_nights=float(np.median(n_night[m])),
             median_span_days=float(np.median(span[m])))
    t["by_size"] = {lab: dict(windows=int((m & (n_exp >= lo_) & (n_exp <= hi_)).sum()),
                              exposures=int(n_exp[m & (n_exp >= lo_) & (n_exp <= hi_)].sum())) for lo_, hi_, lab in SIZES}
    t["by_class"] = {CLS[c]: dict(windows=int((m & (ccls == c)).sum()), exposures=int(n_exp[m & (ccls == c)].sum())) for c in range(3)}
    e = m & (ccls == 0) & (n_exp <= 55)
    t["tested_sky_and_le55"] = dict(windows=int(e.sum()), exposures=int(n_exp[e].sum()))
    e2 = m & (ccls == 0) & (n_exp <= 20)
    t["tested_sky_and_le20"] = dict(windows=int(e2.sum()), exposures=int(n_exp[e2].sum()))
    e3 = m & (ccls != 1) & (n_exp <= 55)
    t["not_crowded_and_le55"] = dict(windows=int(e3.sum()), exposures=int(n_exp[e3].sum()))
    e4 = m & (ccls == 2) & (n_exp <= 55)
    t["other_sky_and_le55"] = dict(windows=int(e4.sum()), exposures=int(n_exp[e4].sum()))
    e5 = m & (ccls == 1) & (n_exp <= 55)
    t["crowded_and_le55"] = dict(windows=int(e5.sum()), exposures=int(n_exp[e5].sum()))
    fw[f"ge{n}_nights"] = t
    print(f"field-windows >= {n} nights:", json.dumps(t), flush=True)
res["field_windows"] = fw
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
