"""Sort DECam's searchable windows into groups small enough to tailor and run side by side.

Window = usable exposures (g r i z Y VR, >= 30 s) of one 1 deg pointing cell in one 60-day bin, with >= 3 nights (the
task's field-window, as in window_share.py). A window's program is the proposal that took most of its exposures. Three levels:
  family = cadence | sky
    cadence  singles (<= 20 exposures, fewer than half of the nights have a second exposure), revisits (<= 20, half
             or more of the nights have one), rolling (21-55), dense (56-1,000), deep (> 1,000)
    sky      ecliptic (within 30 deg of the ecliptic, >= 40 deg from the Galactic plane: where the reference solver
             was tested), high latitude (other clean sky), crowded (within 15 deg of the plane, Magellanic Clouds)
  parent = family | program, for a program with >= --min-group windows in the family; the rest are "other programs".
  group  = a parent, or one part of it when its production would cost more than --max-cost A100-hours. Parts are cut
           in time order (60-day bin), then right ascension ("other programs": program first), with equal cost, and
           keep >= --min-windows windows. Groups are numbered g001... in the order printed.
Cost model (A100-hours of production, DECAM_PLAN.md section 6): 0.2276 per window of <= 20 exposures (the reference
fills its 900 s budget: 8,193 s on the 10 sealed fields), 0.0209 per exposure for larger windows (untested scaling).
V1 / Vstack = the V magnitude a TNO must beat at 5 sigma in the window's median exposure / in the stack of all of them.

Only exposures public on --asof are used (release date in the archive list); the rest are counted and wait.

Output: OUT.json (families, their parents and groups, statistics, 14 sample window keys per group), OUT_windows.json
(every window with its family, parent, group and group id), OUT_exposures.json (window key -> its exposures: stamp,
exposure number, MJD of the start, exposure time, band, RA, Dec, proposal). PIs come from --pis (proposal -> {pi}), written once from
the archive's DTPI field.

Usage: ~/anaconda3/bin/python -I footprint/groups.py data_real/noirlab_decam_meta footprint/groups.json
"""
import argparse, datetime, glob, json, math, os
from collections import Counter, defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("meta"); ap.add_argument("out")
ap.add_argument("--min-group", type=int, default=50, help="windows a program needs in a family to be its own parent")
ap.add_argument("--max-cost", type=float, default=50, help="A100-hours of production above which a parent is cut into parts")
ap.add_argument("--min-windows", type=int, default=20, help="windows every part keeps (practice + validation need 14)")
ap.add_argument("--asof", default=str(datetime.date.today()), help="use exposures released on or before this date")
ap.add_argument("--instcal", default=None, help="InstCal file list (fetch_archive_instcal.py); default META/../noirlab_decam_instcal")
ap.add_argument("--pis", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "proposal_titles.json"))
a = ap.parse_args()

BANDS = ("g", "r", "i", "z", "Y", "VR")
D90 = {"g": 23.52, "r": 23.30, "i": 22.79, "z": 22.23, "Y": 21.30, "VR": 23.50}     # NSC DR2 median depth at 90 s
VM = {"g": -0.45, "r": 0.3, "i": 0.6, "z": 0.8, "Y": 0.85, "VR": 0.15}             # TNO colour V - band
D0 = datetime.date(2012, 9, 1)
SKY = ("ecliptic", "high latitude", "crowded")
CADENCE = ("singles", "revisits", "rolling", "dense", "deep")


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


ra, dec, day, expt, band, prop, tsec, stamp, expnum = [], [], [], [], [], [], [], [], []
n_held = 0
# raw files of 2012-2013 have old names without the stamp: take it from the InstCal products of the same exposure number
e2s = {}
idir = a.instcal or os.path.join(os.path.dirname(os.path.abspath(a.meta)), "noirlab_decam_instcal")
for y in (2012, 2013):
    f = os.path.join(idir, f"instcal_{y}.json")
    if os.path.exists(f):
        for row in json.load(open(f))["rows"]:
            if len(row) > 4 and row[4] is not None:
                e2s[int(row[4])] = row[0][:17]
for f in sorted(glob.glob(os.path.join(a.meta, "decam_meta_*.json"))):
    for r in json.load(open(f))["rows"]:
        if r[0] is None or r[1] is None or not r[2] or r[4] not in BANDS or r[3] is None or float(r[3]) < 30:
            continue
        d = (datetime.date.fromisoformat(r[2][:10]) - D0).days
        if d < 0:
            continue
        if (r[9] or "9999")[:10] > a.asof:
            n_held += 1
            continue
        stamp.append(r[10] or e2s.get(r[7])); expnum.append(r[7])
        ra.append(float(r[0])); dec.append(float(r[1])); day.append(d); expt.append(float(r[3])); band.append(r[4])
        prop.append(r[5] or "none")
        try:
            t = datetime.datetime.fromisoformat((r[6] or "")[:19])
            tsec.append((t - datetime.datetime(2012, 9, 1)).total_seconds())
        except ValueError:
            tsec.append(d * 86400.0 + 43200)
ra, dec, day, expt, tsec = map(np.array, (ra, dec, day, expt, tsec))
band, prop = np.array(band), np.array(prop)
lim = np.array([D90[b] + VM[b] for b in band]) + 1.25 * np.log10(expt / 90) - 0.3      # V a TNO must beat, 5 sigma
print(f"{ra.size} usable exposures public on {a.asof} ({n_held} more not yet public), {np.unique(prop).size} proposals; "
      f"{sum(x is None for x in stamp)} without a file stamp", flush=True)

j = np.floor(dec).astype(np.int64)
nra = np.maximum(np.round(360 * np.cos(np.radians(j + 0.5))).astype(np.int64), 1)
i = np.floor(ra / 360 * nra).astype(np.int64) % nra
key = (j + 100) * 10 ** 7 + i * 1000 + day // 60
order = np.argsort(key, kind="stable")
uk, start = np.unique(key[order], return_index=True)
bounds = np.r_[start, order.size]
ue = unit(ra, dec)
eps = np.radians(23.43928)
gp, lmc, smc = unit(192.85948, 27.12825)[0], unit(80.89, -69.76)[0], unit(13.19, -72.83)[0]

W, WE = [], {}
for k in range(uk.size):
    idx = order[bounds[k]:bounds[k + 1]]
    nights, per_night = np.unique(day[idx], return_counts=True)
    if nights.size < 3:
        continue
    c = ue[idx].sum(0); c /= np.linalg.norm(c)
    beta = float(np.degrees(np.arcsin(-np.sin(eps) * c[1] + np.cos(eps) * c[2])))
    b = float(np.degrees(np.arcsin(c @ gp)))
    crowded = abs(b) < 15 or c @ lmc > np.cos(np.radians(8)) or c @ smc > np.cos(np.radians(5))
    sky = 2 if crowded else (0 if abs(beta) <= 30 and abs(b) >= 40 else 1)
    pc = Counter(prop[idx]).most_common()
    # longest same-night gap between two exposures (minutes): a second look that shows motion
    gap = 0.0
    for n in nights[per_night >= 2]:
        t = tsec[idx][day[idx] == n]
        gap = max(gap, float(t.max() - t.min()) / 60)
    idx = idx[np.argsort(tsec[idx], kind="stable")]
    WE[int(uk[k])] = [[stamp[x], expnum[x], round(56171 + tsec[x] / 86400, 6), float(expt[x]), str(band[x]),
                       round(float(ra[x]), 5), round(float(dec[x]), 5), str(prop[x])] for x in idx]
    W.append(dict(key=int(uk[k]), n_exp=int(idx.size), n_nights=int(nights.size), span=int(nights.max() - nights.min()),
                  prop=pc[0][0], prop_frac=round(pc[0][1] / idx.size, 2), n_props=len(pc),
                  bands="".join(b_ for b_ in BANDS if (band[idx] == b_).any()), exptime=float(np.median(expt[idx])),
                  rep_frac=round(float((per_night >= 2).mean()), 2), gap_min=round(gap, 1),
                  depth1=round(float(np.median(lim[idx])), 2),
                  depth=round(float(1.25 * np.log10(np.sum(10 ** (0.8 * lim[idx])))), 2),
                  beta=round(beta, 1), b=round(b, 1), ra=round(float(np.degrees(np.arctan2(c[1], c[0])) % 360), 2),
                  dec=round(float(np.degrees(np.arcsin(c[2]))), 2), sky=sky,
                  year=2012 + int((int(nights.min()) + 244) // 365.25),
                  first=str(D0 + datetime.timedelta(days=int(nights.min()))),
                  last=str(D0 + datetime.timedelta(days=int(nights.max())))))
print(f"{len(W)} windows of >= 3 nights, {sum(w['n_exp'] for w in W)} exposures", flush=True)

PI = {}
if os.path.exists(a.pis):
    PI = {k: (v.get("pi") or "").strip() for k, v in json.load(open(a.pis)).items()}


def cadence(w):
    n = w["n_exp"]
    if n > 1000:
        return "deep"
    if n > 55:
        return "dense"
    if n > 20:
        return "rolling"
    return "revisits" if w["rep_frac"] >= 0.5 else "singles"


def cost(w):
    return 0.2276 if w["n_exp"] <= 20 else 0.0209 * w["n_exp"]


for w in W:
    w["cadence"] = cadence(w)
    w["family"] = f"{w['cadence']} | {SKY[w['sky']]}"
nf = Counter((w["family"], w["prop"]) for w in W)
Pm = defaultdict(list)
for w in W:
    own = nf[(w["family"], w["prop"])] >= a.min_group
    w["parent"] = f"{w['family']} | {w['prop'] if own else 'other programs'}"
    Pm[w["parent"]].append(w)
for parent, ws in Pm.items():
    ws.sort(key=lambda w: (w["prop"], w["key"] % 1000, w["ra"]))
    c = np.array([cost(w) for w in ws])
    k = max(1, min(math.ceil(c.sum() / a.max_cost), len(ws) // a.min_windows))
    part = np.minimum(((np.cumsum(c) - c / 2) / (c.sum() / k)).astype(int), k - 1)
    for w, q in zip(ws, part):
        w["part"], w["parts"] = int(q) + 1, k
        w["group"] = parent if k == 1 else f"{parent} | {int(q) + 1}/{k}"

ORDER = {c: k for k, c in enumerate(CADENCE)}


def stats(ws):
    q = lambda f: [float(np.percentile([w[f] for w in ws], p)) for p in (10, 50, 90)]
    ne = sum(w["n_exp"] for w in ws)
    return dict(windows=len(ws), exposures=ne, cost=round(sum(cost(w) for w in ws), 1), tb=round(ne * 0.6 / 1000, 2),
                n_exp=q("n_exp"), n_nights=q("n_nights"),
                span=q("span"), exptime=q("exptime"), depth1=q("depth1"), depth=q("depth"),
                abs_beta=[float(np.percentile([abs(w["beta"]) for w in ws], p)) for p in (10, 50, 90)],
                gap_min=q("gap_min"), three_nights=sum(w["n_nights"] == 3 for w in ws),
                bands=Counter("".join(sorted(w["bands"])) for w in ws).most_common(4),
                years=[min(w["year"] for w in ws), max(w["year"] for w in ws)],
                dates=[min(w["first"] for w in ws), max(w["last"] for w in ws)], north=sum(w["dec"] > 10 for w in ws),
                programs=[[p, PI.get(p, ""), c] for p, c in Counter(w["prop"] for w in ws).most_common(6)],
                n_programs=len(set(w["prop"] for w in ws)))


Fm = defaultdict(list)
for w in W:
    Fm[w["family"]].append(w)
out = []
for fam, ws in Fm.items():
    d = dict(family=fam, **stats(ws))
    Pf = defaultdict(list)
    for w in ws:
        Pf[w["parent"]].append(w)
    d["parents"], d["groups"] = [], []
    for parent, pw in sorted(Pf.items(), key=lambda x: -len(x[1])):
        d["parents"].append(dict(parent=parent, parts=pw[0]["parts"], **stats(pw)))
        G = defaultdict(list)
        for w in pw:
            G[w["part"]].append(w)
        for q in sorted(G):
            gw = G[q]
            g = gw[0]["group"]
            rng = np.random.default_rng(int.from_bytes(g.encode()[-8:].rjust(8, b"x"), "little") % 2 ** 32)
            d["groups"].append(dict(group=g, parent=parent, part=q, parts=gw[0]["parts"], **stats(gw),
                                    sample=[int(gw[x]["key"]) for x in rng.choice(len(gw), min(14, len(gw)), replace=False)]))
    out.append(d)
out.sort(key=lambda x: (ORDER[x["family"].split(" | ")[0]], SKY.index(x["family"].split(" | ")[1])))
gid = {}
for x in out:
    for g in x["groups"]:
        g["id"] = gid[g["group"]] = f"g{len(gid) + 1:03d}"
        g.update(id=g.pop("id"))
for w in W:
    w["id"] = gid[w["group"]]
allg = [g for x in out for g in x["groups"]]
json.dump(dict(made=str(datetime.date.today()), asof=a.asof, not_yet_public=n_held, min_group=a.min_group, max_cost=a.max_cost, min_windows=a.min_windows,
               windows=len(W), exposures=sum(w["n_exp"] for w in W), cost=round(sum(cost(w) for w in W)),
               n_parents=sum(len(x["parents"]) for x in out), n_groups=len(allg), families=out), open(a.out, "w"), indent=1)
json.dump(W, open(a.out.replace(".json", "_windows.json"), "w"))
json.dump(dict(fields=["stamp", "expnum", "mjd_start", "exptime", "band", "ra", "dec", "proposal"], windows=WE),
          open(a.out.replace(".json", "_exposures.json"), "w"))
pc = lambda v: "%.0f / %.0f / %.0f" % tuple(np.percentile(v, (0, 50, 100)))
print(f"{len(out)} families, {sum(len(x['parents']) for x in out)} parents, {len(allg)} groups; production {sum(g['cost'] for g in allg):.0f} A100-h")
print(f"per group (min / median / max): windows {pc([g['windows'] for g in allg])}, A100-h {pc([g['cost'] for g in allg])}, "
      f"TB {'%.2f / %.2f / %.2f' % tuple(np.percentile([g['tb'] for g in allg], (0, 50, 100)))}")
print(f"{'family / parent':46s} {'groups':>6s} {'A100h':>6s} {'win':>6s} {'exp':>7s} {'exp/win':>7s} {'nights':>6s} {'span':>5s} {'t_exp':>5s} {'V1':>5s} {'Vstack':>6s} {'|beta|':>6s} {'N>10':>5s}")
for x in out:
    for lev, y in [(0, x)] + [(1, g) for g in x["parents"]]:
        name = y["family"] if lev == 0 else "   " + y["parent"].split(" | ")[2] + " " + PI.get(y["parent"].split(" | ")[2], "")
        ng = len(x["groups"]) if lev == 0 else y["parts"]
        print(f"{name[:46]:46s} {ng:6d} {y['cost']:6.0f} {y['windows']:6d} {y['exposures']:7d} {y['n_exp'][1]:7.0f} {y['n_nights'][1]:6.0f} {y['span'][1]:5.0f} "
              f"{y['exptime'][1]:5.0f} {y['depth1'][1]:5.1f} {y['depth'][1]:6.1f} {y['abs_beta'][1]:6.0f} {y['north']:5d}")
