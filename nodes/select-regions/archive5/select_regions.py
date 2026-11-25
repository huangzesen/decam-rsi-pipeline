#!/usr/bin/env python3
"""Choose the ~5 % of DECam's sky most likely to give a new body of ~200 km and up (archive5 run, 2026-11-12).

Unit: a 3 deg x 3 deg cell (Dec bands of 3 deg, RA width 3 deg / cos Dec). A cell's score is the expected number of
chances to catch a bright body that nobody has reported:
  seasons    observing seasons (+-6 months around the cell's opposition) with >= 3 nights of usable exposures,
             plus 0.3 per season with 1-2 nights (they extend arcs across years);
  ecliptic   weight by ecliptic latitude b: 0.7 exp(-b^2/2/8^2) + 0.3 exp(-b^2/2/25^2) (cold + hot/scattered);
  rubin      bodies seen in DECam around 2016 sit ~8 deg east in ecliptic longitude now (40-45 au); a cell whose
             shifted place Rubin has visited is worth less: 1 / (1 + visits within 3 deg of it / 30);
  des        DES searched its own footprint for TNOs: x (1 - 0.8 * DES share of the cell's exposures);
  plane      crowded fields: x 0.6 for |galactic b| < 10 deg.
Cells are ranked by score and taken until their area is 5 % of the area DECam covers; neighbouring chosen cells are
merged into regions (connected cells, at most 6 per region). Each region's exposures are those whose pointing centre
is inside its cells or within --margin deg of them.

Inputs (local metadata only): the bulk manifest (linker/data_real/decam_bulk/{ls6,vista}.tsv: usable, public, complete
InstCal sets), exposure metadata (noirlab_decam_meta), Rubin visits (rubin_visits). Outputs: archive5/regions.json,
archive5/regions_exposures.json (stems per region), archive5/manifests/<id>.tsv (fetch.py format).
"""
import argparse, glob, json, math, os, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--frac", type=float, default=0.05)
ap.add_argument("--margin", type=float, default=0.5)
ap.add_argument("--max-cells", type=int, default=6)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.dirname(HERE)
DR = os.path.join(PS, "linker", "data_real")
R = math.pi / 180

# 1. usable public exposures with complete InstCal sets, and their files
files = defaultdict(list)
for site in ("ls6", "vista"):
    for line in open(os.path.join(DR, "decam_bulk", site + ".tsv")):
        md5, sz, name, tag = line.rstrip("\n").split("\t")
        files[name[:17]].append((md5, int(sz), name))
meta = {}
for f in sorted(glob.glob(os.path.join(DR, "noirlab_decam_meta", "decam_meta_*.json"))):
    d = json.load(open(f)); ix = {k: i for i, k in enumerate(d["fields"])}
    for r in d["rows"]:
        st = r[ix["stamp"]]
        if st in files:
            meta[st] = (r[ix["ra"]], r[ix["dec"]], r[ix["caldat"]], r[ix["dateobs"]], r[ix["proposal"]], r[ix["filter"]], r[ix["exptime"]])
stems = sorted(meta)
print(f"{len(files)} exposures in the bulk manifest, {len(stems)} with metadata")
ra = np.array([meta[s][0] for s in stems]) % 360; dec = np.array([meta[s][1] for s in stems])
night = np.array([meta[s][2] for s in stems]); prop = np.array([meta[s][4] for s in stems])


def mjd_of(caldat):
    y, m, d = (int(x) for x in caldat[:10].split("-"))
    if m <= 2:
        y -= 1; m += 12
    A = y // 100; B = 2 - A + A // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5 - 2400000.5


mjd = np.array([mjd_of(n) for n in night])


# 2. cells
def cell_of(r, d):
    band = np.clip(np.floor((d + 90) / 3).astype(int), 0, 59)
    dc = -90 + 3 * band + 1.5
    n = np.maximum(1, np.floor(360 * np.cos(dc * R) / 3)).astype(int)
    return band, np.floor(r / 360 * n).astype(int) % n, n


band, col, ncol = cell_of(ra, dec)
key = band * 1000 + col


def centre(k):
    b, c = divmod(k, 1000)
    dc = -90 + 3 * b + 1.5
    n = max(1, math.floor(360 * math.cos(dc * R) / 3))
    return (c + 0.5) * 360 / n, dc, 360 / n * 3 * math.cos(dc * R)   # ra, dec, area deg^2


def ecl(rdeg, ddeg):
    e = 23.4393 * R; r, d = rdeg * R, ddeg * R
    sb = math.sin(d) * math.cos(e) - math.cos(d) * math.sin(e) * math.sin(r)
    b = math.asin(sb); l = math.atan2(math.sin(r) * math.cos(e) + math.tan(d) * math.sin(e), math.cos(r))
    return (l / R) % 360, b / R


def ecl2eq(ldeg, bdeg):
    e = 23.4393 * R; l, b = ldeg * R, bdeg * R
    sd = math.sin(b) * math.cos(e) + math.cos(b) * math.sin(e) * math.sin(l)
    d = math.asin(sd); r = math.atan2(math.sin(l) * math.cos(e) - math.tan(b) * math.sin(e), math.cos(l))
    return (r / R) % 360, d / R


def gal_b(rdeg, ddeg):
    aG, dG = 192.85948 * R, 27.12825 * R; r, d = rdeg * R, ddeg * R
    return math.asin(math.sin(d) * math.sin(dG) + math.cos(d) * math.cos(dG) * math.cos(r - aG)) / R


# Rubin visits to date
rv = []
for f in sorted(glob.glob(os.path.join(DR, "rubin_visits", "visits_*.json"))):
    d = json.load(open(f)); ix = {k: i for i, k in enumerate(d["fields"])}
    rv += [(r[ix["fieldRA"]], r[ix["fieldDec"]]) for r in d["rows"] if r[ix["fieldRA"]] is not None]
rv = np.array(rv, float)
rxyz = np.c_[np.cos(rv[:, 1] * R) * np.cos(rv[:, 0] * R), np.cos(rv[:, 1] * R) * np.sin(rv[:, 0] * R), np.sin(rv[:, 1] * R)]
print(f"{len(rv)} Rubin visits")


def n_within(r0, d0, rad):
    v = np.array([math.cos(d0 * R) * math.cos(r0 * R), math.cos(d0 * R) * math.sin(r0 * R), math.sin(d0 * R)])
    return int((rxyz @ v > math.cos(rad * R)).sum())


cells = {}
order = np.argsort(key)
ks, starts = np.unique(key[order], return_index=True)
bounds = list(starts) + [len(order)]
for i, k in enumerate(ks):
    idx = order[bounds[i]:bounds[i + 1]]
    r0, d0, area = centre(int(k))
    lam, beta = ecl(r0, d0)
    # seasons: +-6 months around opposition (Sun at the cell's ecliptic longitude + 180)
    opp0 = 51544.5 + 79.0 + ((lam + 180) % 360) / 360 * 365.25   # an opposition date (mjd) near 2000
    season = np.floor((mjd[idx] - opp0) / 365.25 + 0.5).astype(int)
    nights_by = defaultdict(set)
    for s, n in zip(season, night[idx]):
        nights_by[s].add(n)
    s3 = sum(1 for v in nights_by.values() if len(v) >= 3)
    s1 = sum(1 for v in nights_by.values() if 0 < len(v) < 3)
    seasons = s3 + 0.3 * s1
    w_ecl = 0.7 * math.exp(-beta ** 2 / 2 / 8 ** 2) + 0.3 * math.exp(-beta ** 2 / 2 / 25 ** 2)
    rs, ds = ecl2eq(lam + 8, beta)
    nrub = n_within(rs, ds, 3.0)
    f_rub = 1 / (1 + nrub / 30)
    des = float(np.mean(prop[idx] == "2012B-0001"))
    f_des = 1 - 0.8 * des
    gb = gal_b(r0, d0)
    f_gal = 0.6 if abs(gb) < 10 else 1.0
    cells[int(k)] = dict(ra=r0, dec=d0, area=area, n_exp=len(idx), s3=s3, s1=s1, beta=beta, glat=gb, rubin=nrub, des=des,
                         score=seasons * w_ecl * f_rub * f_des * f_gal,
                         years=sorted({n[:4] for n in night[idx]}))
cover = sum(c["area"] for c in cells.values())
target = a.frac * cover
print(f"DECam covers {len(cells)} cells, {cover:.0f} deg^2; target {a.frac:.0%} = {target:.0f} deg^2")

# 3. take the best cells to the target area
chosen = []; got = 0.0
for k in sorted(cells, key=lambda k: -cells[k]["score"]):
    if got >= target:
        break
    chosen.append(k); got += cells[k]["area"]
cs = set(chosen)


# 4. merge neighbours into regions (flood fill, at most max-cells each, best cells seed first)
def neighbours(k):
    b, c = divmod(k, 1000)
    r0, d0, _ = centre(k)
    out = []
    for db in (-1, 0, 1):
        bb = b + db
        if not 0 <= bb < 60:
            continue
        dc = -90 + 3 * bb + 1.5
        n = max(1, math.floor(360 * math.cos(dc * R) / 3))
        for dr in (-1, 0, 1):
            cc = (math.floor(r0 / 360 * n) + dr) % n
            kk = bb * 1000 + cc
            if kk != k:
                out.append(kk)
    return out


left = set(chosen); regions = []
for k in chosen:
    if k not in left:
        continue
    grp = [k]; left.discard(k); q = [k]
    while q and len(grp) < a.max_cells:
        for nb in sorted(neighbours(q.pop(0)), key=lambda x: -cells.get(x, {"score": 0})["score"]):
            if nb in left and len(grp) < a.max_cells:
                grp.append(nb); left.discard(nb); q.append(nb)
    regions.append(grp)

# 5. exposures per region: centre in a cell, or within margin of a cell's box
regs = []; rexp = {}
xyz = np.c_[np.cos(dec * R) * np.cos(ra * R), np.cos(dec * R) * np.sin(ra * R), np.sin(dec * R)]
os.makedirs(os.path.join(HERE, "manifests"), exist_ok=True)
for i, grp in enumerate(sorted(regions, key=lambda g: -sum(cells[k]["score"] for k in g)), 1):
    rid = f"R{i:03d}"
    sel = np.zeros(len(stems), bool)
    for k in grp:
        r0, d0, _ = centre(k)
        hw_d = 1.5 + a.margin
        hw_r = (180 / max(1, math.floor(360 * math.cos(d0 * R) / 3)) + a.margin / max(0.05, math.cos(d0 * R)))
        dra = (ra - r0 + 180) % 360 - 180
        sel |= (np.abs(dec - d0) <= hw_d) & (np.abs(dra) <= hw_r)
    idx = np.nonzero(sel)[0]
    st = [stems[j] for j in idx]
    tb = sum(sz for s in st for _, sz, _ in files[s]) / 1e12
    v = xyz[idx].mean(0); v /= np.linalg.norm(v)
    rc, dc = (math.atan2(v[1], v[0]) / R) % 360, math.asin(v[2]) / R
    lam, beta = ecl(rc, dc)
    yrs = sorted({night[j][:4] for j in idx})
    seas = sum(cells[k]["s3"] for k in grp)
    regs.append(dict(id=rid, label=f"RA {rc:.0f}, Dec {dc:+.0f} (ecliptic lat {beta:+.0f})", ra=round(rc, 3), dec=round(dc, 3),
                     deg2=round(sum(cells[k]["area"] for k in grp), 1), cells=grp, exposures=len(st), tb=round(tb, 3),
                     first_year=int(yrs[0]), last_year=int(yrs[-1]), seasons=max(cells[k]["s3"] for k in grp),
                     score=round(sum(cells[k]["score"] for k in grp), 2), ecl_lat=round(beta, 1),
                     rubin_visits=sum(cells[k]["rubin"] for k in grp), des_share=round(float(np.mean([cells[k]["des"] for k in grp])), 2),
                     machine="", status="queued", tb_downloaded=0, exp_detected=0, exp_deleted=0, detections=0))
    rexp[rid] = sorted(st, key=lambda s: meta[s][3])
    with open(os.path.join(HERE, "manifests", rid + ".tsv"), "w") as o:
        for s in rexp[rid]:
            for md5, sz, name in sorted(files[s], key=lambda x: x[2]):
                o.write(f"{md5}\t{sz}\t{name}\t{rid}\n")

# 6. machines: balance bytes, best regions first
tot = {"ls6": 0.0, "vista": 0.0}
for r in regs:
    m = min(tot, key=tot.get); r["machine"] = m; tot[m] += r["tb"]
uniq = set(s for v in rexp.values() for s in v)
U = sum(sz for s in uniq for _, sz, _ in files[s]) / 1e12
from datetime import datetime
out = dict(asof=datetime.now().strftime("%Y-%m-%d %H:%M"), rule=__doc__.split("\n\n")[1].strip(), frac=a.frac, margin=a.margin,
           cover_deg2=round(cover), chosen_deg2=round(got), n_cells=len(chosen), unique_exposures=len(uniq), unique_tb=round(U, 2),
           regions=regs)
json.dump(out, open(os.path.join(HERE, "regions.json.tmp"), "w"), indent=1)
os.replace(os.path.join(HERE, "regions.json.tmp"), os.path.join(HERE, "regions.json"))
json.dump(rexp, open(os.path.join(HERE, "regions_exposures.json"), "w"))
print(f"chose {len(chosen)} cells, {got:.0f} deg^2, in {len(regs)} regions; {len(uniq)} exposures, {U:.1f} TB unique "
      f"(sum over regions {sum(r['tb'] for r in regs):.1f} TB); ls6 {tot['ls6']:.1f} TB, vista {tot['vista']:.1f} TB")
for r in regs[:12]:
    print(r["id"], r["label"], f"{r['deg2']} deg2", r["exposures"], "exp", f"{r['tb']} TB", f"{r['first_year']}-{r['last_year']}",
          "s3max", r["seasons"], "rubin", r["rubin_visits"], "des", r["des_share"], "score", r["score"])
