#!/usr/bin/env python3
"""archive5: one region's single-exposure detections -> the linker's window format (sedna80/build.py's, read by
link_gpu.py), with sources that sit still removed.

1. Load every detection list of the region's exposures (detect.py npz, from one or more DET folders).
2. Static sources: positions in a tangent plane at the region centre, 0.6" cells; a detection is static when detections
   on >= 2 other nights fall in its 3 x 3 block of cells (about 1-2"). Every detection at S/N >= 4.5 counts as a
   witness. A moving body at 30-1,000 au moves > 2" between nights except near a stationary point.
3. Keep: not static, S/N >= --snr (5), not near saturation (flag 1), 0.4 <= sharp <= 2.5 (point-like, no cosmic ray).
4. Seed mask (pairs and clustering start only from these; verification uses every kept detection): flag == 0,
   0.5 <= sharp <= 1.8; an exposure whose seeds exceed 3 x the median and 2,000 gives none (junk-rich exposures made
   the duplicate linkages of the sedna80 pilot).
5. Seasons: +-6 months around the region's opposition; seasons_mjd lists those with exposures.
6. --known POS.json (from known_pos.py): label kept detections within 2" of a known body's position in that exposure
   (det_obj = body index, -1 otherwise) and count each body's nights before and after the static cut.
Writes OUT/<rid>.npz + .json, <rid>_seed.npz, <rid>_labels.npz, <rid>_prep.json (counts).
Usage: python3 -I prep_region.py RID STEMS.json OUT DET [DET ...] [--known POS.json] [--snr 5]
"""
import argparse, glob, json, os, sys, time
import numpy as np

AS = np.pi / 180 / 3600
ap = argparse.ArgumentParser()
ap.add_argument("rid"); ap.add_argument("stems"); ap.add_argument("out"); ap.add_argument("det", nargs="+")
ap.add_argument("--known", default="")
ap.add_argument("--snr", type=float, default=5.0)
ap.add_argument("--cell", type=float, default=0.6)
a = ap.parse_args()
t0 = time.time()
say = lambda *x: print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)
stems = json.load(open(a.stems))
if isinstance(stems, dict):
    stems = stems[a.rid]

# 1. load
cols = {k: [] for k in ("ra", "dec", "snr", "sharp", "flag", "flux", "exp")}
E = []
for s in stems:
    p = None
    for d in a.det:
        q = os.path.join(d, s[4:10], s + ".npz")
        if os.path.exists(q):
            p = q; break
    if p is None:
        continue
    z = np.load(p)
    m = json.loads(str(z["meta"]))
    k = len(E); E.append(m)
    for c in ("ra", "dec", "snr", "sharp", "flag", "flux"):
        cols[c].append(z[c])
    cols["exp"].append(np.full(len(z["ra"]), k, np.int32))
D = {k: np.concatenate(v) for k, v in cols.items()}
ne = len(E)
say(f"{a.rid}: {ne} of {len(stems)} exposures with detections, {len(D['ra']):,} detections")
et = np.array([m["mjd_mid"] for m in E])
night = np.floor(et - 0.5).astype(np.int64)          # CTIO nights change at local noon ~ 0.5 d in MJD... close enough
dnight = night[D["exp"]]

# region centre and tangent plane
ra, dec = np.radians(D["ra"]), np.radians(D["dec"])
v = np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
c = v.mean(0); c /= np.linalg.norm(c)
ra0, dec0 = np.arctan2(c[1], c[0]) % (2 * np.pi), np.arcsin(c[2])
ex = np.array([-np.sin(ra0), np.cos(ra0), 0.0]); ey = np.cross(c, ex)
den = v @ c
x = (v @ ex) / den / AS; y = (v @ ey) / den / AS      # arcsec

# 2. static sources: distinct nights per 3x3 block of cells
cx = np.floor(x / a.cell).astype(np.int64); cy = np.floor(y / a.cell).astype(np.int64)
cx -= cx.min() - 2; cy -= cy.min() - 2
W = int(cy.max() + 3)
cell = cx * W + cy
nu, ninv = np.unique(night, return_inverse=True)
nn = ninv[D["exp"]].astype(np.int64)
cn = np.unique(cell * 65536 + nn)                       # distinct (cell, night)
say(f"{len(cn):,} distinct (cell, night)")
blocks = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        blocks.append(cn + (dx * W + dy) * 65536)
cn9 = np.unique(np.concatenate(blocks)); del blocks
cel9 = cn9 // 65536
uc, cnt = np.unique(cel9, return_counts=True)            # nights seen in each cell's 3x3 block
del cn9, cel9
i = np.searchsorted(uc, cell)
nblk = np.where((i < len(uc)) & (uc[np.minimum(i, len(uc) - 1)] == cell), cnt[np.minimum(i, len(uc) - 1)], 1)
# nblk counts the detection's own night too: static = >= 2 other nights
static = nblk >= 3
say(f"static {static.mean():.1%} of all detections")

# 3. keep
keep = (~static) & (D["snr"] >= a.snr) & ((D["flag"] & 1) == 0) & (D["sharp"] >= 0.4) & (D["sharp"] <= 2.5)
idx = np.flatnonzero(keep)
say(f"kept {len(idx):,} ({len(idx) / max(1, ne):.0f} per exposure)")

# 4. seeds
seed = ((D["flag"][idx] == 0) & (D["sharp"][idx] >= 0.5) & (D["sharp"][idx] <= 1.8))
per = np.bincount(D["exp"][idx][seed], minlength=ne)
med = np.median(per[per > 0]) if (per > 0).any() else 0
junk = (per > 3 * med) & (per > 2000)
seed &= ~junk[D["exp"][idx]]
say(f"seeds {seed.sum():,}; {junk.sum()} junk-rich exposures give none (median {med:.0f} per exposure)")

# 5. seasons
lam = None
eps = np.radians(23.4393)
beta_c = np.arcsin(np.sin(dec0) * np.cos(eps) - np.cos(dec0) * np.sin(eps) * np.sin(ra0))
lam = np.arctan2(np.sin(ra0) * np.cos(eps) + np.tan(dec0) * np.sin(eps), np.cos(ra0)) % (2 * np.pi)
opp0 = 51544.5 + 79.0 + ((np.degrees(lam) + 180) % 360) / 360 * 365.25
sea = np.floor((et - opp0) / 365.25 + 0.5).astype(int)
seasons = []
for s in np.unique(sea):
    t = et[sea == s]
    seasons.append([float(t.min() - 0.5), float(t.max() + 0.5), int(s), int((sea == s).sum()), int(len(np.unique(night[sea == s])))])

# observer positions (heliocentric CTIO, equatorial au)
from astropy.time import Time
from astropy.utils import iers
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "warn"     # UT1-UTC = 0 beyond the bundled table: < 0.5 km in the observer position
from astropy.coordinates import get_body_barycentric, EarthLocation
import astropy.units as u
CTIO = EarthLocation.of_site("ctio") if False else EarthLocation.from_geodetic(-70.80653 * u.deg, -30.16596 * u.deg, 2215 * u.m)
tt = Time(et, format="mjd", scale="utc")
obs = (get_body_barycentric("earth", tt).xyz.to(u.au).value.T - get_body_barycentric("sun", tt).xyz.to(u.au).value.T
       + CTIO.get_gcrs(tt).cartesian.xyz.to(u.au).value.T)

# 6. known bodies
det_obj = -np.ones(len(idx), np.int32); kinfo = []
if a.known and os.path.exists(a.known):
    K = json.load(open(a.known))            # {"bodies": [{name, H, pos: {stem: [ra, dec]}}]}
    stem_k = {m["stem"]: k for k, m in enumerate(E)}
    from scipy.spatial import cKDTree
    order = np.argsort(D["exp"], kind="stable")
    st = np.searchsorted(D["exp"][order], np.arange(ne + 1))
    for bi, b in enumerate(K["bodies"]):
        before = set(); after = set(); n_after = 0
        for s, (bra, bdec) in b["pos"].items():
            k = stem_k.get(s)
            if k is None:
                continue
            j = order[st[k]:st[k + 1]]
            dd = np.hypot((D["ra"][j] - bra) * np.cos(np.radians(bdec)), D["dec"][j] - bdec) * 3600
            hit = j[dd < 2.0]
            if len(hit):
                before.add(int(night[k]))
                h = hit[np.isin(hit, idx)]
                if len(h):
                    after.add(int(night[k]))
                    det_obj[np.searchsorted(idx, h)] = bi; n_after += len(h)
        kinfo.append(dict(name=b["name"], H=b.get("H"), exposures=len(b["pos"]), nights_detected=len(before),
                          nights_after_cut=len(after), dets_kept=n_after))
    say("known bodies:", [(k["name"], k["nights_detected"], k["nights_after_cut"]) for k in kinfo])

os.makedirs(a.out, exist_ok=True)
f = os.path.join(a.out, a.rid)
# elements of the labelled bodies (linker evaluate(): in core, in distance range); ecliptic, radians, M0 at t0 (mjd)
EL = [tuple(b["el"]) for b in (K["bodies"] if kinfo else [])] if kinfo else []
if not EL:
    EL = [(100.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(et[0]))]
objs = {f"obj_{n}": np.array([e[j] for e in EL]) for j, n in enumerate(("a", "e", "inc", "Om", "om", "M0", "t0"))}
filt = np.array([m["band"][:2].strip() for m in E])
np.savez_compressed(f + ".npz", det_ra=D["ra"][idx], det_dec=D["dec"][idx], det_exp=D["exp"][idx], det_obj=det_obj,
                    det_mag=(-2.5 * np.log10(np.clip(D["flux"][idx], 1e-3, None)) + np.array([m["magzero"] if m["magzero"] == m["magzero"] else 30 for m in E])[D["exp"][idx]]).astype(np.float32),
                    det_filt=filt[D["exp"][idx]], det_snr=D["snr"][idx], det_ndet=np.ones(len(idx), np.int8),
                    exp_t=et, exp_ra=np.radians([m["ra0"] if m["ra0"] == m["ra0"] else 0 for m in E]),
                    exp_dec=np.radians([m["dec0"] if m["dec0"] == m["dec0"] else 0 for m in E]), exp_obs=obs,
                    exp_name=np.array([m["stem"] for m in E]), **objs)
np.savez_compressed(f + "_seed.npz", seed=seed)
np.savez_compressed(f + "_labels.npz", det_obj=det_obj)
span_deg = float(np.degrees(np.arccos(np.clip(v @ c, -1, 1)).max()))
json.dump(dict(wid=a.rid, ra0=float(ra0), dec0=float(dec0), seasons_mjd=[s[:2] for s in seasons], n_exp=ne,
               core_deg=round(span_deg, 2)), open(f + ".json", "w"))
json.dump(dict(rid=a.rid, exposures=ne, exposures_listed=len(stems), detections=int(len(D["ra"])), static_frac=float(static.mean()),
               kept=int(len(idx)), seeds=int(seed.sum()), junk_exposures=int(junk.sum()), seasons=seasons,
               radius_deg=round(span_deg, 2), known=kinfo, seconds=round(time.time() - t0, 1)), open(f + "_prep.json", "w"), indent=1)
say("done")
