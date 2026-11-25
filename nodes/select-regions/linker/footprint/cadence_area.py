"""How much sky each camera covered well enough to find a slow-moving body: on a 0.5 deg grid, the most distinct
nights inside any 365-day window (exposure centres within the field radius). Prints the area with >= 1, 3, 5, 10 such
nights, for the whole sky and within 10 deg of the ecliptic, for DECam 2012-2019, DECam 2012-2026 and Rubin.

Usage: python -I footprint/cadence_area.py data_real/noirlab_decam data_real/rubin_visits [--json OUT]
"""
import argparse, datetime, glob, json, os
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("decam")
ap.add_argument("rubin")
ap.add_argument("--json")
a = ap.parse_args()


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def day(s):
    return (datetime.date.fromisoformat(s[:10]) - datetime.date(2000, 1, 1)).days


dra, ddec, dday = [], [], []
for f in sorted(glob.glob(os.path.join(a.decam, "decam_raw_*.json"))):
    for r in json.load(open(f))["rows"]:
        if r[0] is not None and r[1] is not None and r[2]:
            dra.append(r[0]); ddec.append(r[1]); dday.append(day(r[2]))
rra, rdec, rday = [], [], []
for f in sorted(glob.glob(os.path.join(a.rubin, "visits_*.json"))):
    d = json.load(open(f))
    i, j = d["fields"].index("fieldRA"), d["fields"].index("fieldDec")
    for r in d["rows"]:
        if r[i] is not None and r[j] is not None:
            rra.append(r[i]); rdec.append(r[j]); rday.append(day(d["night"]))
dra, ddec, dday = map(np.array, (dra, ddec, dday))
rra, rdec, rday = map(np.array, (rra, rdec, rday))

g_ra = (np.arange(720) + 0.5) * 0.5
g_dec = 90 - (np.arange(360) + 0.5) * 0.5
GR, GD = np.meshgrid(g_ra, g_dec)
cells = unit(GR.ravel(), GD.ravel())
area = (0.25 * np.cos(np.radians(GD))).ravel()
eps = np.radians(23.43928)
beta = np.degrees(np.arcsin(-np.sin(eps) * cells[:, 1] + np.cos(eps) * cells[:, 2]))
near = np.abs(beta) < 10


def best_nights(ra, dec, dd, radius):
    tree = cKDTree(unit(ra, dec))
    idx = tree.query_ball_point(cells, 2 * np.sin(np.radians(radius) / 2), workers=4)
    out = np.zeros(len(cells), int)
    for c, ii in enumerate(idx):
        if not ii:
            continue
        n = np.unique(dd[ii])
        if n.size <= out[c]:
            continue
        hi = np.searchsorted(n, n + 365, side="left")
        out[c] = int((hi - np.arange(n.size)).max())
    return out


res = {}
for name, ra, dec, dd, rad in (("DECam 2012-2019", dra[dday < day("2020-01-01")], ddec[dday < day("2020-01-01")],
                                dday[dday < day("2020-01-01")], 1.0),
                               ("DECam 2012-2026", dra, ddec, dday, 1.0),
                               ("Rubin to 2026-10-30", rra, rdec, rday, 1.75)):
    b = best_nights(ra, dec, dd, rad)
    res[name] = {f"{lab}>={k}": round(float(area[(b >= k) & m].sum())) for lab, m in (("sky", np.ones_like(near)), ("ecl10", near))
                 for k in (1, 3, 5, 10)}
    print(name, res[name], flush=True)
print("area within 10 deg of the ecliptic:", round(float(area[near].sum())))
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
