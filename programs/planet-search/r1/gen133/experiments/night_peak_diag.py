"""Diagnostic: how many injected bodies have a night-seed peak within TOL arcsec on >= 3 nights.
usage: python -I night_peak_diag.py TRUTH.json PEAKS.npz [TOL]
PEAKS.npz is written by stackd/finder.py when TNO_NS_DUMP=path is set (u: unit vectors, z, night index, tN)."""
import json, sys, math
import numpy as np
from scipy.spatial import cKDTree

truth = json.load(open(sys.argv[1])); D = np.load(sys.argv[2])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
u, nk, tN, z = D["u"], D["night"], D["tN"], D["z"]
ARC = math.pi / 180 / 3600


def radec2u(ra, dec):
    ra = np.radians(ra); dec = np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


ks = np.unique(nk)
trees = {k: cKDTree(u[nk == k]) for k in ks}
print("tN", np.round(tN, 2).tolist(), "peaks per night", [int((nk == k).sum()) for k in ks])
res = {}
for b in truth["bodies"]:
    UU = radec2u(np.array(b["ra"]), np.array(b["dec"]))
    hits = tot = 0
    for n in b["nights"]:
        k = int(np.argmin(np.abs(np.floor(tN + 0.3) - n)))
        if k not in trees:
            continue
        tot += 1
        d, _ = trees[k].query(UU, k=1)
        if (d / ARC <= tol).any():   # any of the body's positions (bodies move arcmin/night, so no cross-night aliasing)
            hits += 1
    res.setdefault(b["cls"], []).append((hits, tot))
for c in sorted(res):
    h = np.array(res[c])
    print(c, "n", len(h), ">=3 nights with peak:", int((h[:, 0] >= 3).sum()),
          "hits/nights:", " ".join(f"{a}/{t}" for a, t in h))
