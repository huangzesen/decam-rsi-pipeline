"""Runs on Vista (login node, nice): image stamps for candidates, from the archive5 images on scratch.

stdin JSON: {"sid": SID, "half": 50, "cands": {label: [row, ...]}}
For each detection row: a stamp centred on it (det_x/det_y on its CCD), and a reference stamp of the same RA/Dec in
the nearest exposure of another night (|dt| > 0.5 d, same band preferred) that covers it, so a static source would show.
Writes OUT.npz (stamps float32, NaN outside the CCD) and OUT.json (metadata) to the path given as argv[1].
2026-11-13.
"""
import glob, json, os, sys, warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

warnings.simplefilter("ignore", FITSFixedWarning); warnings.simplefilter("ignore", RuntimeWarning)
SCEN = "$TACC_SCRATCH/dbt/scen"
IMG = "$TACC_SCRATCH/archive5/img"
q = json.load(sys.stdin); out = sys.argv[1]
sid, H = q["sid"], int(q.get("half", 50))
L = lambda k: np.load(f"{SCEN}/{sid}/det_{k}.npy", mmap_mode="r")
ra, dec, ex, ccd, X, Y, sn = L("ra"), L("dec"), L("exp"), L("ccd"), L("x"), L("y"), L("snr")
E = json.load(open(f"{SCEN}/{sid}/exposures.json")); E = E if isinstance(E, list) else E["exposures"]
P = np.array([[e["ra0"], e["dec0"]] for e in E], float); T = np.array([e["mjd_mid"] for e in E], float)


def ooi(stem):
    f = glob.glob(f"{IMG}/{stem[4:10]}/{stem}_ooi_*.fits.fz")
    return sorted(f)[-1] if f else None


def cut(data, x, y):
    s = np.full((2 * H + 1, 2 * H + 1), np.nan, np.float32)
    xi, yi = int(round(x)), int(round(y))
    y0, y1, x0, x1 = yi - H, yi + H + 1, xi - H, xi + H + 1
    ny, nx = data.shape
    a, b, c, d = max(y0, 0), min(y1, ny), max(x0, 0), min(x1, nx)
    if a < b and c < d:
        s[a - y0:b - y0, c - x0:d - x0] = data[a:b, c:d]
    return s


def hdu_by_ccd(hl, n):
    for h in hl[1:]:
        if int(h.header.get("CCDNUM", -1)) == n:
            return h
    return None


def cd(h):
    return [[float(h.header.get("CD1_1", 0)), float(h.header.get("CD1_2", 0))], [float(h.header.get("CD2_1", 0)), float(h.header.get("CD2_2", 0))]]


def masked_frac(img_path, n, x, y):
    """Fraction of flagged pixels (ood mask != 0) in the stamp; 1.0 if the mask file is missing."""
    f = img_path.replace("_ooi_", "_ood_")
    if not os.path.exists(f):
        return 1.0
    with fits.open(f, memmap=False) as hl:
        h = hdu_by_ccd(hl, n)
        if h is None:
            return 1.0
        s = cut(h.data.astype(np.float32), x, y)
    return float(np.mean(~np.isfinite(s) | (s != 0)))


def find_ref(r0, d0, e_own, band, own_exps):
    sep = np.degrees(np.arccos(np.clip(np.sin(np.radians(d0)) * np.sin(np.radians(P[:, 1])) + np.cos(np.radians(d0)) *
                                       np.cos(np.radians(P[:, 1])) * np.cos(np.radians(P[:, 0] - r0)), -1, 1)))
    ok = (sep < 1.1) & (np.abs(T - T[e_own]) > 0.5)
    ok[list(own_exps)] = False
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return None
    key = [(E[i]["band"] != band, abs(T[i] - T[e_own])) for i in idx]
    for i in [idx[j] for j in np.lexsort(np.array(key).T[::-1])][:12]:
        f = ooi(E[i]["stem"])
        if not f:
            continue
        with fits.open(f, memmap=False) as hl:
            for h in hl[1:]:
                w = WCS(h.header)
                x, y = (float(v) for v in w.all_world2pix(r0, d0, 0))
                ny, nx = int(h.header.get("ZNAXIS2", h.header.get("NAXIS2"))), int(h.header.get("ZNAXIS1", h.header.get("NAXIS1")))
                if H + 2 < x < nx - H - 2 and H + 2 < y < ny - H - 2:
                    n = int(h.header["CCDNUM"])
                    if masked_frac(f, n, x, y) > 0.02:          # bad amplifier, bleed trail or edge: try the next exposure
                        break
                    return dict(exp=int(i), stem=E[i]["stem"], band=E[i]["band"], mjd=float(T[i]), ccd=n, cd=cd(h),
                                x=float(x), y=float(y), stamp=cut(h.data.astype(np.float32), x, y))
    return None


stamps, meta = [], {}
for lab, rows in q["cands"].items():
    own = {int(ex[r]) for r in rows}
    M = []
    for r in sorted(rows, key=lambda r: T[int(ex[r])]):
        e = int(ex[r]); stem = E[e]["stem"]; f = ooi(stem)
        m = dict(row=int(r), exp=e, stem=stem, band=E[e]["band"], exptime=E[e]["exptime"], mjd=float(T[e]), ccd=int(ccd[r]),
                 x=float(X[r]), y=float(Y[r]), ra=float(ra[r]), dec=float(dec[r]), snr=float(sn[r]), stamp=None, ref=None)
        if f:
            with fits.open(f, memmap=False) as hl:
                h = hdu_by_ccd(hl, int(ccd[r]))
                if h is not None:
                    stamps.append(cut(h.data.astype(np.float32), X[r], Y[r])); m["stamp"] = len(stamps) - 1; m["cd"] = cd(h)
        rf = find_ref(float(ra[r]), float(dec[r]), e, E[e]["band"], own)
        if rf:
            stamps.append(rf.pop("stamp")); rf["stamp"] = len(stamps) - 1; m["ref"] = rf
        M.append(m)
        print(lab, stem, "ok" if m["stamp"] is not None else "NO IMAGE", "ref" if m["ref"] else "no ref", file=sys.stderr, flush=True)
    meta[lab] = M
np.savez_compressed(out + ".npz", stamps=np.array(stamps, np.float32))
json.dump(meta, open(out + ".json", "w"), indent=1)
print("wrote", out, len(stamps), "stamps", file=sys.stderr)
