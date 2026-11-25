"""Starter finder (generation 0): single-exposure detection + HelioLinC-style linking. Deliberately simple.

1. Detection: per CCD, subtract a coarse median background, PSF-matched filter (Gaussian of the CCD's FWHM) on the
   inverse-variance-weighted image, S/N >= 5 local maxima, pixel -> RA/Dec with the CCD's WCS.
2. Stationary sources: a detection with another detection within 1 arcsec >= 2.4 h apart is dropped.
3. Linking, for each assumed heliocentric distance R: put every detection at distance R from the Sun along its line of
   sight (at emission time), pair detections from different nights whose implied speed is bound, move each pair to a
   common epoch and hash (position, velocity) into cells; cells holding >= 3 nights become candidates, fitted with
   uniform motion (good to well under an arcsec over weeks).
Writes the candidates found so far before the budget ends.
"""
import argparse, csv, json, math, os, sys, time, warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
warnings.simplefilter("ignore")
try:
    import torch
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
except Exception:
    torch = None

GM = 2.959122082855911e-4      # au^3/day^2
C_AU = 173.1446326846693       # au/day
ARC = math.pi / 180 / 3600
T0 = time.time()


def left(budget):
    return budget - (time.time() - T0)


def matched_filter(img, wt, fwhm):
    """S/N map of a Gaussian PSF at every pixel (img sky-subtracted, wt inverse variance, 0 = bad)."""
    s = fwhm / 2.355
    k = int(math.ceil(3 * s))
    x = np.arange(-k, k + 1, dtype=np.float32)
    g = np.exp(-x ** 2 / (2 * s * s)).astype(np.float32)
    if torch is not None:
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None, None].to(DEV)
        gx, gy = t(g[None, :]), t(g[:, None])
        num = torch.nn.functional.conv2d(torch.nn.functional.conv2d(t(img * wt), gx, padding=(0, k)), gy, padding=(k, 0))
        den = torch.nn.functional.conv2d(torch.nn.functional.conv2d(t(wt), gx * gx, padding=(0, k)), gy * gy, padding=(k, 0))
        snr = num / torch.sqrt(torch.clamp(den, min=1e-30))
        samp = snr.flatten()[:: 97]
        samp = samp[den.flatten()[:: 97] > 0]
        med = torch.median(samp)
        snr = (snr - med) / (1.4826 * torch.median(torch.abs(samp - med)) + 1e-12)   # rescale to the measured noise
        peak = (snr == torch.nn.functional.max_pool2d(snr, 5, 1, 2)) & (snr >= 5)
        ys, xs = torch.nonzero(peak[0, 0], as_tuple=True)
        return xs.cpu().numpy(), ys.cpu().numpy(), snr[0, 0, ys, xs].cpu().numpy()
    from scipy.ndimage import convolve1d, maximum_filter
    num = convolve1d(convolve1d(img * wt, g, 1), g, 0)
    den = convolve1d(convolve1d(wt, g * g, 1), g * g, 0)
    snr = num / np.sqrt(np.maximum(den, 1e-30))
    samp = snr.ravel()[:: 97][den.ravel()[:: 97] > 0]
    med = np.median(samp)
    snr = (snr - med) / (1.4826 * np.median(np.abs(samp - med)) + 1e-12)
    ys, xs = np.nonzero((snr == maximum_filter(snr, 5)) & (snr >= 5))
    return xs, ys, snr[ys, xs]


def detect(row, sdir):
    out = []
    with fits.open(os.path.join(sdir, row["image"]), memmap=False) as hi, \
         fits.open(os.path.join(sdir, row["weight"]), memmap=False) as hw, \
         fits.open(os.path.join(sdir, row["mask"]), memmap=False) as hm:
        for k in range(1, len(hi)):
            h = hi[k].header
            if "CTYPE1" not in h or hi[k].data is None:
                continue
            img = hi[k].data.astype(np.float32)
            wt = hw[k].data.astype(np.float32)
            wt[(hm[k].data != 0) | ~np.isfinite(img) | ~np.isfinite(wt)] = 0
            ny, nx = img.shape
            b = 128
            bg = np.nanmedian(img[: ny // b * b, : nx // b * b].reshape(ny // b, b, nx // b, b), axis=(1, 3))
            full = np.repeat(np.repeat(bg, b, 0), b, 1)
            full = np.pad(full, ((0, ny - full.shape[0]), (0, nx - full.shape[1])), mode="edge")
            img = np.nan_to_num(img - full)
            xs, ys, sn = matched_filter(img, wt, float(h.get("FWHM") or 4.0))
            if xs.size:
                ra, de = WCS(h).all_pix2world(xs, ys, 0)
                out += [(float(r), float(d), float(s)) for r, d, s in zip(ra, de, sn)]
    return out


def unit(ra, de):
    ra, de = np.radians(ra), np.radians(de)
    return np.stack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.scenario, "exposures.csv"))))
    cands = []

    def write():
        cands.sort(key=lambda c: -c["p"])
        json.dump({"candidates": cands[:200]}, open(a.out, "w"))
    write()

    # 1. detections
    D = []          # (ra, dec, snr, t_mid, night, obs xyz)
    per_exp = []
    for row in sorted(rows, key=lambda r: float(r["mjd_mid"])):
        if left(a.budget_s) < 0.45 * a.budget_s:
            break
        t0 = time.time()
        det = detect(row, a.scenario)
        t, o = float(row["mjd_mid"]), [float(row["obs_x"]), float(row["obs_y"]), float(row["obs_z"])]
        D += [(r, d, s, t, math.floor(t - 0.5), *o) for r, d, s in det]
        per_exp.append(time.time() - t0)
        print(f"{row['image']}: {len(det)} detections, {time.time() - t0:.1f} s", flush=True)
    if not D:
        return
    D = np.array(D)
    u = unit(D[:, 0], D[:, 1])
    night = D[:, 4].astype(int)

    # 2. stationary sources
    from scipy.spatial import cKDTree
    tree = cKDTree(u)
    static = np.zeros(len(D), bool)
    for i, j in tree.query_pairs(1.0 * ARC):
        if abs(D[i, 3] - D[j, 3]) > 0.1:                  # same spot >= 2.4 h apart: a body beyond 25 au moves > 1"
            static[i] = static[j] = True
    keep = np.flatnonzero(~static)
    print(f"{len(D)} detections, {keep.size} not stationary", flush=True)
    if keep.size < 3:
        return
    u, t, o, sn, nt = u[keep], D[keep, 3], D[keep, 5:8], D[keep, 2], night[keep]
    tref = float(np.mean(t))

    # 3. linking over assumed distances
    best = {}
    for R in (30, 34, 38, 42, 46, 52, 60, 70, 85, 100, 130):
        if left(a.budget_s) < 20:
            break
        ou = np.sum(o * u, 1)
        rho = -ou + np.sqrt(np.maximum(ou ** 2 - np.sum(o * o, 1) + R * R, 0))
        te = t - rho / C_AU                               # emission time
        r = o + rho[:, None] * u                          # heliocentric position at emission
        vmax = math.sqrt(2 * GM / R) * 1.05
        span = float(te.max() - te.min())
        tr = cKDTree(r)
        pairs = tr.query_pairs(vmax * span + 1e-9, output_type="ndarray")
        if pairs.size == 0:
            continue
        i, j = pairs[:, 0], pairs[:, 1]
        ok = nt[i] != nt[j]
        i, j = i[ok], j[ok]
        dt = te[j] - te[i]
        v = (r[j] - r[i]) / dt[:, None]
        ok = np.linalg.norm(v, axis=1) <= vmax
        i, j, v = i[ok], j[ok], v[ok]
        if i.size == 0:
            continue
        p0 = r[i] + v * (tref - te[i])[:, None]
        cell = R * 4 * ARC                                # ~4 arcsec in position and in position change over the span
        key = np.c_[np.floor(p0 / cell), np.floor(v * max(span, 1) / cell)].astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        order = np.argsort(inv, kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1, order.size]
        for a0, a1 in zip(bounds[:-1], bounds[1:]):
            idx = order[a0:a1]
            if idx.size < 2:
                continue
            dets = np.unique(np.r_[i[idx], j[idx]])
            if np.unique(nt[dets]).size < 3:
                continue
            A = np.c_[np.ones(dets.size), te[dets] - tref]
            sol, *_ = np.linalg.lstsq(A, r[dets], rcond=None)
            res = np.linalg.norm(r[dets] - A @ sol, axis=1) / R / ARC
            good = dets[res < 2.0]
            nn = np.unique(nt[good]).size
            if nn < 3:
                continue
            A = np.c_[np.ones(good.size), te[good] - tref]
            sol, *_ = np.linalg.lstsq(A, r[good], rcond=None)
            k = tuple(sorted(good.tolist()))
            score = nn + 0.01 * float(np.sum(sn[good]))
            if k not in best or best[k]["score"] < score:
                best[k] = dict(score=score, state=[*sol[0].tolist(), *sol[1].tolist()], nights=nn)
        cands[:] = [dict(epoch_mjd=tref, state=b["state"], p=round(1 - 1 / (1 + b["score"]), 4)) for b in best.values()]
        write()
        print(f"R {R} au: {len(pairs)} pairs, {len(best)} candidates so far, {left(a.budget_s):.0f} s left", flush=True)
    write()


if __name__ == "__main__":
    main()
