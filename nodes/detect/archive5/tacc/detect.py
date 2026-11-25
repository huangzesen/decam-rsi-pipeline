#!/usr/bin/env python3
"""Single-exposure detection for the archive5 run: every DECam InstCal exposure (ooi image, oow inverse-variance weight,
ood mask) -> one detection list, OUT/<YYMMDD>/<stem>.npz. No stacking.

Per CCD: background from sigma-clipped 64 x 64 px block medians (bilinear), then a matched filter with a Gaussian PSF
at the CCD's FWHM: flux = conv(r w, K) / conv(w, K^2), error = 1 / sqrt(conv(w, K^2)), r = image - background,
w = weight (0 on masked pixels). Peaks with S/N >= --snr that are the maximum of their 5 x 5 box are kept, with a
sub-pixel centre (3 x 3 centroid of the S/N map) and RA/Dec from the CCD's own WCS (TPV).
Columns: ra dec (deg), flux ferr (image units), snr, x y (0-based px), ccd (CCDNUM), sharp (peak pixel / PSF peak of
the fitted flux: ~1 for a star or a moving body, >> 1 for a cosmic ray), flag (1 near saturation, 2 masked pixels in
the 5 x 5 box, 4 within 12 px of the edge). Exposure values: stem expnum mjd_mid band exptime magzero fwhm airmass
ra0 dec0 wcscal.

--watch: keep scanning IMG for exposures whose three files are complete (no .part) and not yet done, until
IMG/../log/FETCH_DONE exists and nothing is left. Images are left in place (scratch purges them).
Usage: python3 -I detect.py IMG OUT [--workers 32] [--snr 4.5] [--watch] [--only STEM,...]
"""
import argparse, glob, json, os, sys, time, traceback, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def ccd_detect(img, wt, msk, fwhm, sat, snr_min):
    from scipy import ndimage as nd
    bad = (msk != 0) | ~(wt > 0) | ~np.isfinite(img)
    ny, nx = img.shape
    # background: 64 px blocks, sigma-clipped median, bilinear back to full size
    B = 64
    py, px = (-ny) % B, (-nx) % B
    a = np.pad(np.where(bad, np.nan, img), ((0, py), (0, px)), constant_values=np.nan)
    blk = a.reshape((ny + py) // B, B, (nx + px) // B, B).transpose(0, 2, 1, 3).reshape((ny + py) // B, (nx + px) // B, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(blk, axis=2)
        mad = np.nanmedian(np.abs(blk - med[..., None]), axis=2)
        blk = np.where(np.abs(blk - med[..., None]) > 3 * 1.4826 * mad[..., None], np.nan, blk)
        med = np.nanmedian(blk, axis=2)
    if np.isnan(med).all():
        return None
    med = np.where(np.isnan(med), np.nanmedian(med), med)
    med = nd.median_filter(med, size=3, mode="nearest")
    yy = (np.arange(ny) + 0.5) / B - 0.5; xx = (np.arange(nx) + 0.5) / B - 0.5
    bkg = nd.map_coordinates(med, np.meshgrid(yy, xx, indexing="ij"), order=1, mode="nearest")
    r = np.where(bad, 0, img - bkg).astype(np.float32)
    w = np.where(bad, 0, wt).astype(np.float32)
    s = max(0.6, fwhm / 2.3548)
    num = nd.gaussian_filter(r * w, s, mode="constant", truncate=3.5)
    den = nd.gaussian_filter(w, s / np.sqrt(2), mode="constant", truncate=3.5) / (4 * np.pi * s * s)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = num / den
        err = 1 / np.sqrt(den)
        snr = np.where(den > 0, flux / err, 0).astype(np.float32)
    pk = (snr >= snr_min) & (snr == nd.maximum_filter(snr, size=5, mode="constant"))
    pk[:12, :] = pk[-12:, :] = False; pk[:, :12] = pk[:, -12:] = False
    ys, xs = np.nonzero(pk)
    if not len(ys):
        return None
    # 3 x 3 centroid of S/N (clipped at 0)
    acc = np.zeros(len(ys)); cx = np.zeros(len(ys)); cy = np.zeros(len(ys))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            v = np.clip(snr[ys + dy, xs + dx], 0, None)
            acc += v; cx += v * dx; cy += v * dy
    acc[acc == 0] = 1
    xc = xs + cx / acc; yc = ys + cy / acc
    f = flux[ys, xs]; fe = err[ys, xs]
    sharp = r[ys, xs] / np.where(f > 0, f / (2 * np.pi * s * s), np.inf)
    nb = nd.maximum_filter(bad.astype(np.uint8), size=5, mode="constant")[ys, xs]
    near_sat = nd.maximum_filter(img, size=3, mode="nearest")[ys, xs] >= 0.9 * sat if sat else np.zeros(len(ys), bool)
    edge = (ys < 24) | (ys >= ny - 24) | (xs < 24) | (xs >= nx - 24)
    flag = (near_sat * 1 + (nb > 0) * 2 + edge * 4).astype(np.uint8)
    return xc, yc, f, fe, snr[ys, xs], sharp, flag


def process(stem, img_dir, out_dir, snr_min):
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy import log as alog
    warnings.simplefilter("ignore")          # CP headers carry non-standard cards; WCS fixes are expected
    alog.setLevel("ERROR")
    d = os.path.join(img_dir, stem[4:10])
    fi = sorted(glob.glob(os.path.join(d, stem + "_ooi_*.fits.fz")))
    if not fi:
        return stem, "no image"
    fw, fd = fi[0].replace("_ooi_", "_oow_"), fi[0].replace("_ooi_", "_ood_")
    t0 = time.time()
    cols = {k: [] for k in ("ra", "dec", "flux", "ferr", "snr", "x", "y", "ccd", "sharp", "flag")}
    with fits.open(fi[0], memmap=False) as hi, fits.open(fw, memmap=False) as hw, fits.open(fd, memmap=False) as hd:
        P = hi[0].header
        wmap = {h.header.get("EXTNAME"): h for h in hw[1:]}
        dmap = {h.header.get("EXTNAME"): h for h in hd[1:]}
        nccd = 0
        for h in hi[1:]:
            H = h.header; ext = H.get("EXTNAME")
            if h.data is None or ext not in wmap or ext not in dmap:
                continue
            fwhm = H.get("FWHM") or (P.get("FWHM") or 4.0)
            try:
                fwhm = float(fwhm)
            except Exception:
                fwhm = 4.0
            if not 1.0 < fwhm < 20:
                fwhm = 4.0
            sat = H.get("SATURATE") or H.get("SATURATA") or 0
            res = ccd_detect(h.data.astype(np.float32), wmap[ext].data.astype(np.float32), dmap[ext].data,
                             fwhm, float(sat or 0), snr_min)
            nccd += 1
            if res is None:
                continue
            xc, yc, f, fe, sn, sh, fl = res
            ra, dec = WCS(H).all_pix2world(xc, yc, 0)
            for k, v in zip(("ra", "dec", "flux", "ferr", "snr", "x", "y", "sharp", "flag"), (ra, dec, f, fe, sn, xc, yc, sh, fl)):
                cols[k].append(v)
            cols["ccd"].append(np.full(len(xc), int(H.get("CCDNUM", 0)), np.int16))
    arr = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    out = dict(ra=arr["ra"].astype(np.float64), dec=arr["dec"].astype(np.float64), flux=arr["flux"].astype(np.float32),
               ferr=arr["ferr"].astype(np.float32), snr=arr["snr"].astype(np.float32), x=arr["x"].astype(np.float32),
               y=arr["y"].astype(np.float32), ccd=arr["ccd"].astype(np.int16), sharp=arr["sharp"].astype(np.float32),
               flag=arr["flag"].astype(np.uint8))
    exptime = float(P.get("EXPTIME", 0) or 0)
    mjd = float(P.get("MJD-OBS", 0) or 0)

    def num(k, dflt=np.nan):
        try:
            return float(P.get(k))
        except Exception:
            return dflt
    meta = dict(stem=stem, file=os.path.basename(fi[0]), expnum=int(P.get("EXPNUM", 0) or 0), mjd_mid=mjd + exptime / 2 / 86400,
                band=str(P.get("BAND", P.get("FILTER", "")))[:12], exptime=exptime, magzero=num("MAGZERO"), fwhm=num("FWHM"),
                airmass=num("AIRMASS"), ra0=num("CENTRA", num("RA")) if not isinstance(P.get("RA"), str) else num("CENTRA"),
                dec0=num("CENTDEC"), wcscal=str(P.get("WCSCAL", "")), nccd=nccd, n=int(len(out["ra"])),
                seconds=round(time.time() - t0, 1))
    od = os.path.join(out_dir, stem[4:10]); os.makedirs(od, exist_ok=True)
    tmp = os.path.join(od, stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **out)
    os.replace(tmp, os.path.join(od, stem + ".npz"))
    return stem, f"ok {meta['n']} det, {nccd} ccd, {meta['seconds']} s"


def complete(img_dir):
    """stems whose ooi, oow and ood files are all on disk (no .part)"""
    have = {}
    for p in glob.glob(os.path.join(img_dir, "*", "c4d_*.fits.fz")):
        b = os.path.basename(p)
        have.setdefault(b[:17], set()).add(b[18:21])
    return {s for s, k in have.items() if {"ooi", "oow", "ood"} <= k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--snr", type=float, default=4.5)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "detect.log"), "a")
    say = lambda *x: (print(time.strftime("%m-%d %H:%M:%S"), *x, file=log, flush=True))
    done = {os.path.basename(p)[:17] for p in glob.glob(os.path.join(a.out, "*", "c4d_*.npz")) if ".tmp." not in p}
    fdone = os.path.join(os.path.dirname(os.path.abspath(a.img)), "log", "FETCH_DONE")
    nok = nerr = 0
    with ProcessPoolExecutor(a.workers) as ex:
        pending = {}
        while True:
            todo = set(a.only.split(",")) if a.only else complete(a.img)
            new = sorted(todo - done - set(pending))
            for s in new:
                pending[s] = ex.submit(process, s, a.img, a.out, a.snr)
            for s, fu in list(pending.items()):
                if fu.done():
                    try:
                        _, msg = fu.result(); nok += 1
                    except Exception as e:
                        msg = "ERROR " + repr(e)[:300]; nerr += 1
                        say(s, traceback.format_exc()[-600:].replace("\n", " | "))
                    say(s, msg); done.add(s); del pending[s]
            if not a.watch or a.only:
                if not pending:
                    break
            elif not pending and os.path.exists(fdone) and not (complete(a.img) - done):
                break
            time.sleep(2 if pending else 30)
    say(f"end: {nok} exposures ok, {nerr} errors")
    print(f"{nok} ok, {nerr} errors")


if __name__ == "__main__":
    main()
