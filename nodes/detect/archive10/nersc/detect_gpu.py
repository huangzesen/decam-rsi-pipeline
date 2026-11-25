#!/usr/bin/env python3
"""GPU version of detect_pad.py (2026-11-16). Same lists, same files; two changes:
1. the per-CCD image filters (background, matched filter, peak finding) run on a GPU with torch; reading and
   decompressing the three files stays on the CPU and is now most of the time;
2. each CCD's arrays are dropped after use, so a worker holds one CCD at a time, not all 60 (detect.py peaks at 6.5 GB
   per exposure, which caps a 512 GB node at 56 workers).
Workers share the node's GPUs: worker k uses GPU (k - 1) mod n. The arithmetic follows scipy's (float64 inside the
filters, float32 between the two passes), so the lists should agree with detect_pad.py's to rounding; stepA/cmp_lists.py
checks that. Known difference: the near-saturation flag ignores NaN pixels in its 3 x 3 box.

Original text of detect.py:
Single-exposure detection for the archive5 run: every DECam InstCal exposure (ooi image, oow inverse-variance weight,
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


_DEV = None


def device():
    """this worker's GPU; CUDA starts here, in the worker, never in the parent"""
    global _DEV
    if _DEV is None:
        import multiprocessing as mp
        import torch
        torch.set_num_threads(1)
        k = (mp.current_process()._identity or (1,))[0]
        _DEV = torch.device(f"cuda:{(k - 1) % torch.cuda.device_count()}")
    return _DEV


def _nanmed(x, torch):
    """np.nanmedian over the last axis: NaN sorts last; mean of the two middle values"""
    s = torch.sort(x, dim=-1).values
    n = (~torch.isnan(x)).sum(-1)
    lo = ((n - 1) // 2).clamp(min=0); hi = (n // 2).clamp(max=x.shape[-1] - 1)
    m = (s.gather(-1, lo[..., None]) + s.gather(-1, hi[..., None]))[..., 0] / 2
    return torch.where(n > 0, m, torch.full_like(m, float("nan")))


def _gauss(x, s, torch, F):
    """scipy.ndimage.gaussian_filter(x, s, mode="constant", truncate=3.5) for float32 x"""
    s = float(s); rad = int(3.5 * s + 0.5)
    k = torch.arange(-rad, rad + 1, dtype=torch.float64, device=x.device)
    k = torch.exp(-0.5 / (s * s) * k * k); k = k / k.sum()
    y = F.conv2d(x.double()[None, None], k.view(1, 1, -1, 1), padding=(rad, 0))
    y = F.conv2d(y.float().double(), k.view(1, 1, 1, -1), padding=(0, rad))
    return y[0, 0].float()


def ccd_detect(img, wt, msk, fwhm, sat, snr_min):
    """img, wt: float32 arrays; msk: bool array, True = masked"""
    import torch
    import torch.nn.functional as F
    from scipy import ndimage as nd
    dev = device(); nan = float("nan")
    img = torch.from_numpy(img).to(dev); wt = torch.from_numpy(wt).to(dev)
    bad = torch.from_numpy(msk).to(dev) | ~(wt > 0) | ~torch.isfinite(img)
    ny, nx = img.shape
    # background: 64 px blocks, sigma-clipped median, bilinear back to full size
    B = 64
    py, px = (-ny) % B, (-nx) % B
    a = F.pad(torch.where(bad, nan, img), (0, px, 0, py), value=nan)
    blk = a.reshape((ny + py) // B, B, (nx + px) // B, B).permute(0, 2, 1, 3).reshape((ny + py) // B, (nx + px) // B, -1)
    med = _nanmed(blk, torch)
    dev_ = (blk - med[..., None]).abs()
    mad = _nanmed(dev_, torch)
    blk = torch.where(dev_ > 3 * 1.4826 * mad[..., None], nan, blk)
    med = _nanmed(blk, torch).cpu().numpy()
    del a, blk, dev_
    if np.isnan(med).all():
        return None
    med = np.where(np.isnan(med), np.nanmedian(med), med)
    med = nd.median_filter(med, size=3, mode="nearest")
    mt = torch.from_numpy(med.astype(np.float64)).to(dev)

    def axis(n, nb):
        c = ((torch.arange(n, dtype=torch.float64, device=dev) + 0.5) / B - 0.5).clamp(0, nb - 1)
        i0 = c.floor().clamp(max=nb - 2).long()
        return i0, c - i0
    iy, ty = axis(ny, mt.shape[0]); ix, tx = axis(nx, mt.shape[1])
    rows = mt[iy] * (1 - ty)[:, None] + mt[iy + 1] * ty[:, None]
    bkg = (rows[:, ix] * (1 - tx) + rows[:, ix + 1] * tx).float()    # scipy's map_coordinates returns the input's dtype, float32
    r = torch.where(bad, 0.0, img - bkg)
    del bkg, rows
    w = torch.where(bad, 0.0, wt)
    s = max(0.6, fwhm / 2.3548)
    num = _gauss(r * w, s, torch, F)
    den = _gauss(w, s / np.sqrt(2), torch, F) / float(4 * np.pi * s * s)
    del w
    flux = num / den
    err = 1 / torch.sqrt(den)
    snr = torch.where(den > 0, flux / err, 0.0)
    pk = (snr >= snr_min) & (snr == F.max_pool2d(snr[None, None], 5, stride=1, padding=2)[0, 0])
    pk[:12, :] = False; pk[-12:, :] = False; pk[:, :12] = False; pk[:, -12:] = False
    ys, xs = torch.nonzero(pk, as_tuple=True)
    if not len(ys):
        return None
    # 3 x 3 centroid of S/N (clipped at 0); peaks are at least 12 px from the edge, so every box is inside the image
    acc = torch.zeros(len(ys), dtype=torch.float64, device=dev); cx = acc.clone(); cy = acc.clone()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            v = snr[ys + dy, xs + dx].clamp(min=0).double()
            acc += v; cx += v * dx; cy += v * dy
    acc[acc == 0] = 1
    xc = xs + cx / acc; yc = ys + cy / acc
    f = flux[ys, xs]; fe = err[ys, xs]
    sharp = r[ys, xs] / torch.where(f > 0, f / float(2 * np.pi * s * s), torch.full_like(f, float("inf")))
    nb = torch.zeros(len(ys), dtype=torch.bool, device=dev)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            nb |= bad[ys + dy, xs + dx]
    if sat:
        top = torch.full((len(ys),), -float("inf"), device=dev)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                top = torch.fmax(top, img[ys + dy, xs + dx])
        near_sat = top >= 0.9 * sat
    else:
        near_sat = torch.zeros(len(ys), dtype=torch.bool, device=dev)
    edge = (ys < 24) | (ys >= ny - 24) | (xs < 24) | (xs >= nx - 24)
    flag = (near_sat.to(torch.uint8) + nb.to(torch.uint8) * 2 + edge.to(torch.uint8) * 4)
    c = lambda t: t.cpu().numpy()
    return c(xc), c(yc), c(f), c(fe), c(snr[ys, xs]), c(sharp), c(flag)


def _drop(h):
    """forget an HDU's decompressed pixels (astropy keeps them until the file closes)"""
    for k in ("data", "compressed_data"):
        try:
            delattr(h, k)
        except Exception:
            pass
    b = getattr(h, "_bintable", None)
    if b is not None:
        try:
            del b.data
        except Exception:
            pass


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
        nccd = 0; tread = tgpu = 0.0; bad = []
        for h in hi[1:]:
            t1 = time.time()
            H = h.header; ext = H.get("EXTNAME")
            try:
                h.data
            except Exception as err:     # a CCD damaged in the archive's own file (c4d_220224_065704 N16: the checksum
                bad.append(f"{ext}: {type(err).__name__}"); continue   # matches NOIRLab's, cfitsio hits the end of the stream)
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
            wd = wmap[ext].data
            if wd.shape == (h.data.shape[0] + 2, h.data.shape[1] + 2):
                # step A, 2026-11-16: CP V4.9 weight maps in the Legacy Surveys ls9 files are 4096 x 2048 with a one-pixel
                # border of zeros around the 4094 x 2046 image (checked on c4d_150225_055441, CCDs N4 and S29: with the
                # border cut, every zero-weight pixel is a masked pixel; with any other shift thousands are not)
                wd = wd[1:-1, 1:-1]
            im, wd, mk = h.data.astype(np.float32), np.ascontiguousarray(wd, dtype=np.float32), dmap[ext].data != 0
            _drop(h); _drop(wmap[ext]); _drop(dmap[ext])
            t2 = time.time(); tread += t2 - t1
            res = ccd_detect(im, wd, mk, fwhm, float(sat or 0), snr_min)
            del im, wd, mk
            tgpu += time.time() - t2
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
                dec0=num("CENTDEC"), wcscal=str(P.get("WCSCAL", "")), nccd=nccd, bad_ccd=bad, n=int(len(out["ra"])),
                seconds=round(time.time() - t0, 1))
    od = os.path.join(out_dir, stem[4:10]); os.makedirs(od, exist_ok=True)
    tmp = os.path.join(od, stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **out)
    os.replace(tmp, os.path.join(od, stem + ".npz"))
    import resource
    return stem, (f"ok {meta['n']} det, {nccd} ccd{' (unreadable: ' + ', '.join(bad) + ')' if bad else ''}, {meta['seconds']} s (read {tread:.1f} gpu {tgpu:.1f}) "
                  f"rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.1f} GB")


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
