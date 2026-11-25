import os
"""Per-exposure detection: read image + mask (weight file skipped: uniform noise per CCD), background, PSF-matched
filter S/N map, local maxima, sub-pixel peak, sharpness, WCS.  Runs in worker processes."""
import math, os, warnings, logging
import numpy as np

warnings.simplefilter("ignore")

_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        try:
            import torch
            torch.set_num_threads(1)
            dev = "cpu"
            if torch.backends.mps.is_available():
                dev = "mps"
            elif torch.cuda.is_available():
                dev = "cuda"
            _TORCH = (torch, dev)
        except Exception:
            _TORCH = (None, "cpu")
    return _TORCH


def gauss1d(s):
    k = int(math.ceil(3 * s))
    x = np.arange(-k, k + 1, dtype=np.float32)
    return np.exp(-x ** 2 / (2 * s * s)).astype(np.float32), k


def background(img, good, b=128):
    """Median per b x b block from a 4x subsample, bilinear upsampled."""
    ny, nx = img.shape
    sub = np.where(good, img, np.nan)[2::4, 2::4]
    sb = b // 4
    my, mx = sub.shape[0] // sb, sub.shape[1] // sb
    blk = sub[: my * sb, : mx * sb].reshape(my, sb, mx, sb).transpose(0, 2, 1, 3).reshape(my, mx, -1)
    bg = np.nanmedian(blk, axis=2)
    if np.isnan(bg).all():
        return None
    # fill nan blocks with global median
    bg = np.where(np.isnan(bg), np.nanmedian(bg), bg).astype(np.float32)
    return bg


def _interp_w(pix, cen):
    """(len(pix), len(cen)) linear interpolation weights, constant beyond the end centres."""
    eye = np.eye(len(cen), dtype=np.float64)
    if len(cen) == 1:
        return np.ones((len(pix), 1), np.float32)
    return np.stack([np.interp(pix, cen, eye[j]) for j in range(len(cen))], 1).astype(np.float32)


def bg_amps(img, good, b=128, split=True):
    """gen 111 (arm M, TNO_BGFIX=3): background with block centres placed at their true pixel positions, blocks
    tiling the whole CCD, a 3x3 median filter over the block grid, and (split) the two amplifier halves (x < nx/2,
    x >= nx/2) estimated and interpolated separately, so an amplifier level step (DECam S7: 78 vs 30 ADU) is not
    smeared over ~100 px either side.  Returns (ny, nx) float32 or None."""
    ny, nx = img.shape
    from scipy.ndimage import median_filter
    halves = [(0, nx // 2), (nx // 2, nx)] if (split and nx >= 4 * b) else [(0, nx)]
    out = np.full((ny, nx), np.nan, np.float32)
    ok_any = False
    for xa, xb in halves:
        sub = np.where(good[2::4, xa + 2:xb:4], img[2::4, xa + 2:xb:4], np.nan)
        nby = max(1, int(round(ny / b))); nbx = max(1, int(round((xb - xa) / b)))
        ey = np.linspace(0, sub.shape[0], nby + 1).astype(int); ex_ = np.linspace(0, sub.shape[1], nbx + 1).astype(int)
        grid = np.full((nby, nbx), np.nan, np.float64)
        for i in range(nby):
            rowblk = sub[ey[i]:ey[i + 1]]
            for j in range(nbx):
                blk = rowblk[:, ex_[j]:ex_[j + 1]]
                if np.isfinite(blk).sum() >= 16:
                    grid[i, j] = np.nanmedian(blk)
        if np.isnan(grid).all():
            continue
        ok_any = True
        grid = np.where(np.isnan(grid), np.nanmedian(grid), grid)
        if min(grid.shape) >= 3:
            grid = median_filter(grid, size=3, mode="nearest")
        cy = 2 + 4 * (ey[:-1] + ey[1:] - 1) / 2.0
        cx = xa + 2 + 4 * (ex_[:-1] + ex_[1:] - 1) / 2.0
        Wy = _interp_w(np.arange(ny, dtype=np.float64), cy); Wx = _interp_w(np.arange(xa, xb, dtype=np.float64), cx)
        out[:, xa:xb] = Wy @ grid.astype(np.float32) @ Wx.T
    if not ok_any:
        return None
    if np.isnan(out).any():
        out = np.where(np.isnan(out), np.float32(np.nanmedian(out)), out)
    return out


def bg_full(img, good, t, F, b=128):
    """gen 111 (arm M): full-resolution background (1, 1, ny, nx) tensor, or None.
    TNO_BGFIX=0 (old): the (my, mx) block grid, which covers only my*b x mx*b pixels, was stretched over the whole
    CCD with bilinear align_corners=False, so block centres were misplaced by up to ~(n mod b) px at the far edges.
    TNO_BGFIX=1: grid upsampled at exactly b px per block, edges replicated (no misplacement).
    TNO_BGFIX=2: as 1, plus a 3x3 median filter over the block grid (SExtractor-style), so one block whose
    median is pulled up by a bright star's halo no longer over-subtracts its neighbourhood (B13 exposures 3/4)."""
    ny, nx = img.shape
    BGFIX = int(os.environ.get("TNO_BGFIX", "0") or 0)   # gen 113 armB: default 0 (gen111 fix: TNO_BGFIX=3)
    if BGFIX >= 3:
        Bn = bg_amps(img, good, b, split=BGFIX >= 3)
        return None if Bn is None else t(Bn)
    bg = background(img, good, b)
    if bg is None:
        return None
    if BGFIX <= 0:
        return F.interpolate(t(bg), size=(ny, nx), mode="bilinear", align_corners=False)
    if BGFIX >= 2 and min(bg.shape) >= 3:
        from scipy.ndimage import median_filter
        bg = median_filter(bg, size=3, mode="nearest").astype(np.float32)
    my, mx = bg.shape
    B = F.interpolate(t(bg), size=(my * b, mx * b), mode="bilinear", align_corners=False)
    if my * b != ny or mx * b != nx:
        B = F.pad(B, (0, nx - mx * b, 0, ny - my * b), mode="replicate")
    return B


def process_ccd(img, msk, fwhm, thresh, want_map=False):
    torch, dev = _torch()
    good = (msk == 0) & np.isfinite(img)
    if good.mean() < 0.2:
        return None
    s = max(fwhm, 2.0) / 2.355
    g, k = gauss1d(s)
    ny, nx = img.shape
    F = torch.nn.functional
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None, None].to(dev)
    I = t(np.where(good, img, 0).astype(np.float32))
    G = t(good.astype(np.float32))
    B = bg_full(img, good, t, F)
    if B is None:
        return None
    R = (I - B) * G
    # robust pixel noise
    samp = R[0, 0, 1::7, 1::7].flatten()
    gs = G[0, 0, 1::7, 1::7].flatten() > 0
    samp = samp[gs]
    med = torch.median(samp)
    sig = 1.4826 * torch.median(torch.abs(samp - med))
    R = (R - med * G) / torch.clamp(sig, min=1e-6)
    R = torch.clamp(R, -50, 1e5)
    gx, gy = t(g[None, :]), t(g[:, None])
    num = F.conv2d(F.conv2d(R, gx, padding=(0, k)), gy, padding=(k, 0))
    den = F.conv2d(F.conv2d(G, gx * gx, padding=(0, k)), gy * gy, padding=(k, 0))
    full = float((g * g).sum()) ** 2
    okd = den > 0.6 * full
    snr = torch.where(okd, num / torch.sqrt(torch.clamp(den, min=1e-6)), torch.zeros_like(num))
    s2 = snr[0, 0, 3::11, 3::11].flatten()
    s2 = s2[okd[0, 0, 3::11, 3::11].flatten()]
    if s2.numel() < 100:
        return None
    m2 = torch.median(s2)
    ns = 1.4826 * torch.median(torch.abs(s2 - m2))
    snr = (snr - m2) / torch.clamp(ns, min=1e-6)
    peak = (snr == F.max_pool2d(snr, 5, 1, 2)) & (snr >= thresh) & okd
    ys, xs = torch.nonzero(peak[0, 0], as_tuple=True)
    # keep away from edges
    e = 3
    ok = (xs >= e) & (xs < nx - e) & (ys >= e) & (ys < ny - e)
    xs, ys = xs[ok], ys[ok]
    if xs.numel() == 0:
        smap = None
        if want_map:
            smap = torch.where(okd, snr, torch.full_like(snr, float('nan')))[0, 0, ::2, ::2].to(torch.float16).cpu().numpy()
        return np.zeros((0, 5)), float(sig), smap
    S = snr[0, 0]
    c = S[ys, xs]
    l, r_ = S[ys, xs - 1], S[ys, xs + 1]
    d_, u = S[ys - 1, xs], S[ys + 1, xs]
    dx = 0.5 * (l - r_) / torch.clamp(l - 2 * c + r_, max=-1e-6)
    dy = 0.5 * (d_ - u) / torch.clamp(d_ - 2 * c + u, max=-1e-6)
    dx = torch.clamp(dx, -0.7, 0.7)
    dy = torch.clamp(dy, -0.7, 0.7)
    # sharpness: central raw pixel (in noise units) vs matched-filter amplitude; cosmic rays are much sharper
    raw = R[0, 0]
    rawc = (raw[ys, xs] + 0.25 * (raw[ys, xs - 1] + raw[ys, xs + 1] + raw[ys - 1, xs] + raw[ys + 1, xs])) / 2.0
    # amplitude estimate in noise units = num/den (Gaussian peak amplitude)
    amp = num[0, 0, ys, xs] / torch.clamp(den[0, 0, ys, xs], min=1e-6)
    sharp = rawc / torch.clamp(amp, min=1e-3)
    flux = amp * sig * (2 * math.pi * s * s)          # ADU, total
    sm = float(sig)
    out = np.stack([(xs.float() + dx).cpu().numpy(), (ys.float() + dy).cpu().numpy(), c.cpu().numpy(),
                    sharp.cpu().numpy(), flux.cpu().numpy()], 1).astype(np.float64)
    smap = None
    if want_map:
        sm_ = torch.where(okd, snr, torch.full_like(snr, float('nan')))
        smap = sm_[0, 0, ::2, ::2].to(torch.float16).cpu().numpy()
    return out, sm, smap


def ccd_geom(w, shape):
    """Quadratic map from gnomonic (xi, eta) about the CCD centre to pixel (x, y), fitted on a grid."""
    ny, nx = shape
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 17))
    gx, gy = gx.ravel(), gy.ravel()
    ra, de = w.all_pix2world(gx, gy, 0)
    rc, dc = w.all_pix2world([(nx - 1) / 2], [(ny - 1) / 2], 0)
    rc, dc = math.radians(float(rc[0])), math.radians(float(dc[0]))
    xi, eta = gnomonic(np.radians(ra), np.radians(de), rc, dc)
    A = poly2(xi, eta)
    cx, *_ = np.linalg.lstsq(A, gx, rcond=None)
    cy, *_ = np.linalg.lstsq(A, gy, rcond=None)
    return dict(rc=rc, dc=dc, cx=cx, cy=cy, nx=nx, ny=ny, rad=float(np.max(np.hypot(xi, eta))))


def gnomonic(ra, de, rc, dc):
    cosc = np.sin(dc) * np.sin(de) + np.cos(dc) * np.cos(de) * np.cos(ra - rc)
    xi = np.cos(de) * np.sin(ra - rc) / cosc
    eta = (np.cos(dc) * np.sin(de) - np.sin(dc) * np.cos(de) * np.cos(ra - rc)) / cosc
    return xi, eta


def poly2(xi, eta):
    s = 1e3      # scale to keep the normal equations well conditioned
    a, b = xi * s, eta * s
    return np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], -1)


def detect_exposure(args):
    """args: (exp_index, image_path, mask_path, thresh). Returns dict with arrays."""
    ei, ipath, mpath, thresh, want_map = args
    maps = []
    import astropy
    astropy.log.setLevel("ERROR")
    from astropy.io import fits
    from astropy.wcs import WCS
    res = []
    fw = []
    import fastfits
    try:
        hi = fits.open(ipath, memmap=False)
        hm = fits.open(mpath, memmap=False)
        hbi = fits.open(ipath, memmap=False, disable_image_compression=True)
        hbm = fits.open(mpath, memmap=False, disable_image_compression=True)
    except Exception as ex:
        return dict(ei=ei, det=np.zeros((0, 8)), fwhm=np.nan, err=str(ex), maps=[])
    mext = {}
    for k in range(1, len(hbm)):
        mext[hbm[k].header.get("EXTNAME", k)] = k

    def getdata(hc, hb, k):
        a = None
        try:
            if k < len(hb) and hb[k].header.get("ZIMAGE", False):
                a = fastfits.read_tiled(hb[k])
        except Exception:
            a = None
        if a is None:
            a = hc[k].data
        return a

    for k in range(1, len(hi)):
        try:
            h = hi[k].header
            if "CTYPE1" not in h:
                continue
            img = getdata(hi, hbi, k)
            if img is None:
                continue
            img = img.astype(np.float32, copy=False)
            km = mext.get(hbi[k].header.get("EXTNAME", h.get("EXTNAME", k)), k)
            msk = getdata(hm, hbm, km)
            if msk is None or msk.shape != img.shape:
                msk = np.zeros(img.shape, np.int16)
            fwhm = float(h.get("FWHM") or 0) or 4.0
            if not (1.5 < fwhm < 15):
                fwhm = 4.0
            r = process_ccd(img, msk, fwhm, thresh, want_map)
            if r is None:
                continue
            d, sig, smap = r
            if smap is not None:
                maps.append((ccd_geom(WCS(h), img.shape), smap))
            w = WCS(h)
            ra, de = w.all_pix2world(d[:, 0], d[:, 1], 0)
            ccd = float(h.get("CCDNUM") or k)
            res.append(np.c_[ra, de, d[:, 2], d[:, 3], d[:, 4], np.full(len(d), ccd), d[:, 0], d[:, 1]])
            fw.append(fwhm)
        except Exception:
            continue
    for x in (hi, hm, hbi, hbm):
        x.close()
    det = np.concatenate(res) if res else np.zeros((0, 8))
    return dict(ei=ei, det=det, fwhm=float(np.median(fw)) if fw else np.nan, err=None, maps=maps)


POOL_MODE = os.environ.get("TNO_POOL", "max")


def snr_pooled(img, msk, fwhm, pool=4, sig_out=None):
    """Matched-filter S/N map (as in process_ccd), max-pooled pool x pool. Returns float16 array (nan = no data).
    gen 19: if sig_out is a list, the robust pixel noise (ADU) is appended to it (for depth weights)."""
    torch, dev = _torch()
    good = (msk == 0) & np.isfinite(img)
    if good.mean() < 0.2:
        return None
    s = max(fwhm, 2.0) / 2.355
    g, k = gauss1d(s)
    ny, nx = img.shape
    F = torch.nn.functional
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None, None].to(dev)
    I = t(np.where(good, img, 0).astype(np.float32))
    G = t(good.astype(np.float32))
    B = bg_full(img, good, t, F)
    if B is None:
        return None
    R = (I - B) * G
    samp = R[0, 0, 1::7, 1::7].flatten()
    gs = G[0, 0, 1::7, 1::7].flatten() > 0
    samp = samp[gs]
    med = torch.median(samp)
    sig = 1.4826 * torch.median(torch.abs(samp - med))
    if sig_out is not None:
        sig_out.append(float(sig))
    R = (R - med * G) / torch.clamp(sig, min=1e-6)
    R = torch.clamp(R, -50, 1e5)
    gx, gy = t(g[None, :]), t(g[:, None])
    num = F.conv2d(F.conv2d(R, gx, padding=(0, k)), gy, padding=(k, 0))
    den = F.conv2d(F.conv2d(G, gx * gx, padding=(0, k)), gy * gy, padding=(k, 0))
    full = float((g * g).sum()) ** 2
    okd = den > 0.6 * full
    snr = torch.where(okd, num / torch.sqrt(torch.clamp(den, min=1e-6)), torch.zeros_like(num))
    s2 = snr[0, 0, 3::11, 3::11].flatten()
    s2 = s2[okd[0, 0, 3::11, 3::11].flatten()]
    if s2.numel() < 100:
        if sig_out is not None:
            sig_out.pop()
        return None
    m2 = torch.median(s2)
    ns = 1.4826 * torch.median(torch.abs(s2 - m2))
    snr = (snr - m2) / torch.clamp(ns, min=1e-6)
    if POOL_MODE == "avg":
        # gen 18: mean-pool the matched-filter S/N (keeps ~all of a point source's S/N at 1"; max-pool kept ~55-80 %)
        okf = okd.float()
        num_ = F.avg_pool2d(torch.where(okd, snr, torch.zeros_like(snr)), pool, pool, ceil_mode=True)
        fr_ = F.avg_pool2d(okf, pool, pool, ceil_mode=True)
        pm = torch.where(fr_ >= 0.5, num_ / torch.clamp(fr_, min=1e-3), torch.full_like(num_, float("nan")))
    else:
        snr = torch.where(okd, snr, torch.full_like(snr, -1e4))
        pm = F.max_pool2d(snr, pool, pool, ceil_mode=True)
        pm = torch.where(pm < -1e3, torch.full_like(pm, float("nan")), pm)
    return pm[0, 0].to(torch.float16).cpu().numpy()


def map_exposure(args):
    """args: (exp_index, image_path, mask_path). Returns dict(ei, maps=[(geom, pooled_map)], fwhm)."""
    ei, ipath, mpath = args
    import astropy
    astropy.log.setLevel("ERROR")
    from astropy.io import fits
    from astropy.wcs import WCS
    import fastfits
    maps, fw, sg, skyh = [], [], [], []
    try:
        hi = fits.open(ipath, memmap=False)
        hm = fits.open(mpath, memmap=False)
        hbi = fits.open(ipath, memmap=False, disable_image_compression=True)
        hbm = fits.open(mpath, memmap=False, disable_image_compression=True)
    except Exception as ex:
        return dict(ei=ei, maps=[], fwhm=np.nan, err=str(ex))

    def hnum(h, key):
        try:
            v = float(h.get(key))
            return v if math.isfinite(v) else None
        except Exception:
            return None
    # gen 19: photometric zero point for depth weights (primary header; CP: MAGZERO = MAGZPT + 2.5 log10(EXPTIME))
    zp = None
    try:
        h0 = hi[0].header
        zp = hnum(h0, "MAGZERO")
        if zp is None and hnum(h0, "MAGZPT") is not None and (hnum(h0, "EXPTIME") or 0) > 0:
            zp = hnum(h0, "MAGZPT") + 2.5 * math.log10(hnum(h0, "EXPTIME"))
        if zp is not None and not (15 < zp < 40):
            zp = None
    except Exception:
        zp = None
    mext = {}
    for k in range(1, len(hbm)):
        mext[hbm[k].header.get("EXTNAME", k)] = k

    def getdata(hc, hb, k):
        a = None
        try:
            if k < len(hb) and hb[k].header.get("ZIMAGE", False):
                a = fastfits.read_tiled(hb[k])
        except Exception:
            a = None
        if a is None:
            a = hc[k].data
        return a

    for k in range(1, len(hi)):
        try:
            h = hi[k].header
            if "CTYPE1" not in h:
                continue
            img = getdata(hi, hbi, k)
            if img is None:
                continue
            img = img.astype(np.float32, copy=False)
            km = mext.get(hbi[k].header.get("EXTNAME", h.get("EXTNAME", k)), k)
            msk = getdata(hm, hbm, km)
            if msk is None or msk.shape != img.shape:
                msk = np.zeros(img.shape, np.int16)
            fwhm = float(h.get("FWHM") or 0) or 4.0
            if not (1.5 < fwhm < 15):
                fwhm = 4.0
            sl_ = []
            pm = snr_pooled(img, msk, fwhm, sig_out=sl_)
            if pm is None:
                continue
            maps.append((ccd_geom(WCS(h), img.shape), pm))
            fw.append(fwhm)
            sg += sl_
            if hnum(h, "SKYSIGMA") is not None and hnum(h, "SKYSIGMA") > 0:
                skyh.append(hnum(h, "SKYSIGMA"))
        except Exception:
            continue
    for x in (hi, hm, hbi, hbm):
        x.close()
    # sky noise: header SKYSIGMA if present on the CCDs, else the measured robust pixel noise
    sky = float(np.median(skyh)) if skyh else (float(np.median(sg)) if sg else np.nan)
    return dict(ei=ei, maps=maps, fwhm=float(np.median(fw)) if fw else np.nan, err=None,
                zp=zp, sky=sky)
