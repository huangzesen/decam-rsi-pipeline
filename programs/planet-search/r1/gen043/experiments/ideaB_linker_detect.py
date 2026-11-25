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


def process_ccd(img, msk, fwhm, thresh, want_map=False, full_out=None):
    torch, dev = _torch()
    good = (msk == 0) & np.isfinite(img)
    if good.mean() < 0.2:
        return None
    bg = background(img, good)
    if bg is None:
        return None
    s = max(fwhm, 2.0) / 2.355
    g, k = gauss1d(s)
    ny, nx = img.shape
    F = torch.nn.functional
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None, None].to(dev)
    I = t(np.where(good, img, 0).astype(np.float32))
    G = t(good.astype(np.float32))
    B = F.interpolate(t(bg), size=(ny, nx), mode="bilinear", align_corners=False)
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
    if full_out is not None:
        full_out.append((torch.where(okd, snr, torch.zeros_like(snr))[0, 0], okd[0, 0], float(sig), float((g * g).sum()), s))
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


# ------------------------------------------------------------------ gen015 B: same-pointing short co-adds (dense nights)
def detect_group(args):
    """args: (gid, [(image_path, mask_path), ...], thresh).  Co-add the members' matched-filter S/N maps per CCD
    (integer WCS shift onto the first member's pixel grid, sum / sqrt(n)), renormalise, detect peaks.  Returns
    dict(gid, det) with det columns as in detect_exposure (sharp set to a neutral 0.8)."""
    gid, members, thresh = args
    import astropy
    astropy.log.setLevel("ERROR")
    from astropy.io import fits
    from astropy.wcs import WCS
    import fastfits
    torch, dev = _torch()
    F = torch.nn.functional
    res = []
    hs = []
    try:
        for ip, mp in members:
            hs.append((fits.open(ip, memmap=False), fits.open(mp, memmap=False),
                       fits.open(ip, memmap=False, disable_image_compression=True),
                       fits.open(mp, memmap=False, disable_image_compression=True)))
    except Exception as ex:
        for x in hs:
            for y in x:
                y.close()
        return dict(gid=gid, det=np.zeros((0, 8)), err=str(ex))

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

    def ext_index(hb):
        d = {}
        for k in range(1, len(hb)):
            d[hb[k].header.get("EXTNAME", k)] = k
        return d

    idx = [(ext_index(x[2]), ext_index(x[3])) for x in hs]
    hi0, hm0, hbi0, hbm0 = hs[0]
    for k in range(1, len(hi0)):
        try:
            h0 = hi0[k].header
            if "CTYPE1" not in h0:
                continue
            name = hbi0[k].header.get("EXTNAME", h0.get("EXTNAME", k))
            w0 = WCS(h0)
            acc = cnt = None
            shape = None
            fl = []
            for mi, (hi, hm, hbi, hbm) in enumerate(hs):
                ki = idx[mi][0].get(name, k if mi == 0 else None)
                if ki is None:
                    continue
                h = hi[ki].header
                img = getdata(hi, hbi, ki)
                if img is None:
                    continue
                img = img.astype(np.float32, copy=False)
                if shape is None:
                    shape = img.shape
                if img.shape != shape:
                    continue
                km = idx[mi][1].get(name, ki)
                msk = getdata(hm, hbm, km)
                if msk is None or msk.shape != img.shape:
                    msk = np.zeros(img.shape, np.int16)
                fwhm = float(h.get("FWHM") or 0) or 4.0
                if not (1.5 < fwhm < 15):
                    fwhm = 4.0
                fo = []
                r = process_ccd(img, msk, fwhm, 99.0, False, full_out=fo)
                if r is None or not fo:
                    continue
                S, ok, sig, g2, s_ = fo[0]
                ny, nx = shape
                if mi == 0:
                    sx = sy = 0
                else:
                    wm = WCS(h)
                    rc, dc = wm.all_pix2world([(nx - 1) / 2], [(ny - 1) / 2], 0)
                    x0, y0 = w0.all_world2pix(rc, dc, 0)
                    sx = int(round(float(x0[0]) - (nx - 1) / 2)); sy = int(round(float(y0[0]) - (ny - 1) / 2))
                    if abs(sx) > 200 or abs(sy) > 200:
                        continue
                if acc is None:
                    acc = torch.zeros((ny, nx), dtype=torch.float32, device=S.device)
                    cnt = torch.zeros((ny, nx), dtype=torch.float32, device=S.device)
                # member pixel (x, y) -> first-member pixel (x + sx, y + sy)
                ys0, ys1 = max(0, sy), min(ny, ny + sy)
                xs0, xs1 = max(0, sx), min(nx, nx + sx)
                acc[ys0:ys1, xs0:xs1] += S[ys0 - sy:ys1 - sy, xs0 - sx:xs1 - sx]
                cnt[ys0:ys1, xs0:xs1] += ok[ys0 - sy:ys1 - sy, xs0 - sx:xs1 - sx].float()
                fl.append((sig, g2, s_))
            n = len(fl)
            if acc is None or n < 2:
                continue
            okc = cnt >= max(2.0, 0.75 * n)
            snr = torch.where(okc, acc / torch.sqrt(torch.clamp(cnt, min=1.0)), torch.zeros_like(acc))
            s2 = snr[3::11, 3::11].flatten()
            s2 = s2[okc[3::11, 3::11].flatten()]
            if s2.numel() < 100:
                continue
            m2 = torch.median(s2)
            ns = 1.4826 * torch.median(torch.abs(s2 - m2))
            snr = torch.where(okc, (snr - m2) / torch.clamp(ns, min=1e-6), torch.zeros_like(snr))
            S4 = snr[None, None]
            peak = ((S4 == F.max_pool2d(S4, 5, 1, 2))[0, 0]) & (snr >= thresh) & okc
            ys, xs = torch.nonzero(peak, as_tuple=True)
            ny, nx = shape
            e = 3
            okp = (xs >= e) & (xs < nx - e) & (ys >= e) & (ys < ny - e)
            xs, ys = xs[okp], ys[okp]
            if xs.numel() == 0:
                continue
            c = snr[ys, xs]
            l, r_ = snr[ys, xs - 1], snr[ys, xs + 1]
            d_, u = snr[ys - 1, xs], snr[ys + 1, xs]
            dx = torch.clamp(0.5 * (l - r_) / torch.clamp(l - 2 * c + r_, max=-1e-6), -0.7, 0.7)
            dy = torch.clamp(0.5 * (d_ - u) / torch.clamp(d_ - 2 * c + u, max=-1e-6), -0.7, 0.7)
            X = (xs.float() + dx).cpu().numpy().astype(np.float64)
            Y = (ys.float() + dy).cpu().numpy().astype(np.float64)
            cc = c.cpu().numpy().astype(np.float64)
            sig, g2, s_ = np.median([f[0] for f in fl]), np.median([f[1] for f in fl]), fl[0][2]
            # single-exposure-equivalent ADU (1/1.88: empirical match to single-exposure fluxes, scen02)
            flux = (cc / math.sqrt(n)) / g2 * sig * (2 * math.pi * s_ * s_) / 1.88
            ra, de = w0.all_pix2world(X, Y, 0)
            ccd = float(h0.get("CCDNUM") or k)
            res.append(np.c_[ra, de, cc, np.full(len(X), 0.8), flux, np.full(len(X), ccd), X, Y])
        except Exception:
            continue
    for x in hs:
        for y in x:
            y.close()
    det = np.concatenate(res) if res else np.zeros((0, 8))
    return dict(gid=gid, det=det, err=None, n=len(members))
