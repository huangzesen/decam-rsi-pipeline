"""Per-night shift-and-stack of the stored matched-filter S/N maps (gen 2).

For every night with >= 2 exposures that kept an S/N map, the maps of the other exposures are resampled onto the
pixel grid (2x decimated) of the night's middle exposure, shifted along a grid of sky rates that covers what a body
beyond 25 au can do (observer reflex along the direction set by the observer velocity, plus a bound-orbit margin),
summed and divided by sqrt(n).  The maximum over rates is searched for peaks.  Each peak becomes a detection of a
pseudo-exposure (same time / observer as the middle exposure) that link.py uses only as a seed: a body too faint for
one exposure but visible in the night stack can then be linked across nights.
"""
import math, time
import numpy as np
from detect import gnomonic, poly2

ARCR = math.pi / 180 / 3600


def _inv(g):
    if "ixi" in g:
        return g["ixi"], g["ieta"]
    r = g["rad"]
    a, b = np.meshgrid(np.linspace(-r, r, 25), np.linspace(-r, r, 25))
    a, b = a.ravel(), b.ravel()
    A = poly2(a, b)
    x, y = A @ g["cx"], A @ g["cy"]
    P = np.stack([np.ones_like(x), x / 1e3, y / 1e3, (x / 1e3) ** 2, x * y / 1e6, (y / 1e3) ** 2], -1)
    ixi, *_ = np.linalg.lstsq(P, a, rcond=None)
    ieta, *_ = np.linalg.lstsq(P, b, rcond=None)
    g["ixi"], g["ieta"] = ixi, ieta
    return ixi, ieta


def _pix2unit(g, x, y):
    ixi, ieta = _inv(g)
    P = np.stack([np.ones_like(x), x / 1e3, y / 1e3, (x / 1e3) ** 2, x * y / 1e6, (y / 1e3) ** 2], -1)
    xi, eta = P @ ixi, P @ ieta
    # inverse gnomonic about (rc, dc)
    rc, dc = g["rc"], g["dc"]
    z = np.array([math.cos(dc) * math.cos(rc), math.cos(dc) * math.sin(rc), math.sin(dc)])
    ex = np.array([-math.sin(rc), math.cos(rc), 0.0])
    ey = np.cross(z, ex)
    v = z[None] + xi[:, None] * ex[None] + eta[:, None] * ey[None]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _unit2pix(g, u):
    ra = np.arctan2(u[:, 1], u[:, 0]); de = np.arcsin(np.clip(u[:, 2], -1, 1))
    xi, eta = gnomonic(ra, de, g["rc"], g["dc"])
    A = poly2(xi, eta)
    return A @ g["cx"], A @ g["cy"]


def _cu(g):
    return np.array([math.cos(g["dc"]) * math.cos(g["rc"]), math.cos(g["dc"]) * math.sin(g["rc"]), math.sin(g["dc"])])


def night_stacks(results, et, eobs, vobs, frame, deadline, log, thresh=5.0, cap=3000, fwhm_as=1.1):
    """Returns list of (ref_exposure, det (N,8) array [ra, dec, sn, 1, nan, ccd, x, y], nexp)."""
    import torch
    F = torch.nn.functional
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    night = np.floor(et - 0.5).astype(int)
    have = [e for e in sorted(results) if results[e].get("maps")]
    out = []
    nights = sorted(set(night[have].tolist())) if have else []
    for nn in nights:
        ex = [e for e in have if night[e] == nn]
        if len(ex) < 2:
            continue
        if time.time() > deadline:
            log("night stacks: deadline")
            break
        t0 = time.time()
        ex.sort(key=lambda e: et[e])
        ex = ex[:8]
        ref = ex[len(ex) // 2]
        dth = (et[ex] - et[ref]) * 24.0                     # hours
        span = float(dth.max() - dth.min())
        # rate grid in the tangent frame (rad/h): reflex direction from the observer velocity
        vo = frame @ vobs[ref]                               # au/day in frame axes
        vperp = vo[:2]
        sp = float(np.hypot(*vperp))
        if sp < 1e-4:
            dirv = np.array([1.0, 0.0])
        else:
            dirv = -vperp / sp                                # body appears to move opposite to the observer
        perp = np.array([-dirv[1], dirv[0]])
        # along-rate (arcsec/h) for gamma 1/150 .. 1/25 plus margin
        a_lo = sp / 150 / 24 / ARCR * 0.6
        a_hi = sp / 25 / 24 / ARCR * 1.25 + 0.5
        step = float(np.clip(0.7 * fwhm_as / max(span, 1e-3), 0.3, 50.0))
        if span * (a_hi + 1.5) < 0.6 * fwhm_as:
            rates = [np.zeros(2)]
        else:
            rates = []
            for a in np.arange(a_lo, a_hi + step / 2, step):
                w = 0.25 * a + 0.6
                for b in np.arange(-w, w + step / 2, step):
                    rates.append(a * dirv + b * perp)
            if len(rates) > 400:
                rates = rates[:: int(math.ceil(len(rates) / 400))]
        R = np.array(rates) * ARCR                            # rad/h, frame (xi east, eta north)
        nR = len(R)
        dets = []
        rmaps = results[ref]["maps"]
        for g_r, m_r in rmaps:
            h, w = m_r.shape
            ys, xs = np.mgrid[0:h, 0:w]
            xf, yf = (xs.ravel() * 2.0).astype(np.float64), (ys.ravel() * 2.0).astype(np.float64)
            uref = _pix2unit(g_r, xf, yf)
            cr = _cu(g_r)
            S = torch.zeros((nR, h, w), device=dev)
            Nn = torch.zeros((1, h, w), device=dev)
            mr = torch.from_numpy(np.nan_to_num(m_r.astype(np.float32), nan=0.0)).to(dev)
            vr = torch.from_numpy(np.isfinite(m_r).astype(np.float32)).to(dev)
            S += mr[None]
            Nn += vr[None]
            for e, dt in zip(ex, dth):
                if e == ref:
                    continue
                for g, m in results[e]["maps"]:
                    if float(cr @ _cu(g)) < math.cos(g["rad"] + g_r["rad"]):
                        continue
                    x, y = _unit2pix(g, uref)
                    inside = (x > -60) & (x < g["nx"] + 60) & (y > -60) & (y < g["ny"] + 60)
                    if not inside.any():
                        continue
                    # local Jacobian d(pixel)/d(frame tangent) at the CCD centre via finite differences
                    uc = _cu(g)
                    du = 1e-5
                    fx, fy = frame[0], frame[1]
                    x0, y0 = _unit2pix(g, uc[None])
                    x1, y1 = _unit2pix(g, (uc + du * fx)[None])
                    x2, y2 = _unit2pix(g, (uc + du * fy)[None])
                    J = np.array([[x1[0] - x0[0], x2[0] - x0[0]], [y1[0] - y0[0], y2[0] - y0[0]]]) / du
                    off = (R * dt) @ J.T                     # (nR, 2) pixel offsets
                    mm = torch.from_numpy(np.nan_to_num(m.astype(np.float32), nan=0.0)).to(dev)[None, None]
                    vv = torch.from_numpy(np.isfinite(m).astype(np.float32)).to(dev)[None, None]
                    mh, mw = m.shape
                    X = torch.from_numpy(x.reshape(h, w).astype(np.float32)).to(dev)
                    Y = torch.from_numpy(y.reshape(h, w).astype(np.float32)).to(dev)
                    OF = torch.from_numpy(off.astype(np.float32)).to(dev)
                    # valid mask (rate-independent within ~10 px; use the mid rate)
                    gxv = ((X + OF[nR // 2, 0]) / 2.0) / (mw - 1) * 2 - 1
                    gyv = ((Y + OF[nR // 2, 1]) / 2.0) / (mh - 1) * 2 - 1
                    gridv = torch.stack([gxv, gyv], -1)[None]
                    val = F.grid_sample(vv, gridv, mode="nearest", padding_mode="zeros", align_corners=True)[0]
                    if float(val.sum()) < 10:
                        continue
                    Nn += (val > 0.5).float()
                    for k0 in range(0, nR, 16):
                        k1 = min(nR, k0 + 16)
                        gx_ = ((X[None] + OF[k0:k1, 0, None, None]) / 2.0) / (mw - 1) * 2 - 1
                        gy_ = ((Y[None] + OF[k0:k1, 1, None, None]) / 2.0) / (mh - 1) * 2 - 1
                        grid = torch.stack([gx_, gy_], -1)
                        smp = F.grid_sample(mm.expand(k1 - k0, 1, mh, mw), grid, mode="bilinear",
                                            padding_mode="zeros", align_corners=True)[:, 0]
                        S[k0:k1] += smp * (val > 0.5).float()
            ok = Nn[0] >= 2
            St = S / torch.sqrt(torch.clamp(Nn, min=1))
            best, arg = St.max(0)
            best = torch.where(ok, best, torch.zeros_like(best))
            # renormalise by the robust scatter of the stacked map
            samp = best[ok][::7]
            if samp.numel() < 1000:
                continue
            med = torch.median(samp)
            sig = 1.4826 * torch.median(torch.abs(samp - med))
            bz = (best - med) / torch.clamp(sig, min=1e-3)
            # max over rates biases the scatter low: compensate with the single-rate scatter
            s1 = St[nR // 2][ok][::7]
            sig1 = float(1.4826 * torch.median(torch.abs(s1 - torch.median(s1))))
            bz = (best - torch.median(s1)) / max(sig1, 1e-3)
            peak = (bz == F.max_pool2d(bz[None, None], 5, 1, 2)[0, 0]) & (bz >= thresh) & ok
            yy, xx = torch.nonzero(peak, as_tuple=True)
            if xx.numel() == 0:
                continue
            e_ = 3
            keep = (xx >= e_) & (xx < w - e_) & (yy >= e_) & (yy < h - e_)
            xx, yy = xx[keep], yy[keep]
            c = bz[yy, xx]
            l, r_ = bz[yy, xx - 1], bz[yy, xx + 1]
            d_, u_ = bz[yy - 1, xx], bz[yy + 1, xx]
            dx = torch.clamp(0.5 * (l - r_) / torch.clamp(l - 2 * c + r_, max=-1e-6), -0.7, 0.7)
            dy = torch.clamp(0.5 * (d_ - u_) / torch.clamp(d_ - 2 * c + u_, max=-1e-6), -0.7, 0.7)
            px = ((xx.float() + dx) * 2.0).cpu().numpy().astype(np.float64)
            py = ((yy.float() + dy) * 2.0).cpu().numpy().astype(np.float64)
            uu = _pix2unit(g_r, px, py)
            ra = np.degrees(np.arctan2(uu[:, 1], uu[:, 0])) % 360
            de = np.degrees(np.arcsin(np.clip(uu[:, 2], -1, 1)))
            cc = c.cpu().numpy().astype(np.float64)
            dets.append(np.c_[ra, de, cc, np.ones_like(cc), np.full_like(cc, np.nan), np.full_like(cc, -1.0), px, py])
            del S, St
        if dets:
            d = np.concatenate(dets)
            if len(d) > cap:
                d = d[np.argsort(-d[:, 2])[:cap]]
            out.append((ref, d, len(ex)))
            log(f"night {nn}: stack of {len(ex)} exp (span {span:.2f} h, {nR} rates): {len(d)} peaks >= {thresh} "
                f"({time.time() - t0:.1f}s)")
        if dev == "mps":
            torch.mps.empty_cache()
    return out
