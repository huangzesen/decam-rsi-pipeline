"""Pair-count proxy for the linker's cost on a patch-window, from exposure metadata only.

The pair stage (link_gpu.py) takes every two exposures of a window at least `dtmin` nights apart and pairs their
detections whose BK-frame positions phi = theta + gamma x_E (f ~ 1) lie within R = npara (dn + 1), npara =
sqrt(2 GM gamma^3) (rad/day). Its time follows the raw pair count (2.5-4.5 ns per pair on a GH200, every run so far).
For uniform detection densities sigma_1, sigma_2 over two DECam disks (radius r0) whose BK-frame centres are s
apart, the raw pair count is sigma_1 sigma_2 g(s, R), g = the autocorrelation of the disk convolved with a disk of
radius R (table by FFT). The patch keeps detections within `keep` deg of its centre, so each exposure is weighted by
the fraction of its disk inside the patch.
"""
import numpy as np

GM = 2.9591220828559115e-04          # au^3 / day^2
R0 = 1.1                             # deg, DECam field radius (disk model)


def g_table(r0=R0, step=0.04, smax=9.0, rmax=7.0):
    """g(s, R) in deg^4 on grids s = 0..smax, R = 0..rmax (deg), for two disks of radius r0."""
    n = int(np.ceil(2 * (smax + rmax + r0) / step / 2)) * 2
    x = (np.arange(n) - n // 2) * step
    X, Y = np.meshgrid(x, x, indexing="ij")
    rr = np.hypot(X, Y)
    u = (rr <= r0).astype(float)
    Fu = np.fft.fft2(np.fft.ifftshift(u))
    Lf = Fu * np.conj(Fu) * step ** 2                       # FT of the lens-area function L(s) = |D(0) & D(s)|
    s = np.arange(0, smax + 1e-9, step); R = np.arange(0, rmax + 1e-9, step)
    G = np.zeros((len(s), len(R)))
    i0 = n // 2; si = i0 + np.round(s / step).astype(int)
    for k, r in enumerate(R):
        if r == 0:
            continue
        d = (rr <= r).astype(float)
        c = np.real(np.fft.fftshift(np.fft.ifft2(Lf * np.fft.fft2(np.fft.ifftshift(d))))) * step ** 2
        G[:, k] = c[si, i0]
    return s, R, G


_K = 48                                # sunflower points in the unit disk, equal area: quadrature for small R
_Q = np.sqrt((np.arange(_K) + 0.5) / _K)[:, None] * np.stack([np.cos(2.39996323 * np.arange(_K)), np.sin(2.39996323 * np.arange(_K))], 1)
RSMALL = 0.3                           # deg; below it the FFT table's pixels (0.04 deg) matter, so integrate directly


_SMALL = None


def _small_table(step=0.005):
    """Mean lens area over an R-disk, for R < RSMALL: s = 0..2 R0 + RSMALL, R = 0..RSMALL (48-point quadrature)."""
    s = np.arange(0, 2 * R0 + RSMALL + step, step); R = np.arange(0, RSMALL + step, step)
    S, RR = np.meshgrid(s, R, indexing="ij")
    L = lens(R0, R0, np.hypot(S[..., None] + RR[..., None] * _Q[:, 0], RR[..., None] * _Q[:, 1])).mean(-1)
    return s, R, L


def lookup(tab, s, R):
    """g at (s, R) arrays (deg): bilinear in the FFT table for R >= RSMALL (1-2% of Monte Carlo there), else
    pi R^2 times the mean lens area over the R-disk (tabulated from a 48-point quadrature; the lens area has a cusp
    at s = 0, so pi R^2 L(s) alone is 8% high at R = 0.2)."""
    global _SMALL
    if _SMALL is None:
        _SMALL = _small_table()
    s, R = np.broadcast_arrays(np.asarray(s, float), np.asarray(R, float))
    out = np.where(R >= RSMALL, _lookup(tab, s, R), 0.0)
    m = R < RSMALL
    if m.any():
        out[m] = np.pi * R[m] ** 2 * _lookup(_SMALL, s[m], R[m])
    return out


def _lookup(tab, s, R):
    sg, Rg, G = tab
    ds, dR = sg[1] - sg[0], Rg[1] - Rg[0]
    fs = np.clip(s / ds, 0, len(sg) - 1 - 1e-9); fr = np.clip(R / dR, 0, len(Rg) - 1 - 1e-9)
    i, j = fs.astype(int), fr.astype(int); a, b = fs - i, fr - j
    v = (G[i, j] * (1 - a) * (1 - b) + G[i + 1, j] * a * (1 - b) + G[i, j + 1] * (1 - a) * b + G[i + 1, j + 1] * a * b)
    return np.where(s > sg[-1], 0.0, v)


def lens(r1, r2, d):
    """Area of the intersection of disks of radii r1, r2 at centre distance d (arrays, same units)."""
    d = np.maximum(d, 1e-12)
    a = np.clip((d * d + r1 * r1 - r2 * r2) / (2 * d * r1), -1, 1)
    b = np.clip((d * d + r2 * r2 - r1 * r1) / (2 * d * r2), -1, 1)
    k = np.sqrt(np.maximum((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2), 0))
    v = r1 * r1 * np.arccos(a) + r2 * r2 * np.arccos(b) - 0.5 * k
    v = np.where(d >= r1 + r2, 0.0, v)
    return np.where(d <= np.abs(r1 - r2), np.pi * np.minimum(r1, r2) ** 2, v)


def tangent(ra0, dec0, ra, dec):
    """Gnomonic coordinates (deg) of (ra, dec) (deg) about (ra0, dec0)."""
    a0, d0, a, d = map(np.radians, (ra0, dec0, ra, dec))
    c = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    x = np.cos(d) * np.sin(a - a0) / c
    y = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / c
    return np.degrees(x), np.degrees(y)


def basis(ra0, dec0):
    """Unit vectors (east, north) of the tangent plane at (ra0, dec0), equatorial."""
    a, d = np.radians(ra0), np.radians(dec0)
    e = np.array([-np.sin(a), np.cos(a), 0.0])
    n = np.array([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)])
    return e, n


def window_pairs(tab, ra0, dec0, ra, dec, night, n1, xe, gammas, keep=2.6, dtmin=10, chunk=4_000_000):
    """Predicted raw pairs per hypothesis at each gamma (1/au) for one patch-window.
    ra, dec: exposure centres (deg); night: integer nights; n1: detections per exposure over its whole disk;
    xe: heliocentric observer positions (au, equatorial, (n, 3))."""
    x, y = tangent(ra0, dec0, ra, dec)
    w = lens(R0, keep, np.hypot(x, y)) / (np.pi * R0 ** 2)
    sig = n1 / (np.pi * R0 ** 2) * w
    e, nn = basis(ra0, dec0)
    px, py = np.degrees(xe @ e), np.degrees(xe @ nn)        # deg per (1/au) of gamma
    m = sig > 0
    x, y, px, py, sig, night = x[m], y[m], px[m], py[m], sig[m], night[m]
    out = np.zeros(len(gammas))
    if len(x) < 2:
        return out
    I, J = np.triu_indices(len(x), 1)
    ok = np.abs(night[J] - night[I]) >= dtmin
    I, J = I[ok], J[ok]
    for c0 in range(0, len(I), chunk):
        i, j = I[c0:c0 + chunk], J[c0:c0 + chunk]
        dn = np.abs(night[j] - night[i]).astype(float)
        ss = sig[i] * sig[j]
        for k, g in enumerate(gammas):
            s = np.hypot(x[i] - x[j] + g * (px[i] - px[j]), y[i] - y[j] + g * (py[i] - py[j]))
            R = np.degrees(np.sqrt(2 * GM * g ** 3)) * (dn + 1)
            out[k] += (ss * lookup(tab, s, R)).sum()
    return out
