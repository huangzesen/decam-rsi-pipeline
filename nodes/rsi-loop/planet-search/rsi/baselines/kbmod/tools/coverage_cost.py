"""coverage_cost.py SCEN [RATE] : what a full KBMOD search of one field would cost, with finder.py's own grid rules.

For the finder's window (shortest run of >= 3 nights) and for the whole scenario span: the EBD distance cells over
28-150 au (curvature <= 1 px per cell, cell <= 0.004 in 1/r), the velocity grid per cell for KBMOD's default angle range
(+-pi/15 around the ecliptic, here also (v_t/v_c)^2 0.8-1.25 and radial speed +-0.2 v_c) at 2 px end-point spacing, and
the GPU time = start pixels x velocities x images / RATE (trajectory-images per second measured on the A100)."""
import csv, math, os, sys
import numpy as np
GM = 2.959122082855911e-4; ASEC = 180 * 3600 / math.pi; PIX = 0.263 / ASEC; EPS = math.radians(23.43928)
scen = sys.argv[1]; RATE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.1e11
NPIX = 29679 * 26892                                   # start pixels of the finder's common grid for one DECam field
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
mjd = np.array([float(r["mjd_mid"]) for r in rows]); night = np.floor(mjd + 0.3).astype(int)
O = np.array([[float(r[c]) for c in ("obs_x", "obs_y", "obs_z")] for r in rows])
cen = np.array([[math.cos(math.radians(float(r["dec_center"]))) * math.cos(math.radians(float(r["ra_center"]))),
                 math.cos(math.radians(float(r["dec_center"]))) * math.sin(math.radians(float(r["ra_center"]))),
                 math.sin(math.radians(float(r["dec_center"])))] for r in rows]).mean(0)
cen /= np.linalg.norm(cen)
kecl = np.array([0.0, -math.sin(EPS), math.cos(EPS)])


def basis(c):
    ez = c / np.linalg.norm(c); ex = np.cross([0.0, 0.0, 1.0], ez); ex /= np.linalg.norm(ex); return np.stack([ex, np.cross(ez, ex), ez])


def ray_sphere(Oi, s, r):
    b = np.sum(Oi * s, -1); return Oi + (-b + np.sqrt(np.maximum(b * b - np.sum(Oi * Oi, -1) + r * r, 0)))[..., None] * s


def cost(W):
    tw = mjd[W]; Ow = O[W]; Tw = tw.max() - tw.min(); tmid = 0.5 * (tw.min() + tw.max())
    Om = np.array([np.interp(tmid, tw, Ow[:, k]) for k in range(3)]); Rf = basis(cen)
    A = np.vstack([np.ones_like(tw), tw - tmid]).T; Op = Ow @ Rf[:2].T
    RO = float(np.abs(Op - A @ np.linalg.lstsq(A, Op, rcond=None)[0]).max()) + 1e-6
    Vperp = float(np.linalg.norm(Op[-1] - Op[0])) / max(Tw, 1e-3); dv = 2.0 / max(Tw, 0.5)
    tot = 0; ncell = 0; out = []
    lo_ = 1 / 150
    while lo_ < 1 / 28:
        dinv = min(2 * PIX / RO, 0.004, 1.2 * math.sqrt(GM * lo_) * lo_ / max(Vperp, 1e-6))
        ic = lo_ + dinv / 2; lo_ += dinv
        d = 1 / ic; rlo, rhi = 1 / (ic + dinv / 2), 1 / (ic - dinv / 2); Rb = basis(ray_sphere(Om, cen, d) / d)
        def pix(P, Oi):
            s = P - Oi; s /= np.linalg.norm(s); E = ray_sphere(Oi, s, d) / d; e = Rb @ E; return np.array([e[0] / e[2], e[1] / e[2]]) / PIX
        V = []
        for r in np.linspace(rlo, rhi, 5):
            P0 = ray_sphere(Om, cen, r); ph = P0 / r; e1 = np.cross(kecl, ph); e1 /= np.linalg.norm(e1); e2 = np.cross(ph, e1); vc = math.sqrt(GM / r)
            for beta in np.radians(np.linspace(-12, 12, 7)):
                for w in (0.8, 0.9, 1.0, 1.12, 1.25):
                    for f in (-0.2, 0.0, 0.2):
                        Vv = vc * math.sqrt(w) * (math.cos(beta) * e1 + math.sin(beta) * e2) + f * vc * ph
                        V.append((pix(P0 + Vv * (tw[-1] - tmid), Ow[-1]) - pix(P0 + Vv * (tw[0] - tmid), Ow[0])) / Tw)
        V = np.array(V); sp = np.hypot(*V.T); an = np.arctan2(V[:, 1], V[:, 0]); a0 = np.median(an)
        da = (an - a0 + math.pi) % (2 * math.pi) - math.pi
        vmin, vmax = max(0, sp.min() - dv), sp.max() + dv
        nv = math.ceil((vmax - vmin) / dv) + 1; na = math.ceil((da.max() - da.min()) * vmax / dv) + 1
        if vmin < 2 * dv:
            na = math.ceil(2 * math.pi * vmax / dv) + 1
        tot += nv * na; ncell += 1; out.append((d, nv * na))
    sec = NPIX * tot * len(W) / RATE
    cl = [c for c in out if 38 <= c[0] <= 48]
    sec_cl = NPIX * sum(c[1] for c in cl) * len(W) / RATE
    return Tw, len(W), ncell, tot, sec, sec_cl


un = np.unique(night); best = None
for i in range(len(un) - 2):
    sel = np.nonzero((night >= un[i]) & (night <= un[i + 2]))[0]; key = (np.ptp(mjd[sel]), -len(sel))
    if best is None or key < best[0]:
        best = (key, sel)
name = os.path.basename(os.path.normpath(scen))
for label, W in (("window", list(best[1])), ("all nights", list(range(len(rows))))):
    Tw, n, ncell, tot, sec, sec_cl = cost(W)
    print(f"{name} {label:10s}: span {Tw:5.1f} d, {n:2d} images, {ncell:4d} distance cells 28-150 au, {tot:.3g} velocities -> "
          f"{sec / 3600:.3g} A100-hours ({sec_cl / 3600:.3g} h for cells at 38-48 au)")
