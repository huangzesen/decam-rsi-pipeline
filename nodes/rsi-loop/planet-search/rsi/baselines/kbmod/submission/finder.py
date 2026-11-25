"""KBMOD baseline for the DECam distant-body task: KBMOD's standard shift-and-stack pipeline wrapped in the task's
interface.  See ../README.md for the configuration and what is KBMOD's and what is plumbing.

    python -I finder.py --scenario SCEN --out OUT.json --budget-s 900

Pipeline (one scenario):
 1. pick the search window: the shortest run of >= 3 nights (KBMOD searches straight lines, so the window must be short
    enough that bodies move linearly once parallax is removed);
 2. read the CCDs (image + data-quality mask) of the window's exposures and of up to MAX_LOAD exposures in all,
    subtract a block-median sky, put every exposure on one common tangent-plane grid at the native 0.263"/px, on the
    GPU (bilinear); flux-normalise each image to one zero point with typical TNO colours (KBMOD's color_scale);
 3. static sky: median of all loaded exposures (a self-template), subtracted from the window's images; pixels where the
    template is bright are masked (KBMOD v1 used a "global mask" of pixels flagged in several images);
 4. for each explicit barycentric (here: heliocentric) distance guess d (EBD, KBMOD's reprojection frame): resample the
    window's difference images into the EBD frame for d, tile the field, and run kbmod.run_search.SearchRunner on each
    tile (PSF-weighted psi/phi likelihood images, GPU grid search over an EclipticCenteredSearch velocity grid,
    GPU sigma-G filtering, peak-offset stamp filter, nn_start_end clustering), then a chi^2 light-curve test;
 5. every KBMOD trajectory -> sky positions at the window's exposures -> a heliocentric state (rays from the observer
    intersected with a sphere |P| = r, straight line in time with light time, r scanned for the best bound fit);
    p = a monotonic map of KBMOD's likelihood.  Output is rewritten after every tile.
"""
import argparse, csv, json, math, os, sys, time, warnings

import numpy as np

T0 = time.time()
GM = 2.959122082855911e-4            # au^3 / day^2
C_AU = 173.1446326846693             # au / day
ASEC = 180 * 3600 / math.pi
PIXS = 0.263                         # arcsec / pixel (DECam native, kept for the common grid)
PIX = PIXS / ASEC
EPS = math.radians(23.43928)
ZREF = 30.0                          # common zero point of the normalised images
# typical TNO colours relative to r (mag): used like KBMOD's color_scale so a body has one flux in every band
COLOUR = {"g": -0.6, "r": 0.0, "VR": 0.0, "i": 0.25, "z": 0.4, "Y": 0.45}
MAX_LOAD = int(os.environ.get("KB_MAX_LOAD", 12))
DEBUG_NCCD = int(os.environ.get("KB_DEBUG_NCCD", 0))     # >0: only this many CCDs per exposure (development)


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def write_out(path, cands):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"candidates": cands[:200]}, f)
    os.replace(tmp, path)


def unit(ra, dec):
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def basis(c):
    """Tangent-plane basis at unit vector c: rows ex (east), ey (north), ez (= c)."""
    ez = c / np.linalg.norm(c)
    ex = np.cross([0.0, 0.0, 1.0], ez); ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return np.stack([ex, ey, ez])


def ray_sphere(O, s, r):
    """Point on the ray O + rho s (s unit, rho > 0) with |P| = r (heliocentric)."""
    b = np.sum(O * s, -1)
    disc = b * b - np.sum(O * O, -1) + r * r
    return O + (-b + np.sqrt(np.maximum(disc, 0)))[..., None] * s


def _terms(u, v):
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3], -1)


# ------------------------------------------------------------------------------------------------ two-body helpers
def propagate(P0, V0, dt, nstep=8):
    """Two-body RK4 of one state to many offsets dt [K] (days)."""
    K = len(dt)
    P = np.repeat(P0[None], K, 0).astype(float); V = np.repeat(V0[None], K, 0).astype(float)
    h = (np.asarray(dt, float) / nstep)[:, None]
    acc = lambda p: -GM * p / (np.sum(p * p, 1, keepdims=True) ** 1.5)
    for _ in range(nstep):
        k1p, k1v = V, acc(P)
        k2p, k2v = V + 0.5 * h * k1v, acc(P + 0.5 * h * k1p)
        k3p, k3v = V + 0.5 * h * k2v, acc(P + 0.5 * h * k2p)
        k4p, k4v = V + h * k3v, acc(P + h * k3p)
        P = P + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        V = V + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return P, V


def observe(P0, V0, t0, t, O):
    """Unit vectors (equatorial) from observers O [K,3] at times t [K] to a body with state (P0, V0) at t0, light time."""
    lt = np.linalg.norm(P0 - O, axis=1) / C_AU
    for _ in range(3):
        P, _ = propagate(P0, V0, t - lt - t0)
        D = P - O
        lt = np.linalg.norm(D, axis=1) / C_AU
    return D / np.linalg.norm(D, axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------ CCD loading (worker)
def load_ccd(task):
    """Read one CCD: sky-subtracted, flux-normalised image (float16, NaN = masked), its noise, PSF FWHM and a cubic
    map from field-frame tangent-plane coordinates (radians) to CCD pixels."""
    warnings.simplefilter("ignore")
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy import ndimage as ndi
    img_path, mask_path, k, Rf, fwhm_fb, fz, inj = task
    with fits.open(img_path, memmap=False) as hi:
        d = np.asarray(hi[k].data, dtype=np.float32)
        hdr = hi[k].header
    ny, nx = d.shape
    w = WCS(hdr)
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 17))
    gx = gx.ravel(); gy = gy.ravel()
    ra, dec = w.all_pix2world(gx, gy, 0)
    uu = unit(np.radians(ra), np.radians(dec)) @ Rf.T
    X = uu[:, 0] / uu[:, 2]; Y = uu[:, 1] / uu[:, 2]
    X0, Y0 = X.mean(), Y.mean(); SX = max(np.ptp(X), np.ptp(Y)) / 2
    Ai = _terms((X - X0) / SX, (Y - Y0) / SX)
    cpx = np.linalg.lstsq(Ai, gx, rcond=None)[0]; cpy = np.linalg.lstsq(Ai, gy, rcond=None)[0]
    fw = hdr.get("FWHM", None)
    try:
        fw = float(fw)
    except Exception:
        fw = None
    if fw is None or not (1.5 < fw < 15):
        fw = fwhm_fb / PIXS
    if inj is not None and len(inj):           # development only: synthetic bodies (X, Y rad, counts)
        s = min(max(fw / 2.3548, 1.0), 5.0)
        T = _terms((inj[:, 0] - X0) / SX, (inj[:, 1] - Y0) / SX)
        ipx = T @ cpx; ipy = T @ cpy
        for qx, qy, qf in zip(ipx, ipy, inj[:, 2]):
            if 12 < qx < nx - 12 and 12 < qy < ny - 12:
                x0 = int(round(qx)); y0 = int(round(qy)); h = int(4 * s) + 2
                yy, xx = np.mgrid[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1]
                d[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1] += (qf / (2 * math.pi * s * s) *
                                                          np.exp(-((xx - qx) ** 2 + (yy - qy) ** 2) / (2 * s * s))).astype(np.float32)
    good = np.isfinite(d)
    if mask_path:
        with fits.open(mask_path, memmap=False) as hm:
            try:
                m = hm[hdr.get("EXTNAME", k)].data
            except Exception:
                m = hm[k].data
        good &= (np.asarray(m) == 0)
    # block-median sky (128 px blocks on a 4x subsample), bilinear back to full size
    B = 128
    gyb, gxb = (ny + B - 1) // B, (nx + B - 1) // B
    sub = np.full((gyb * B // 4, gxb * B // 4), np.nan, np.float32)
    s4 = np.where(good[::4, ::4], d[::4, ::4], np.nan)
    sub[:s4.shape[0], :s4.shape[1]] = s4
    blk = sub.reshape(gyb, B // 4, gxb, B // 4).transpose(0, 2, 1, 3).reshape(gyb, gxb, -1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(blk, axis=2)
    if not np.isfinite(med).any():
        return None
    med = np.where(np.isfinite(med), med, np.nanmedian(med))
    bg = ndi.zoom(med, (B, B), order=1, mode="nearest")[:ny, :nx] if (gyb > 1 and gxb > 1) else np.full((ny, nx), med.mean())
    if bg.shape != (ny, nx):
        bg = np.pad(bg, ((0, ny - bg.shape[0]), (0, nx - bg.shape[1])), mode="edge")
    r = d - bg
    rs = r[::4, ::4][good[::4, ::4]]
    if rs.size < 1000:
        return None
    sig = 1.4826 * float(np.median(np.abs(rs - np.median(rs))))
    out = np.where(good, r * fz, np.nan).astype(np.float16)
    return dict(k=k, sci=out, var=(sig * fz) ** 2, fwhm=fw, cpx=cpx, cpy=cpy, X0=X0, Y0=Y0, SX=SX, nx=nx, ny=ny,
                Xmin=X.min(), Xmax=X.max(), Ymin=Y.min(), Ymax=Y.max())


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=900)
    a = ap.parse_args()
    budget = a.budget_s
    deadline = T0 + budget - 45
    write_out(a.out, [])
    warnings.simplefilter("ignore")

    rows = list(csv.DictReader(open(os.path.join(a.scenario, "exposures.csv"))))
    nE = len(rows)
    mjd = np.array([float(r["mjd_mid"]) for r in rows])
    night = np.floor(mjd + 0.3).astype(int)
    bands = [r["band"] for r in rows]
    O = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows])
    mz = []
    for r in rows:
        try:
            mz.append(float(r["magzero"]))
        except Exception:
            mz.append(float("nan"))
    mz = np.array(mz)
    for e in range(nE):
        if not (np.isfinite(mz[e]) and 20 < mz[e] < 35):
            same = [mz[k] for k in range(nE) if bands[k] == bands[e] and np.isfinite(mz[k]) and 20 < mz[k] < 35]
            allz = [mz[k] for k in range(nE) if np.isfinite(mz[k]) and 20 < mz[k] < 35]
            mz[e] = np.median(same) if same else (np.median(allz) if allz else 29.5)
    zph = mz + np.array([COLOUR.get(b, 0.0) for b in bands])
    fz = 10 ** (0.4 * (ZREF - zph))                  # multiply counts by fz: a body of r-mag m has flux 10^-0.4(m-ZREF)
    seeing = []
    for r in rows:
        try:
            seeing.append(float(r["seeing_arcsec"]))
        except Exception:
            seeing.append(1.2)

    # ---- 1. search window: shortest run of >= 3 nights (ties: more exposures)
    un = np.unique(night)
    best = None
    for i0 in range(len(un)):
        for i1 in range(i0 + 2, len(un)):
            sel = np.nonzero((night >= un[i0]) & (night <= un[i1]))[0]
            span = mjd[sel].max() - mjd[sel].min()
            key = (span, -len(sel))
            if best is None or key < best[0]:
                best = (key, sel)
            break                                  # the shortest window starting at un[i0] has exactly 3 nights
    if best is None:
        log("fewer than 3 nights: nothing KBMOD can search for 3-night bodies"); return
    win = sorted(best[1].tolist(), key=lambda i: mjd[i])
    if len(win) > 14:
        win = win[:14]
    Tw = mjd[win].max() - mjd[win].min()
    log(f"{nE} exposures, {len(un)} nights; window {len(win)} exposures on nights {sorted(set(night[win].tolist()))}, span {Tw:.2f} d")
    others = [i for i in np.argsort(mjd) if i not in win]
    # template exposures: spread over the other nights (round-robin by night)
    by_n = {}
    for i in others:
        by_n.setdefault(night[i], []).append(i)
    rr = []
    lists = list(by_n.values())
    while any(lists):
        for l in lists:
            if l:
                rr.append(l.pop(0))
    load = win + rr[:max(0, MAX_LOAD - len(win))]

    # ---- field frame: centre of the window's pointings
    ctr = unit(np.radians([float(rows[i]["ra_center"]) for i in win]), np.radians([float(rows[i]["dec_center"]) for i in win])).mean(0)
    Rf = basis(ctr)

    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

    # injection (development only): same convention as the RSI program's FINDER_INJECT (Gaussian PSF, flux from magzero
    # + colour), so both programs see the same synthetic bodies
    inj_all = None
    if os.environ.get("FINDER_INJECT"):
        bodies = json.load(open(os.environ["FINDER_INJECT"]))
        inj_all = np.zeros((nE, len(bodies), 3))
        for b, bd in enumerate(bodies):
            st = np.array(bd["state"], float)
            s = observe(st[:3], st[3:], bd["epoch_mjd"], mjd, O) @ Rf.T
            inj_all[:, b, 0] = s[:, 0] / s[:, 2]; inj_all[:, b, 1] = s[:, 1] / s[:, 2]
            inj_all[:, b, 2] = 10 ** (-0.4 * (bd["mag"] - zph))
        log(f"injecting {len(bodies)} synthetic bodies")

    # ---- 2. read CCDs in parallel, place every loaded exposure on one tangent-plane grid on the GPU
    from astropy.io import fits
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = 4
    ncpu = max(1, min(8, ncpu, int(os.environ.get("OMP_NUM_THREADS", "8") or 8)))
    os.environ["OMP_NUM_THREADS"] = "1"
    grid = None                                    # (Xmin, Ymin, W, H)
    S = {}                                         # exposure -> float16 [H, W] tensor (NaN = no data)
    VAR, FWHM = {}, {}
    load_deadline = T0 + 0.30 * budget

    def place(e, c):
        Xmin, Ymin, W, H = grid
        gx0 = max(0, int(math.floor((c["Xmin"] - Xmin) / PIX)) - 2); gx1 = min(W, int(math.ceil((c["Xmax"] - Xmin) / PIX)) + 3)
        gy0 = max(0, int(math.floor((c["Ymin"] - Ymin) / PIX)) - 2); gy1 = min(H, int(math.ceil((c["Ymax"] - Ymin) / PIX)) + 3)
        if gx1 <= gx0 or gy1 <= gy0:
            return
        X = (Xmin + torch.arange(gx0, gx1, device=dev, dtype=torch.float64) * PIX - c["X0"]) / c["SX"]
        Y = (Ymin + torch.arange(gy0, gy1, device=dev, dtype=torch.float64) * PIX - c["Y0"]) / c["SX"]
        Yg, Xg = torch.meshgrid(Y, X, indexing="ij")
        T = torch.stack([torch.ones_like(Xg), Xg, Yg, Xg * Xg, Xg * Yg, Yg * Yg, Xg ** 3, Xg * Xg * Yg, Xg * Yg * Yg, Yg ** 3], -1)
        px = (T @ torch.as_tensor(c["cpx"], device=dev)).float(); py = (T @ torch.as_tensor(c["cpy"], device=dev)).float()
        del T, Xg, Yg
        nx, ny = c["nx"], c["ny"]
        inside = (px >= 0) & (px <= nx - 1) & (py >= 0) & (py <= ny - 1)
        img = torch.from_numpy(c["sci"]).to(dev)
        val = torch.isfinite(img)
        img = torch.where(val, img, torch.zeros_like(img)).float()
        g = torch.stack([2 * px / (nx - 1) - 1, 2 * py / (ny - 1) - 1], -1)[None]
        sv = F.grid_sample(torch.stack([img, val.float()])[None], g, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        ok = inside & (sv[1] > 0.999)
        reg = S[e][gy0:gy1, gx0:gx1]
        reg[ok] = sv[0][ok].half()

    with ProcessPoolExecutor(ncpu, mp_context=multiprocessing.get_context("spawn")) as pool:
        for n_done, e in enumerate(load):
            if n_done >= len(win) and time.time() > load_deadline:
                log(f"load time limit: {n_done} exposures loaded"); break
            r = rows[e]
            ip = os.path.join(a.scenario, r["image"]); mp = os.path.join(a.scenario, r["mask"]) if r.get("mask") else None
            try:
                with fits.open(ip) as h:
                    hd = [k for k in range(len(h)) if h[k].header.get("NAXIS", 0) == 2 or h[k].header.get("ZNAXIS", 0) == 2]
            except Exception as ex:
                log("cannot open", ip, ex); continue
            if DEBUG_NCCD:
                hd = hd[len(hd) // 2 - DEBUG_NCCD // 2: len(hd) // 2 - DEBUG_NCCD // 2 + DEBUG_NCCD]
            inj = None if inj_all is None else inj_all[e]
            ccds = [c for c in pool.map(load_ccd, [(ip, mp, k, Rf, seeing[e], fz[e], inj) for k in hd]) if c is not None]
            if not ccds:
                continue
            if grid is None:                       # grid = this exposure's footprint + the window's pointing offsets
                cu = unit(np.radians([float(rows[i]["ra_center"]) for i in win]), np.radians([float(rows[i]["dec_center"]) for i in win])) @ Rf.T
                ox = cu[:, 0] / cu[:, 2]; oy = cu[:, 1] / cu[:, 2]
                Xmin = min(c["Xmin"] for c in ccds) + min(0, (ox - ox[0]).min()) - 20 * PIX
                Xmax = max(c["Xmax"] for c in ccds) + max(0, (ox - ox[0]).max()) + 20 * PIX
                Ymin = min(c["Ymin"] for c in ccds) + min(0, (oy - oy[0]).min()) - 20 * PIX
                Ymax = max(c["Ymax"] for c in ccds) + max(0, (oy - oy[0]).max()) + 20 * PIX
                lim = math.radians(1.35)
                Xmin, Xmax, Ymin, Ymax = max(Xmin, -lim), min(Xmax, lim), max(Ymin, -lim), min(Ymax, lim)
                grid = (Xmin, Ymin, int((Xmax - Xmin) / PIX) + 1, int((Ymax - Ymin) / PIX) + 1)
                log(f"common grid {grid[2]} x {grid[3]} px ({grid[2] * PIXS / 3600:.2f} x {grid[3] * PIXS / 3600:.2f} deg)")
            S[e] = torch.full((grid[3], grid[2]), float("nan"), dtype=torch.float16, device=dev)
            for c in ccds:
                place(e, c)
            VAR[e] = float(np.median([c["var"] for c in ccds]))
            FWHM[e] = float(np.median([c["fwhm"] for c in ccds]))
            del ccds
            log(f"exposure {e} ({bands[e]}, night {night[e]}{', window' if e in win else ''}) on the grid; "
                f"sigma {math.sqrt(VAR[e]):.3g}, FWHM {FWHM[e] * PIXS:.2f}\"")
    win = [e for e in win if e in S]
    if len(set(night[win].tolist())) < 3:
        log("window lost nights while loading"); return
    loaded = list(S.keys())
    Xmin, Ymin, W, H = grid

    # ---- 3. static sky: median of all loaded exposures, subtracted from the window images; bright static pixels masked
    nT = len(loaded)
    gk = torch.exp(-0.5 * (torch.arange(-4, 5, device=dev, dtype=torch.float32) / 1.5) ** 2)
    gk = gk / gk.sum()
    if nT >= 3:
        CH = max(32, int(0.8e9 / (4 * W * nT)))
        for y0 in range(0, H, CH):
            st = torch.stack([S[e][y0:y0 + CH].float() for e in loaded])
            T = torch.nanmedian(st, 0).values
            T[torch.isfinite(st).sum(0) < 3] = float("nan")          # too few exposures here for a static template
            sigT = 1.2533 * math.sqrt(float(np.mean([VAR[e] for e in loaded])) / nT)
            # footprints of static sources: the template smoothed by a 1.5 px Gaussian above 3 sigma (sigma of the
            # smoothed noise = sigT / (2 sqrt(pi) 1.5)), grown by 2 px; the self-template is not PSF-matched, so these
            # pixels hold subtraction residuals in every image
            Tz = torch.nan_to_num(T, nan=0.0)[None, None]
            Ts = F.conv2d(F.conv2d(Tz, gk.view(1, 1, 1, -1), padding=(0, 4)), gk.view(1, 1, -1, 1), padding=(4, 0))[0, 0]
            bright = Ts > 3 * sigT / (2 * math.sqrt(math.pi) * 1.5)
            bright = F.max_pool2d(bright[None, None].float(), 5, 1, 2)[0, 0] > 0
            for e in win:
                blk = S[e][y0:y0 + CH]
                d_ = (blk.float() - torch.nan_to_num(T, nan=0.0))
                d_[bright | torch.isnan(T)] = float("nan")
                S[e][y0:y0 + CH] = d_.half()
            del st, T
        log(f"self-template from {nT} exposures subtracted")
    for e in loaded:
        if e not in win:
            del S[e]
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---- 4. EBD search
    from kbmod.configuration import SearchConfiguration
    from kbmod.core.image_stack_py import ImageStackPy
    from kbmod.core.psf import PSF
    from kbmod.run_search import SearchRunner
    from kbmod.trajectory_generator import create_trajectory_generator
    import kbmod.search as kb

    tw = mjd[win]; t0 = tw[0]; tmid = 0.5 * (tw.min() + tw.max())
    Ow = O[win]
    Om = np.array([np.interp(tmid, tw, Ow[:, k]) for k in range(3)])
    c_obs = ctr / np.linalg.norm(ctr)
    nights_w = night[win]
    # observer's departure from straight-line motion across the line of sight (sets the EBD distance spacing)
    A = np.vstack([np.ones_like(tw), tw - tmid]).T
    Op = Ow @ Rf[:2].T
    RO = float(np.abs(Op - A @ np.linalg.lstsq(A, Op, rcond=None)[0]).max()) + 1e-6
    TOL_C = 1.0                                      # px: curvature left in the EBD frame at the edge of a distance cell
    TOL_V = 2.0                                      # px: velocity grid step x window span (KBMOD's ~1 px end-point rule)
    Vperp = float(np.linalg.norm(Op[-1] - Op[0])) / max(Tw, 1e-3)

    def cell_width(ic):
        # in 1/r: curvature <= TOL_C px at the cell edge; the parallax drift across the cell stays below the body's own
        # motion so the cell's velocities form one ecliptic-centred fan; at most 0.004
        d_ = 1 / ic
        return min(2 * TOL_C * PIX / RO, 0.004, 1.2 * math.sqrt(GM / d_) / d_ / max(Vperp, 1e-6))

    # distance cells in 1/r: the one containing 42 au first, then alternately inwards and outwards
    inner, outer = [], []
    w0 = cell_width(1 / 42.0)
    cells_raw = [(1 / 42.0 - w0 / 2, 1 / 42.0 + w0 / 2)]
    a_, b_ = cells_raw[0]
    while b_ < 1 / 28:                              # inwards (larger 1/r)
        w_ = cell_width(b_); inner.append((b_, min(b_ + w_, 1 / 28))); b_ += w_
    while a_ > 1 / 150:                             # outwards
        w_ = cell_width(a_); outer.append((max(a_ - w_, 1 / 150), a_)); a_ -= w_
    for k in range(max(len(inner), len(outer))):
        if k < len(inner): cells_raw.append(inner[k])
        if k < len(outer): cells_raw.append(outer[k])
    cells = [(2 / (lo + hi), 1 / hi, 1 / lo) for lo, hi in cells_raw]     # (d, r_lo, r_hi)
    kecl = np.array([0.0, -math.sin(EPS), math.cos(EPS)])

    def ebd_basis(d):
        P = ray_sphere(Om, c_obs, d)
        return basis(P / d)

    def ebd_pix(P, Oi, d, Rb):
        s = P - Oi; s = s / np.linalg.norm(s, axis=-1, keepdims=True)
        E = ray_sphere(Oi, s, d) / d
        e = E @ Rb.T
        return np.stack([(e[..., 0] / e[..., 2] - Xmin) / PIX, (e[..., 1] / e[..., 2] - Ymin) / PIX], -1)

    def vel_grid(d, rlo, rhi, Rb):
        """Velocity range (pixels/day in the EBD frame for d) of bound TNO-like orbits with r in [rlo, rhi]: inclination
        direction within KBMOD's default +-pi/15 of the ecliptic, (v_t/v_c)^2 in 0.8-1.25, radial speed +-0.2 v_c."""
        V = []
        for r in np.linspace(rlo, rhi, 5):
            P0 = ray_sphere(Om, c_obs, r); ph = P0 / r
            e1 = np.cross(kecl, ph); e1 /= np.linalg.norm(e1); e2 = np.cross(ph, e1)
            vc = math.sqrt(GM / r)
            for beta in np.radians(np.linspace(-12, 12, 7)):
                for w in (0.8, 0.9, 1.0, 1.12, 1.25):
                    for f in (-0.2, 0.0, 0.2):
                        Vv = vc * math.sqrt(w) * (math.cos(beta) * e1 + math.sin(beta) * e2) + f * vc * ph
                        p0 = ebd_pix(P0 + Vv * (tw[0] - tmid), Ow[0], d, Rb); p1 = ebd_pix(P0 + Vv * (tw[-1] - tmid), Ow[-1], d, Rb)
                        V.append((p1 - p0) / max(Tw, 1e-3))
        V = np.array(V)
        sp = np.hypot(V[:, 0], V[:, 1]); an = np.arctan2(V[:, 1], V[:, 0])
        # ecliptic direction: circular orbit in the ecliptic at r = d
        P0 = ray_sphere(Om, c_obs, d); ph = P0 / d; e1 = np.cross(kecl, ph); e1 /= np.linalg.norm(e1)
        Vv = math.sqrt(GM / d) * e1
        v0 = (ebd_pix(P0 + Vv * (tw[-1] - tmid), Ow[-1], d, Rb) - ebd_pix(P0 + Vv * (tw[0] - tmid), Ow[0], d, Rb))
        ecl = math.atan2(v0[1], v0[0])
        da = (an - ecl + math.pi) % (2 * math.pi) - math.pi
        dv = TOL_V / max(Tw, 0.5)
        vmin, vmax = max(0.0, sp.min() - dv), sp.max() + dv
        amin, amax = da.min(), da.max()
        nv = int(math.ceil((vmax - vmin) / dv)) + 1
        na = int(math.ceil((amax - amin) * vmax / dv)) + 1
        if vmin < 2 * dv:                          # the velocity set reaches zero: angles span everything
            amin, amax = -math.pi, math.pi
            na = int(math.ceil(2 * math.pi * vmax / dv)) + 1
        return dict(ecl=ecl, vmin=vmin, vmax=vmax, nv=nv, amin=amin, amax=amax, na=na, v0=float(np.hypot(*v0)) / max(Tw, 1e-3))

    # ---- tiles of the EBD grid (start pixels), with padding for motion
    nW = len(win)
    TC = 8192 if nW <= 6 else (6144 if nW <= 10 else 4096)
    if DEBUG_NCCD:
        TC = 1024
    results = []                                    # dicts with lh, positions, state
    rate = None                                     # measured trajectory-image evaluations per second
    psf_k = [PSF.make_gaussian_kernel(math.sqrt((FWHM[e] / 2.3548) ** 2 + 1 / 3)) for e in win]
    var_w = [VAR[e] for e in win]
    times = tw.copy()
    ecl_cache = {}

    def to_candidates():
        res = sorted(results, key=lambda c: -c["lh"])
        keep = []
        for c in res:                               # one entry per body: within 12" at the first and last exposure
            dup = False
            for k in keep:
                if np.max(np.linalg.norm(c["sky"][[0, -1]] - k["sky"][[0, -1]], axis=1)) < 12.0 / ASEC:
                    dup = True; break
            if not dup:
                keep.append(c)
            if len(keep) >= 200:
                break
        out = []
        for c in keep:
            p = 1.0 / (1.0 + math.exp(-(c["lh"] - 9.0) / 1.5))
            out.append({"epoch_mjd": float(c["epoch"]), "state": [float(x) for x in c["state"]], "p": round(min(max(p, 1e-4), 0.9999), 5),
                        "mag": float(c["mag"]) if np.isfinite(c["mag"]) else None})
        return out

    def fit_state(sdir, d):
        """Heliocentric state reproducing the observed directions sdir [K,3] at the window's exposures: rays from the
        observer intersected with |P| = r, a straight line in emission time; r scanned, best bound fit kept."""
        bestf = None
        for r in sorted(set(np.r_[np.geomspace(26, 200, 60), d])):
            P = ray_sphere(Ow, sdir, r)
            te = tw - np.linalg.norm(P - Ow, axis=1) / C_AU
            A_ = np.vstack([np.ones_like(te), te - tmid]).T
            coef = np.linalg.lstsq(A_, P, rcond=None)[0]
            P0, V0 = coef[0], coef[1]
            rr_ = np.linalg.norm(P0)
            if rr_ < 25.5 or np.dot(V0, V0) / 2 - GM / rr_ >= -1e-9:
                continue
            pr = (A_ @ coef) - Ow
            pr /= np.linalg.norm(pr, axis=1, keepdims=True)
            err = float(np.max(np.arccos(np.clip(np.sum(pr * sdir, 1), -1, 1)))) * ASEC
            score = err + 0.02 * abs(math.log(r / d))
            if bestf is None or score < bestf[0]:
                bestf = (score, P0, V0, err)
        if bestf is None:
            return None
        return tmid, bestf[1], bestf[2], bestf[3]

    def run_tile(d, Rb, vg, tx0, ty0, pad):
        """Resample the window into the EBD frame for d over one padded tile, run KBMOD, convert the results."""
        nonlocal rate
        Wt = Ht = TC + 2 * pad
        # mapping EBD-grid pixel -> common-grid pixel for each window exposure, on a coarse lattice (float64)
        step = 16
        cx = np.arange(tx0 - pad, tx0 - pad + Wt + step, step, dtype=float)
        cy = np.arange(ty0 - pad, ty0 - pad + Ht + step, step, dtype=float)
        XX, YY = np.meshgrid(Xmin + cx * PIX, Ymin + cy * PIX)
        uE = (XX[..., None] * Rb[0] + YY[..., None] * Rb[1] + Rb[2]) / np.sqrt(1 + XX ** 2 + YY ** 2)[..., None]
        sci, var, nvalid0 = [], [], 0
        for j, e in enumerate(win):
            Pp = d * uE - O[e]
            q = Pp @ Rf.T
            gxp = (q[..., 0] / q[..., 2] - Xmin) / PIX; gyp = (q[..., 1] / q[..., 2] - Ymin) / PIX
            gx0 = int(max(0, math.floor(gxp.min()) - 2)); gx1 = int(min(W, math.ceil(gxp.max()) + 3))
            gy0 = int(max(0, math.floor(gyp.min()) - 2)); gy1 = int(min(H, math.ceil(gyp.max()) + 3))
            if gx1 - gx0 < 4 or gy1 - gy0 < 4:
                sci.append(np.full((Ht, Wt), np.nan, np.float32)); var.append(np.full((Ht, Wt), var_w[j], np.float32)); continue
            crop = S[e][gy0:gy1, gx0:gx1].float()
            val = torch.isfinite(crop)
            crop = torch.where(val, crop, torch.zeros_like(crop))
            m = torch.stack([torch.as_tensor((gxp - gx0) / max(gx1 - gx0 - 1, 1) * 2 - 1, device=dev),
                             torch.as_tensor((gyp - gy0) / max(gy1 - gy0 - 1, 1) * 2 - 1, device=dev)], 0).float()
            m = F.interpolate(m[None], size=((len(cy) - 1) * step + 1, (len(cx) - 1) * step + 1), mode="bilinear",
                              align_corners=True)[0]
            m = m[:, :Ht, :Wt].permute(1, 2, 0)[None]
            sv = F.grid_sample(torch.stack([crop, val.float()])[None], m, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
            im = torch.where(sv[1] > 0.999, sv[0], torch.full_like(sv[0], float("nan")))
            if j == 0:
                nvalid0 = int(torch.isfinite(im[pad:pad + TC, pad:pad + TC]).sum())
            sci.append(im.cpu().numpy()); var.append(np.full((Ht, Wt), var_w[j], np.float32))
            del crop, val, sv, im, m
        if dev == "cuda":
            torch.cuda.empty_cache()
        if nvalid0 < 0.02 * TC * TC or sum(np.isfinite(x[pad:pad + TC:64, pad:pad + TC:64]).mean() > 0.02 for x in sci) < 3:
            return 0
        stack = ImageStackPy(times=times, sci=sci, var=var, psfs=psf_k)
        gen_cfg = {"name": "EclipticCenteredSearch", "velocities": [vg["vmin"], vg["vmax"], vg["nv"]],
                   "angles": [vg["amin"], vg["amax"], vg["na"]], "angle_units": "radian", "velocity_units": "pix / d",
                   "given_ecliptic": vg["ecl"]}
        cfg = SearchConfiguration.from_dict({
            "generator_config": gen_cfg, "lh_level": LH, "num_obs": max(3, nW - 2), "sigmaG_filter": True,
            "gpu_filter": dev == "cuda", "sigmaG_lims": [25, 75], "do_clustering": True, "cluster_type": "nn_start_end",
            "cluster_eps": 20.0, "x_pixel_bounds": [pad, pad + TC], "y_pixel_bounds": [pad, pad + TC],
            "results_per_pixel": 1, "max_results": 5000, "near_dup_thresh": 10, "stamp_type": None, "coadds": [],
            "peak_offset_max": 3.0,
            "save_all_stamps": False, "compute_ra_dec": False, "max_masked_pixels": 1.0, "result_filename": None,
            "cpu_only": dev != "cuda"})
        gen = create_trajectory_generator(cfg)
        tt = time.time()
        res = SearchRunner().run_search(cfg, stack, trj_generator=gen)
        dt = time.time() - tt
        rate_now = TC * TC * vg["nv"] * vg["na"] * nW / max(dt, 1e-3)
        rate = rate_now if rate is None else 0.5 * (rate + rate_now)
        n_new = 0
        tab = res.table
        n_chi = 0
        if len(tab):
            xs = np.asarray(tab["x"], float); ys = np.asarray(tab["y"], float)
            vxs = np.asarray(tab["vx"], float); vys = np.asarray(tab["vy"], float)
            lhs = np.asarray(tab["likelihood"], float); fls = np.asarray(tab["flux"], float)
            ov = np.asarray(tab["obs_valid"], bool) if "obs_valid" in tab.colnames else np.ones((len(tab), nW), bool)
            # light-curve consistency (the job of the Kalman filter in Whidden et al. 2019 and of sigma-G now, which
            # cannot reject 2 outliers among 4-6 points): chi^2 of the per-image fluxes psi/phi about KBMOD's flux,
            # with a 20% allowance for colour, PSF and zero-point errors
            chi = np.zeros(len(tab))
            if "psi_curve" in tab.colnames and "phi_curve" in tab.colnames:
                ps = np.nan_to_num(np.asarray(tab["psi_curve"], float)); ph = np.nan_to_num(np.asarray(tab["phi_curve"], float))
                v = ov & (ph > 0)
                fhat = np.sum(ps * v, 1) / np.maximum(np.sum(ph * v, 1), 1e-30)
                f = np.where(v, ps / np.where(ph > 0, ph, 1), 0)
                ve = 1 / np.where(ph > 0, ph, 1) + (0.2 * fhat[:, None]) ** 2
                chi = np.sum(v * (f - fhat[:, None]) ** 2 / ve, 1) / np.maximum(v.sum(1) - 1, 1)
            n_chi = 0
            for k in np.argsort(-lhs)[:600]:            # the tile's best trajectories become candidates
                # nights with data under the trajectory (a hit needs the body's place on 3 nights, not 3 detections)
                pxk = np.rint(xs[k] + vxs[k] * (times - t0)).astype(int); pyk = np.rint(ys[k] + vys[k] * (times - t0)).astype(int)
                cov = [0 <= pxk[j] < Wt and 0 <= pyk[j] < Ht and np.isfinite(sci[j][pyk[j], pxk[j]]) for j in range(nW)]
                nn = len(set(nights_w[np.array(cov, bool)].tolist()))
                if nn < 3 or len(set(nights_w[ov[k]].tolist())) < 2:
                    continue
                if chi[k] > 3.0:
                    n_chi += 1; continue
                gx = tx0 - pad + xs[k] + vxs[k] * (times - t0)
                gy = ty0 - pad + ys[k] + vys[k] * (times - t0)
                X = Xmin + gx * PIX; Y = Ymin + gy * PIX
                uEk = (X[:, None] * Rb[0] + Y[:, None] * Rb[1] + Rb[2]) / np.sqrt(1 + X ** 2 + Y ** 2)[:, None]
                sdir = d * uEk - Ow
                sdir /= np.linalg.norm(sdir, axis=1, keepdims=True)
                fs = fit_state(sdir, d)
                if fs is None:
                    continue
                ep, P0, V0, err = fs
                if err > 1.0:
                    continue
                mag = ZREF - 2.5 * math.log10(fls[k]) if fls[k] > 0 else float("nan")
                results.append(dict(lh=float(lhs[k]), sky=sdir, epoch=ep, state=np.r_[P0, V0], mag=mag, d=d, nn=nn))
                n_new += 1
        log(f"  d {d:.1f} tile ({tx0},{ty0}): {len(tab)} KBMOD results, {n_new} kept ({n_chi} failed the light-curve test); search {dt:.1f} s "
            f"({rate_now:.3g} trajectory-images/s)")
        return dt

    LH = float(os.environ.get("KB_LH", 6.0))
    tiles = [(tx, ty) for ty in range(0, H, TC) for tx in range(0, W, TC)]
    npass = 0
    for (d, rlo, rhi) in cells:
        if time.time() > deadline - 20:
            break
        Rb = ebd_basis(d)
        vg = vel_grid(d, rlo, rhi, Rb)
        if DEBUG_NCCD:                              # development on a CPU: a token grid
            vg["nv"], vg["na"] = max(3, min(vg["nv"], 12)), max(3, min(vg["na"], 8))
        nvel = vg["nv"] * vg["na"]
        pad = int(math.ceil(vg["vmax"] * Tw)) + 16
        est = (len(tiles) * TC * TC * nvel * nW / rate) if rate else None
        log(f"EBD distance {d:.1f} au (cell {rlo:.1f}-{rhi:.1f}): {vg['nv']} speeds {vg['vmin']:.1f}-{vg['vmax']:.1f} px/d x "
            f"{vg['na']} angles {math.degrees(vg['amin']):.1f}..{math.degrees(vg['amax']):.1f} deg = {nvel} velocities, pad {pad}"
            + (f"; estimated {est:.0f} s for all tiles" if est else ""))
        narrowed = False
        for ti, (tx, ty) in enumerate(tiles):
            if time.time() > deadline - 10:
                log("time budget reached"); break
            # plan with the measured speed, or before the first tile with the A100's typical 9e10 trajectory-images/s
            r_plan = rate if rate is not None else (9e10 if dev == "cuda" else 3e7)
            if True:
                need = TC * TC * nvel * nW / r_plan
                left = len(tiles) - ti
                if npass == 0 and not narrowed and time.time() + 1.15 * left * (need + 3) > deadline:
                    narrowed = True
                    # the first (most populated) distance cell must cover the whole field: narrow its velocity grid
                    # around the circular-orbit ecliptic motion rather than leave tiles unsearched
                    f = max(0.02, (deadline - time.time()) / (1.15 * left) - 3) / need
                    if f < 1:
                        g = max(math.sqrt(f), 0.3)        # never below 30% of the range per axis; then tiles drop
                        vc_ = min(max(vg["v0"], vg["vmin"]), vg["vmax"]); hv = 0.5 * (vg["vmax"] - vg["vmin"]) * g
                        ha = 0.5 * (vg["amax"] - vg["amin"]) * g
                        ac_ = min(max(0.0, vg["amin"] + ha), vg["amax"] - ha)
                        vg.update(vmin=max(0.0, vc_ - hv), vmax=vc_ + hv, amin=ac_ - ha, amax=ac_ + ha,
                                  nv=max(3, int(round(vg["nv"] * g))), na=max(3, int(round(vg["na"] * g))))
                        nvel = vg["nv"] * vg["na"]
                        log(f"  narrowing the grid to {vg['nv']} x {vg['na']} = {nvel} velocities so the cell covers the field")
                        need = TC * TC * nvel * nW / r_plan
                if rate is not None and time.time() + need > deadline:
                    log(f"  skipping tile ({tx},{ty}): needs {need:.0f} s"); continue
            try:
                run_tile(d, Rb, vg, tx, ty, pad)
            except Exception as ex:
                log(f"  tile ({tx},{ty}) failed: {type(ex).__name__}: {ex}")
            write_out(a.out, to_candidates())
        npass += 1
    cands = to_candidates()
    write_out(a.out, cands)
    if inj_all is not None:                          # development: every kept trajectory, to see where the planted bodies rank
        json.dump([dict(lh=c["lh"], d=c["d"], nn=c["nn"], mag=c["mag"], epoch=c["epoch"], state=[float(x) for x in c["state"]])
                   for c in sorted(results, key=lambda c: -c["lh"])[:20000]], open(a.out + ".debug.json", "w"))
    log(f"done: {npass} distance cells searched, {len(results)} trajectories, {len(cands)} candidates written")


if __name__ == "__main__":
    main()
