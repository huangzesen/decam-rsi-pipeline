"""TNO finder: matched-filter detection per CCD -> static-source veto -> seed-pair x (distance, radial velocity)
orbit hypotheses scored on the GPU against S/N maps of every exposure -> orbit fit.  See NOTES.md."""
import argparse, csv, json, math, os, sys, time, warnings
import numpy as np

T0 = time.time()
GM = 2.959122082855911e-4
C_AU = 173.1446326846693
ASEC = 180 * 3600 / math.pi
EPS = math.radians(23.43928)
THR_LOW = 3.0          # catalogue threshold (support detections)
RAD = 1.5              # arcsec, match radius in the grid search
CELL = 3.0             # arcsec, lookup cell
AREA = 3.0 * 3600 ** 2 * 3.0   # arcsec^2 of one exposure (nominal)
PS = 0.526             # arcsec, pixel of the common S/N-map lattice (2 native pixels)


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------------------------- detection
def _terms(u, v):
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3], -1)


def detect_chunk(task):
    """Worker: detect sources on a few CCDs of one exposure.
    Returns (catalogue [n, 5] = X, Y (field frame, arcsec), snr, flux, fwhm_arcsec;
             list of (iy0, ix0, int8 S/N map resampled on the common lattice, -128 = no data))."""
    warnings.simplefilter("ignore")
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy import ndimage as ndi
    img_path, mask_path, hdus, thr, fwhm_fb, R, inj = task
    out = []; pieces = []
    try:
        hi = fits.open(img_path, memmap=False)
        hm = fits.open(mask_path, memmap=False) if mask_path else None
    except Exception:
        return np.zeros((0, 5)), []
    for k in hdus:
        try:
            d = np.asarray(hi[k].data, dtype=np.float32)
            hdr = hi[k].header
            if d.ndim != 2:
                continue
            ny, nx = d.shape
            # ---- cubic polynomial pixel <-> field-frame tangent plane (arcsec)
            w = WCS(hdr)
            gx, gy = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 17))
            gx = gx.ravel(); gy = gy.ravel()
            ra, dec = w.all_pix2world(gx, gy, 0)
            rr, dd = np.radians(ra), np.radians(dec)
            uu = np.stack([np.cos(dd) * np.cos(rr), np.cos(dd) * np.sin(rr), np.sin(dd)], 1) @ R.T
            if not np.all(np.isfinite(uu)) or uu[:, 2].min() < 0.9:
                continue
            GX = uu[:, 0] / uu[:, 2] * ASEC; GY = uu[:, 1] / uu[:, 2] * ASEC
            pu = lambda px: (px - nx / 2) / (nx / 2); pv = lambda py: (py - ny / 2) / (ny / 2)
            Af = _terms(pu(gx), pv(gy))
            cX = np.linalg.lstsq(Af, GX, rcond=None)[0]; cY = np.linalg.lstsq(Af, GY, rcond=None)[0]
            X0, Y0 = GX.mean(), GY.mean(); SX = max(np.ptp(GX), np.ptp(GY)) / 2
            Ai = _terms((GX - X0) / SX, (GY - Y0) / SX)
            cpx = np.linalg.lstsq(Ai, gx, rcond=None)[0]; cpy = np.linalg.lstsq(Ai, gy, rcond=None)[0]
            fw = hdr.get("FWHM", None)
            try:
                fw = float(fw)
            except Exception:
                fw = None
            if fw is None or not (1.5 < fw < 15):
                fw = fwhm_fb / 0.263
            s = min(max(fw / 2.3548, 1.0), 5.0)
            # ---- development only: inject synthetic point sources
            if inj is not None and len(inj):
                T = _terms((inj[:, 0] - X0) / SX, (inj[:, 1] - Y0) / SX)
                ipx = T @ cpx; ipy = T @ cpy
                for qx, qy, qf in zip(ipx, ipy, inj[:, 2]):
                    if 12 < qx < nx - 12 and 12 < qy < ny - 12 and abs(qx) < 1e5:
                        x0 = int(round(qx)); y0 = int(round(qy)); h = int(4 * s) + 2
                        yy, xx = np.mgrid[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1]
                        d[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1] += (qf / (2 * math.pi * s * s) * np.exp(-((xx - qx) ** 2 + (yy - qy) ** 2) / (2 * s * s))).astype(np.float32)
            good = np.isfinite(d)
            if hm is not None:
                try:
                    m = hm[hdr.get("EXTNAME", k)].data
                except Exception:
                    m = hm[k].data
                good &= (m == 0)
            d = np.where(good, d, np.float32(0))
            B = 128
            gyb, gxb = (ny + B - 1) // B, (nx + B - 1) // B
            sub = np.full((gyb * B // 4, gxb * B // 4), np.nan, np.float32)
            s4 = np.where(good[::4, ::4], d[::4, ::4], np.nan)
            sub[:s4.shape[0], :s4.shape[1]] = s4
            blk = sub.reshape(gyb, B // 4, gxb, B // 4).transpose(0, 2, 1, 3).reshape(gyb, gxb, -1)
            with np.errstate(all="ignore"):
                med = np.nanmedian(blk, axis=2)
                mad = 1.4826 * np.nanmedian(np.abs(blk - med[:, :, None]), axis=2)
            nval = np.isfinite(blk).sum(2)
            bad = ~np.isfinite(med) | (nval < 100)
            if bad.all():
                continue
            med[bad] = np.median(med[~bad]); mad[bad] = np.median(mad[~bad])
            med = ndi.median_filter(med, 3, mode="nearest"); mad = ndi.median_filter(mad, 3, mode="nearest")
            mad = np.maximum(mad, 1e-3)
            yy = (np.arange(ny) + 0.5) / B - 0.5; xx = (np.arange(nx) + 0.5) / B - 0.5
            coords = np.meshgrid(yy, xx, indexing="ij")
            bg = ndi.map_coordinates(med, coords, order=1, mode="nearest").astype(np.float32)
            sg = ndi.map_coordinates(mad, coords, order=1, mode="nearest").astype(np.float32)
            del coords
            d -= bg
            d *= good
            num = ndi.gaussian_filter(d, s, truncate=3.0, mode="constant")
            den = ndi.gaussian_filter(good.astype(np.float32), s / math.sqrt(2), truncate=3.0, mode="constant")
            norm = 1.0 / (4 * math.pi * s * s)
            # cosmic rays / sharp artifacts: a pixel far above the peak a PSF-shaped source of the locally fitted flux
            # could have (a point source never exceeds it; 2.5x allows for seeing better than the header FWHM)
            cr = (d > 4 * sg) & (d > 2.5 * num / (np.maximum(den, 0.05) * norm * 2 * math.pi * s * s) + 4 * sg)
            # cosmic-ray tracks (1 pixel wide): a pixel well above the noise whose neighbours are faint in at least one
            # of the four directions through it; a PSF is bright on all sides
            c_ = d[1:-1, 1:-1]
            mn = np.minimum(np.minimum(d[1:-1, :-2] + d[1:-1, 2:], d[:-2, 1:-1] + d[2:, 1:-1]),
                            np.minimum(d[:-2, :-2] + d[2:, 2:], d[:-2, 2:] + d[2:, :-2]))
            thr_ = 4.5 * sg[1:-1, 1:-1]
            cr[1:-1, 1:-1] |= (c_ > thr_) & (c_ - thr_ > 1.5 * mn)
            del mn, thr_, c_
            n_cr = int(cr.sum())
            if 0 < n_cr < 0.02 * cr.size:
                good &= ~ndi.binary_dilation(cr, iterations=1)
                d *= good
                num = ndi.gaussian_filter(d, s, truncate=3.0, mode="constant")
                den = ndi.gaussian_filter(good.astype(np.float32), s / math.sqrt(2), truncate=3.0, mode="constant")
            del cr
            snr = num / (sg * np.sqrt(np.maximum(den, 0.05) * norm))
            valid = den >= 0.8
            snr[~valid] = 0
            # ---- S/N map on the common lattice
            try:
                ix0 = int(math.ceil(GX.min() / PS)); ix1 = int(math.floor(GX.max() / PS))
                iy0 = int(math.ceil(GY.min() / PS)); iy1 = int(math.floor(GY.max() / PS))
                LX, LY = np.meshgrid(np.arange(ix0, ix1 + 1, dtype=np.float64) * PS, np.arange(iy0, iy1 + 1, dtype=np.float64) * PS)
                T = _terms((LX - X0) / SX, (LY - Y0) / SX)
                qx = (T @ cpx).astype(np.float32); qy = (T @ cpy).astype(np.float32)
                del T, LX, LY
                rs = ndi.map_coordinates(snr, [qy, qx], order=1, mode="constant", cval=0.0)
                rv = ndi.map_coordinates(valid.astype(np.float32), [qy, qx], order=1, mode="constant", cval=0.0)
                m8 = np.clip(np.rint(rs * 10), -127, 127).astype(np.int8)
                m8[rv < 0.999] = -128
                pieces.append((iy0, ix0, m8))
            except Exception as ex:
                sys.stderr.write(f"ccd {k} resample: {type(ex).__name__}: {ex}\n")
            mx = ndi.maximum_filter(snr, 3)
            pk = (snr >= thr) & (snr == mx)
            pk[:8] = False; pk[-8:] = False; pk[:, :8] = False; pk[:, -8:] = False
            py, px = np.nonzero(pk)
            if len(py) == 0:
                continue
            sv = snr[py, px]
            if len(py) > 40000:
                o = np.argsort(-sv)[:40000]; py, px, sv = py[o], px[o], sv[o]
            ds = 8
            cy, cx = ny // ds, nx // ds
            pool = snr[:cy * ds, :cx * ds].reshape(cy, ds, cx, ds).max(axis=(1, 3))
            pool = ndi.maximum_filter(pool, 5)
            nb = pool[np.minimum(py // ds, cy - 1), np.minimum(px // ds, cx - 1)]
            keep = ~((nb > 40) & (sv < 0.5 * nb))
            flux = num[py, px] / (np.maximum(den[py, px], 0.05) * norm)
            pkv = d[py, px]
            keep &= ~((pkv - 3 * sg[py, px]) > 0.6 * flux)
            py, px, sv, flux = py[keep], px[keep], sv[keep], flux[keep]
            if len(py) == 0:
                continue
            c0 = snr[py, px]
            def off(a, b):
                dd_ = a + b - 2 * c0
                return np.clip(np.where(dd_ < 0, 0.5 * (a - b) / np.where(dd_ < 0, dd_, -1), 0), -0.5, 0.5)
            ox = off(snr[py, px - 1], snr[py, px + 1])
            oy = off(snr[py - 1, px], snr[py + 1, px])
            T = _terms(pu(px + ox), pv(py + oy))
            out.append(np.stack([T @ cX, T @ cY, sv, flux, np.full(len(sv), fw * 0.263)], 1))
        except Exception as ex:   # a broken CCD must not kill the run
            sys.stderr.write(f"ccd {k}: {type(ex).__name__}: {ex}\n")
    hi.close()
    if hm is not None:
        hm.close()
    return (np.concatenate(out) if out else np.zeros((0, 5))), pieces


# ----------------------------------------------------------------------------------------------- orbit helpers (CPU, f64)
def propagate(P0, V0, dt, nstep=6):
    """Two-body RK4 from one state to many time offsets dt [K]. Returns P [K,3], V[K,3]."""
    K = len(dt)
    P = np.repeat(P0[None], K, 0).astype(float); V = np.repeat(V0[None], K, 0).astype(float)
    h = (dt / nstep)[:, None]
    def acc(p):
        return -GM * p / (np.sum(p * p, 1, keepdims=True) ** 1.5)
    for _ in range(nstep):
        k1p, k1v = V, acc(P)
        k2p, k2v = V + 0.5 * h * k1v, acc(P + 0.5 * h * k1p)
        k3p, k3v = V + 0.5 * h * k2v, acc(P + 0.5 * h * k2p)
        k4p, k4v = V + h * k3v, acc(P + h * k3p)
        P = P + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        V = V + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return P, V


def predict_xy(P0, V0, t0, t, O):
    """Gnomonic (x, y) arcsec in the field frame at observation times t [K] from observers O [K,3]."""
    lt = np.full(len(t), np.linalg.norm(P0 - O[0]) / C_AU)
    for _ in range(3):
        P, _ = propagate(P0, V0, t - lt - t0)
        D = P - O
        lt = np.linalg.norm(D, axis=1) / C_AU
    return np.stack([D[:, 0] / D[:, 2], D[:, 1] / D[:, 2]], 1) * ASEC


def q_to_state(q):
    Z = 1.0 / q[2]
    return np.array([q[0] * Z, q[1] * Z, Z]), np.array([q[3] * Z, q[4] * Z, q[5] * Z])


def state_to_q(P, V):
    Z = P[2]
    return np.array([P[0] / Z, P[1] / Z, 1 / Z, V[0] / Z, V[1] / Z, V[2] / Z])


def fit_orbit(P0, V0, t0, t, O, xy, sig):
    from scipy.optimize import least_squares
    def res(q):
        P, V = q_to_state(q)
        r = np.linalg.norm(P)
        if not np.isfinite(r) or q[2] <= 0:
            return np.full(2 * len(t) + 3, 1e3)
        pr = predict_xy(P, V, t0, t, O)
        vc = math.sqrt(GM / r)
        rdot = np.dot(P, V) / r
        en = np.dot(V, V) * r / (2 * GM)
        return np.concatenate([((pr - xy) / sig[:, None]).ravel(),
                               [rdot / (0.3 * vc), 20 * max(0.0, en - 0.9), 20 * max(0.0, 26.0 - r)]])
    q0 = state_to_q(P0, V0)
    try:
        s = least_squares(res, q0, x_scale=[1e-6, 1e-6, 1e-4, 1e-7, 1e-7, 1e-5], max_nfev=40)
        q = s.x
    except Exception:
        q = q0
    P, V = q_to_state(q)
    pr = predict_xy(P, V, t0, t, O)
    return P, V, np.hypot(*(pr - xy).T)


# ----------------------------------------------------------------------------------------------- main
def write_out(path, cands):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"candidates": cands[:200]}, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=900)
    a = ap.parse_args()
    budget = a.budget_s
    deadline = T0 + budget - 20
    write_out(a.out, [])
    warnings.simplefilter("ignore")

    rows = list(csv.DictReader(open(os.path.join(a.scenario, "exposures.csv"))))
    nE_all = len(rows)
    mjd = np.array([float(r["mjd_mid"]) for r in rows])
    night_all = np.round(mjd - 0.2).astype(int)
    ra0 = math.radians(float(np.median([float(r["ra_center"]) for r in rows])))
    # robust centre (RA wrap-safe)
    rac = np.array([math.radians(float(r["ra_center"])) for r in rows]); dec = np.array([math.radians(float(r["dec_center"])) for r in rows])
    cvec = np.stack([np.cos(dec) * np.cos(rac), np.cos(dec) * np.sin(rac), np.sin(dec)], 1).mean(0)
    cvec /= np.linalg.norm(cvec)
    ra0 = math.atan2(cvec[1], cvec[0]); dec0 = math.asin(cvec[2])
    ez = cvec; ex = np.array([-math.sin(ra0), math.cos(ra0), 0.0]); ey = np.cross(ez, ex)
    R = np.stack([ex, ey, ez])                       # equatorial -> field frame
    tref = 0.5 * (mjd.min() + mjd.max())

    COL = {"g": -0.6, "r": 0.0, "VR": 0.0, "i": 0.25, "z": 0.4, "Y": 0.45}
    mzh = []
    for r in rows:
        try:
            mzh.append(float(r["magzero"]))
        except Exception:
            mzh.append(float("nan"))
    mzh = np.array(mzh); bands_all = [r["band"] for r in rows]
    for e in range(nE_all):
        if not np.isfinite(mzh[e]) or not (20 < mzh[e] < 35):
            same = [mzh[k] for k in range(nE_all) if bands_all[k] == bands_all[e] and np.isfinite(mzh[k]) and 20 < mzh[k] < 35]
            allz = [mzh[k] for k in range(nE_all) if np.isfinite(mzh[k]) and 20 < mzh[k] < 35]
            mzh[e] = np.median(same) if same else (np.median(allz) if allz else 29.5)
    zph = mzh + np.array([COL.get(b_, 0.0) for b_ in bands_all])       # header zero point, colour-shifted to ~r
    O_all = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows]) @ R.T

    import torch
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log("device", dev)
    map_cap = 11e9 if dev == "cuda" else 6e9          # bytes of raw S/N maps kept on the device

    # ---- development only: synthetic bodies injected into the pixels
    inj_all = None
    if os.environ.get("FINDER_INJECT"):
        bodies = json.load(open(os.environ["FINDER_INJECT"]))
        inj_all = np.zeros((nE_all, len(bodies), 3))
        for b_, bd in enumerate(bodies):
            st = np.array(bd["state"], float)
            xy = predict_xy(R @ st[:3], R @ st[3:], bd["epoch_mjd"] - tref, mjd - tref, O_all)
            inj_all[:, b_, :2] = xy
            inj_all[:, b_, 2] = 10 ** (-0.4 * (bd["mag"] - zph))

    # ---- order exposures round-robin over nights so a time cut still covers every night
    by_night = {}
    for i in np.argsort(mjd):
        by_night.setdefault(night_all[i], []).append(i)
    order = []
    lists = [list(v) for v in by_night.values()]
    while any(lists):
        for l in lists:
            if l:
                order.append(l.pop(0))
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = 4
    ncpu = max(1, min(8, ncpu, int(os.environ.get("OMP_NUM_THREADS", "8") or 8)))
    os.environ["OMP_NUM_THREADS"] = "1"
    det_deadline = T0 + 0.5 * budget
    from concurrent.futures import ProcessPoolExecutor
    from astropy.io import fits
    dets = {}; maps = {}
    max_exp = 150
    map_bytes = 0
    cache = os.environ.get("FINDER_CACHE")
    cfile = os.path.join(cache, os.path.basename(os.path.normpath(a.scenario)) + ".pkl") if cache else None
    if cfile and os.path.exists(cfile):
        import pickle
        dets, maps = pickle.load(open(cfile, "rb")); order = []
    with ProcessPoolExecutor(ncpu) as pool:
        t_per = None
        for n_done, i in enumerate(order[:max_exp]):
            if t_per is not None and time.time() + 1.3 * t_per > det_deadline and n_done >= 3:
                log(f"detection time limit: {n_done}/{nE_all} exposures used"); break
            if map_bytes > map_cap:
                log(f"map memory limit: {n_done}/{nE_all} exposures used"); break
            t1 = time.time()
            r = rows[i]
            ip = os.path.join(a.scenario, r["image"]); mp = os.path.join(a.scenario, r["mask"]) if r.get("mask") else None
            try:
                with fits.open(ip) as h:
                    nh = len(h)
                    hd = [k for k in range(nh) if h[k].header.get("NAXIS", 0) == 2 or h[k].header.get("ZNAXIS", 0) == 2]
            except Exception as ex:
                log("cannot open", ip, ex); continue
            try:
                fb = float(r["seeing_arcsec"])
            except Exception:
                fb = 1.2
            nchunk = max(ncpu * 2, 1)
            tasks = [(ip, mp, hd[c::nchunk], THR_LOW, fb, R, None if inj_all is None else inj_all[i]) for c in range(nchunk) if hd[c::nchunk]]
            res = list(pool.map(detect_chunk, tasks))
            cats = [x[0] for x in res if len(x[0])]
            pieces = [p for x in res for p in x[1]]
            if not cats or not pieces:
                continue
            # assemble the exposure's S/N map on the common lattice (drop CCDs with a wild WCS)
            cy_ = np.median([p[0] + p[2].shape[0] / 2 for p in pieces]); cx_ = np.median([p[1] + p[2].shape[1] / 2 for p in pieces])
            lim = 1.3 * 3600 / PS
            pieces = [p for p in pieces if abs(p[0] - cy_) < lim and abs(p[1] - cx_) < lim and p[2].shape[0] < 6000 and p[2].shape[1] < 6000]
            if not pieces:
                continue
            y0 = min(p[0] for p in pieces); x0 = min(p[1] for p in pieces)
            H = max(p[0] + p[2].shape[0] for p in pieces) - y0; W = max(p[1] + p[2].shape[1] for p in pieces) - x0
            arr = np.full((H, W), -128, np.int8)
            for py_, px_, m8 in pieces:
                v = arr[py_ - y0:py_ - y0 + m8.shape[0], px_ - x0:px_ - x0 + m8.shape[1]]
                np.maximum(v, m8, out=v)
            del pieces, res
            dets[i] = np.concatenate(cats)
            maps[i] = (y0, x0, arr)
            map_bytes += arr.size
            t_per = time.time() - t1
            log(f"exp {i} night {night_all[i]} band {r['band']}: {len(dets[i])} detections, map {H}x{W}, {t_per:.1f}s")
    if cfile and not os.path.exists(cfile):
        import pickle
        pickle.dump((dets, maps), open(cfile, "wb"), protocol=4)
    used = sorted(k for k in dets if len(dets[k]) > 0)
    if len(used) < 3:
        log("too few exposures"); return
    E = len(used)
    t = mjd[used] - tref
    nights_u, night = np.unique(night_all[used], return_inverse=True)
    NN = len(nights_u)
    O = O_all[used]
    bands = [rows[i]["band"] for i in used]

    # ---- catalogue in the field frame
    cat = []
    for e, i in enumerate(used):
        d = dets[i]
        cat.append(np.column_stack([d, np.full(len(d), e)]))
    cat = np.concatenate(cat)
    cat = cat[np.all(np.isfinite(cat), 1) & (cat[:, 3] > 0)]
    X, Y, SNR, FLUX, FW, EXP = cat.T
    EXP = EXP.astype(int)
    N = len(cat)
    del dets
    log(f"{N} detections in {E} exposures, {NN} nights")

    # ---- zero points: header magzero (+ colour shift to ~r); exposures without one are tied to a same-band exposure
    # through the static sources they share
    from scipy.spatial import cKDTree
    zp = zph[used].copy()
    has_mz = []
    for i in used:
        try:
            has_mz.append(20 < float(rows[i]["magzero"]) < 35)
        except Exception:
            has_mz.append(False)
    for e in range(E):
        if has_mz[e]:
            continue
        me = np.nonzero((EXP == e) & (SNR > 20) & (SNR < 2000))[0]
        offs = []
        for k in range(E):
            if not has_mz[k] or bands[k] != bands[e] or len(me) < 30:
                continue
            mr = np.nonzero((EXP == k) & (SNR > 20) & (SNR < 2000))[0]
            if len(mr) < 30:
                continue
            dd_, ii_ = cKDTree(np.column_stack([X[mr], Y[mr]])).query(np.column_stack([X[me], Y[me]]), distance_upper_bound=0.7)
            okm = np.isfinite(dd_)
            if okm.sum() >= 30:
                offs.append(zp[k] + float(np.median(2.5 * np.log10(FLUX[me[okm]] / FLUX[mr[ii_[okm]]]))))
        if offs:
            zp[e] = float(np.median(offs))
            log(f"  exp {e}: no header magzero, self-calibrated zero point {zp[e] - COL.get(bands[e], 0.0):.2f}")
    FC = FLUX * 10 ** (-0.4 * (zp[EXP] - 25.0))              # flux in a common system (1 = mag 25)
    sigF = np.ones(E)
    for e in range(E):
        m = (EXP == e) & (SNR > 3.5) & (SNR < 50)
        if m.any():
            sigF[e] = float(np.median(FC[m] / SNR[m]))
    log("5-sigma depth (mag):", np.round(25 - 2.5 * np.log10(5 * sigF), 2).tolist())

    # ---- S/N maps on the device; pixel-level static mask from the other nights
    f32 = torch.float32
    tt = lambda v, dt=f32: torch.as_tensor(np.asarray(v), dtype=dt, device=dev)
    my0 = [maps[i][0] for i in used]; mx0 = [maps[i][1] for i in used]
    raw = [torch.from_numpy(maps[i][2]).to(dev) for i in used]
    del maps
    mH = [int(r_.shape[0]) for r_ in raw]; mW = [int(r_.shape[1]) for r_ in raw]
    masked = []
    for e in range(E):
        oth = [k for k in range(E) if night[k] != night[e]]
        if len(oth) > 12:
            oth = [oth[int(q)] for q in np.linspace(0, len(oth) - 1, 12)]
        # a static source is there on every other night; a slow body (near its stationary point) can sit on the same
        # pixel on one neighbouring night. So the strongest other night is left out of the stack when others exist.
        shp = (mH[e], mW[e]); i16 = torch.int16
        sm = torch.zeros(shp, dtype=i16, device=dev); cnt = torch.zeros(shp, dtype=i16, device=dev)
        top_s = torch.full(shp, -32000, dtype=i16, device=dev); top_c = torch.zeros(shp, dtype=i16, device=dev)
        mxv = torch.full(shp, -128, dtype=torch.int8, device=dev)
        for n in sorted(set(int(night[k]) for k in oth)):
            ns = torch.zeros(shp, dtype=i16, device=dev); nc = torch.zeros(shp, dtype=i16, device=dev)
            for k in oth:
                if night[k] != n:
                    continue
                ya = max(my0[e], my0[k]); yb = min(my0[e] + mH[e], my0[k] + mH[k])
                xa = max(mx0[e], mx0[k]); xb = min(mx0[e] + mW[e], mx0[k] + mW[k])
                if yb <= ya or xb <= xa:
                    continue
                sub = raw[k][ya - my0[k]:yb - my0[k], xa - mx0[k]:xb - mx0[k]]
                v = sub != -128
                se = (slice(ya - my0[e], yb - my0[e]), slice(xa - mx0[e], xb - mx0[e]))
                ns[se] += torch.where(v, sub, torch.zeros_like(sub)).to(i16)
                nc[se] += v.to(i16)
                mxv[se] = torch.maximum(mxv[se], sub)
            sm += ns; cnt += nc
            upd = (nc > 0) & (ns > top_s)
            top_s = torch.where(upd, ns, top_s); top_c = torch.where(upd, nc, top_c)
            del ns, nc, upd
        rest_c = (cnt - top_c).to(f32); rest_s = (sm - torch.clamp(top_s, min=-3000)).to(f32)
        # (only with >= 4 other nights: with fewer, the nights left differ too much in depth and band to stand alone)
        loo = (rest_c > 0) if len(set(int(night[k]) for k in oth)) >= 4 else torch.zeros(shp, dtype=torch.bool, device=dev)
        stat = torch.where(loo, rest_s > 24.0 * torch.sqrt(torch.clamp(rest_c, min=1.0)),
                           (sm.to(f32) > 28.0 * torch.sqrt(cnt.to(f32))) | (mxv >= 35)) | (cnt == 0)   # cnt == 0: cannot be checked
        del sm, cnt, top_s, top_c, rest_c, rest_s
        st = torch.nn.functional.max_pool2d(stat.to(f32)[None, None], 3, 1, 1)[0, 0] > 0
        st |= torch.nn.functional.max_pool2d(((mxv >= 125) & stat).to(f32)[None, None], 11, 1, 5)[0, 0] > 0
        del stat, mxv
        frac = float(st.to(f32).mean())
        masked.append(torch.where(st, torch.full_like(raw[e], -128), raw[e]))
        del st
        log(f"  exp {e}: static mask covers {100 * frac:.1f}% of the map")
    raw = masked
    del masked
    pooled = [torch.nn.functional.max_pool2d(r_.to(f32)[None, None], 3, 1, 1)[0, 0].to(torch.int8) for r_ in raw]
    rawf = [r_.reshape(-1) for r_ in raw]; poolf = [p_.reshape(-1) for p_ in pooled]

    def pool_at(k, x, y):
        """Max S/N in the 3x3 lattice pixels around (x, y) arcsec in exposure k; valid flag."""
        ix = torch.round(x / PS).long() - mx0[k]; iy = torch.round(y / PS).long() - my0[k]
        inb = (ix >= 0) & (ix < mW[k]) & (iy >= 0) & (iy < mH[k])
        v = poolf[k][(iy.clamp(0, mH[k] - 1) * mW[k] + ix.clamp(0, mW[k] - 1))]
        ok = inb & (v != -128)
        return v.to(f32) * 0.1, ok

    def bil_at(k, x, y):
        fx = x / PS - mx0[k]; fy = y / PS - my0[k]
        ix = torch.floor(fx); iy = torch.floor(fy)
        ax = fx - ix; ay = fy - iy
        ix = ix.long(); iy = iy.long()
        inb = (ix >= 0) & (ix < mW[k] - 1) & (iy >= 0) & (iy < mH[k] - 1)
        base = iy.clamp(0, mH[k] - 2) * mW[k] + ix.clamp(0, mW[k] - 2)
        v00 = rawf[k][base]; v01 = rawf[k][base + 1]; v10 = rawf[k][base + mW[k]]; v11 = rawf[k][base + mW[k] + 1]
        ok = inb & (v00 != -128) & (v01 != -128) & (v10 != -128) & (v11 != -128)
        val = (v00.to(f32) * (1 - ax) + v01.to(f32) * ax) * (1 - ay) + (v10.to(f32) * (1 - ax) + v11.to(f32) * ax) * ay
        return val * 0.1, ok

    # how surprising is a pooled S/N >= v at a random place of exposure k: -ln of the empirical tail probability
    # (noise + whatever junk survived the masks), tabulated in steps of 0.1 sigma
    postab = np.zeros((E, 128))
    for k in range(E):
        q = poolf[k][::23].cpu().numpy()
        hcnt = np.bincount(q[q != -128].astype(np.int64) + 127, minlength=255)[127:]      # values 0 .. 127
        tail = np.cumsum(hcnt[::-1])[::-1].astype(float)
        nall = max(float((q != -128).sum()), 1.0)
        postab[k] = np.clip(-np.log(np.maximum(tail, 0.5) / nall) + math.log(max(tail[0], 0.5) / nall), 0.0, 14.0)
        del q
    log("evidence of a pooled S/N of 3, 4, 5, 7:", [np.round(postab[k][[30, 40, 50, 70]], 1).tolist() for k in range(E)])
    postab_t = tt(postab)

    def evid(k, v, ok, mu, shift):
        """Log-likelihood ratio (body of expected S/N mu vs. no body) of a measured S/N v in exposure k."""
        pos = postab_t[k][torch.clamp(torch.round(v * 10), 0, 127).long()]
        neg = torch.clamp((v - shift - mu) ** 2 / (2 * (1 + (0.25 * mu) ** 2)), max=15.0)
        return torch.where(ok, pos - neg, torch.zeros_like(v))

    # ---- static-source veto
    tree = cKDTree(np.column_stack([X, Y]))
    pr = tree.query_pairs(2.5, output_type="ndarray")
    i1, i2 = pr[:, 0], pr[:, 1]
    d2 = (X[i1] - X[i2]) ** 2 + (Y[i1] - Y[i2]) ** 2
    dn = night[EXP[i1]] != night[EXP[i2]]
    static = np.zeros(N, bool)
    rad = np.where(np.maximum(SNR[i1], SNR[i2]) > 30, 2.5, 1.0)
    m = dn & (d2 < rad ** 2)
    # (gen001 vetoed every cross-night pair here; that also kills slow bodies, the pixel mask above is the veto now)
    del pr, i1, i2, d2, dn, m
    # anything within 6" of a very bright detection (any exposure) is star/galaxy structure
    bm = np.nonzero(SNR > 150)[0]
    if len(bm) > 1:
        bt = cKDTree(np.column_stack([X[bm], Y[bm]]))
        dd_, ii_ = bt.query(np.column_stack([X, Y]), k=2, distance_upper_bound=6.0)
        other = (np.isfinite(dd_[:, 0]) & (bm[np.minimum(ii_[:, 0], len(bm) - 1)] != np.arange(N))) | np.isfinite(dd_[:, 1])
        static |= other
    n_cat = int(static.sum()); n_crowd = 0
    dbg_idx = []
    if inj_all is not None:
        for b_ in range(inj_all.shape[1]):
            for e in range(E):
                xi, yi = inj_all[used[e], b_, :2]
                m = np.nonzero((EXP == e) & (np.abs(X - xi) < 1.0) & (np.abs(Y - yi) < 1.0) & (SNR > 4.5))[0]
                if len(m):
                    dbg_idx.append((b_, e, m[np.argmax(SNR[m])]))
        log("  inj lost to catalogue veto:", [(b_, e, round(float(SNR[d_]), 1)) for b_, e, d_ in dbg_idx if static[d_]])
        st0 = static.copy()
    for e in range(E):                      # pixel-level veto: the detection sits on a masked (static) lattice pixel
        m = np.nonzero(EXP == e)[0]
        ix_ = np.clip(np.rint(X[m] / PS).astype(np.int64) - mx0[e], 0, mW[e] - 1); iy_ = np.clip(np.rint(Y[m] / PS).astype(np.int64) - my0[e], 0, mH[e] - 1)
        okn = (rawf[e][torch.as_tensor(iy_ * mW[e] + ix_, device=dev)] != -128).cpu().numpy()
        static[m[~okn]] = True
    # crowded bright detections of one exposure (trails, ghosts, halos, bad rows): a real body is isolated
    for e in range(E):
        m = np.nonzero((EXP == e) & (SNR >= 4.5) & ~static)[0]
        if len(m) > 3:
            pts = np.column_stack([X[m], Y[m]])
            nnb = cKDTree(pts).query_ball_point(pts, 20.0, return_length=True) - 1
            exp_nb = len(m) / AREA * math.pi * 400.0
            bad = nnb >= max(3, int(math.ceil(exp_nb + 4 * math.sqrt(exp_nb))))
            static[m[bad]] = True; n_crowd += int(bad.sum())
    if inj_all is not None:
        log("  inj lost to crowding / pixel mask:", [(b_, e, round(float(SNR[d_]), 1)) for b_, e, d_ in dbg_idx if static[d_] and not st0[d_]])
    log(f"static veto: catalogue {n_cat}, crowded {n_crowd}, with pixel mask {static.sum()} of {N}; moving-candidate detections: {(~static).sum()}")
    keep = ~static
    X, Y, SNR, FLUX, FW, EXP, FC = X[keep], Y[keep], SNR[keep], FLUX[keep], FW[keep], EXP[keep], FC[keep]
    N = len(X)
    # ---- same-night groups: exposures of one night taken within ~18 minutes (a TNO moves < 1.3" in that time)
    groups = []
    for n in range(NN):
        cur = []
        for k in sorted(np.nonzero(night == n)[0], key=lambda k: t[k]):
            if cur and t[k] - t[cur[0]] > 0.0125:
                groups.append(cur); cur = []
            cur.append(int(k))
        if cur:
            groups.append(cur)
    partner = {k: [q for q in g_ if q != k] for g_ in groups for k in g_}
    wS_np = 1.0 / sigF
    # (a) a >= 4 sigma detection that the other exposure(s) of its group should show and do not is junk (cosmic ray)
    keepc = np.ones(N, bool)
    if not os.environ.get("FINDER_NOCONS"):
        for e in range(E):
            m = np.nonzero((EXP == e) & (SNR >= 4.0))[0]
            if not len(m) or not partner[e]:
                continue
            x_ = tt(X[m]); y_ = tt(Y[m]); fc_ = tt(FC[m])
            for k in partner[e]:
                v, ok = pool_at(k, x_, y_)
                bad = (ok & (v < 0.4 * fc_ * float(wS_np[k]) - 1.0)).cpu().numpy()
                if bad.mean() < 0.9:            # (a veto that removes everything means a wrong zero point, not junk)
                    keepc[m[bad]] = False
        log(f"same-night consistency veto: {(~keepc).sum()} of {(SNR >= 4.0).sum()} detections >= 4 sigma")
        X, Y, SNR, FLUX, FW, EXP, FC = X[keepc], Y[keepc], SNR[keepc], FLUX[keepc], FW[keepc], EXP[keepc], FC[keepc]
        N = len(X)
    # (b) night-stack seeds: peaks of the depth-weighted sum of a group's S/N maps, entered as pseudo-detections of the
    # group's deepest exposure (weight-2 bodies: below the single-exposure threshold, above it in a night's stack)
    STACK_THR = float(os.environ.get("FINDER_STACKTHR", 4.5))
    n_stack = []
    RIM = int(os.environ.get('FINDER_RIM', 7))
    if not os.environ.get("FINDER_NOSTACK"):
        for g_ in groups:
            if len(g_) < 2:
                continue
            try:
                e0 = min(g_, key=lambda k: sigF[k])
                shp = (mH[e0], mW[e0])
                num = torch.zeros(shp, dtype=f32, device=dev); den = torch.zeros(shp, dtype=f32, device=dev)
                cnt = torch.zeros(shp, dtype=torch.int8, device=dev)
                for k in g_:
                    ya = max(my0[e0], my0[k]); yb = min(my0[e0] + mH[e0], my0[k] + mH[k])
                    xa = max(mx0[e0], mx0[k]); xb = min(mx0[e0] + mW[e0], mx0[k] + mW[k])
                    if yb <= ya or xb <= xa:
                        continue
                    sub = raw[k][ya - my0[k]:yb - my0[k], xa - mx0[k]:xb - mx0[k]]
                    v = sub != -128
                    se = (slice(ya - my0[e0], yb - my0[e0]), slice(xa - mx0[e0], xb - mx0[e0]))
                    w_ = float(wS_np[k])
                    num[se] += torch.where(v, sub.to(f32) * (0.1 * w_), torch.zeros((), dtype=f32, device=dev))
                    den[se] += v.to(f32) * (w_ * w_)
                    cnt[se] += v.to(torch.int8)
                    del sub, v
                Zs = num / torch.sqrt(den.clamp(min=1e-20))
                del num
                pk = (cnt >= 2) & (Zs >= STACK_THR)
                pk &= Zs >= torch.nn.functional.max_pool2d(Zs[None, None], 3, 1, 1)[0, 0]
                # not next to blanked pixels: the wings of a masked static source peak at the rim of its mask
                rim = torch.nn.functional.max_pool2d((raw[e0] == -128).to(f32)[None, None], RIM, 1, RIM // 2)[0, 0] > 0
                n_rim = int((pk & rim).sum())
                pk &= ~rim
                del rim
                pk[:2] = False; pk[-2:] = False; pk[:, :2] = False; pk[:, -2:] = False
                if inj_all is not None:            # development: the stack at the injected bodies
                    for b_ in range(inj_all.shape[1]):
                        xi, yi = inj_all[used[e0], b_, :2]
                        ix_ = int(round(xi / PS)) - mx0[e0]; iy_ = int(round(yi / PS)) - my0[e0]
                        if 2 <= ix_ < mW[e0] - 2 and 2 <= iy_ < mH[e0] - 2:
                            zz = float(Zs[iy_ - 1:iy_ + 2, ix_ - 1:ix_ + 2].max())
                            if zz >= 3.5:
                                log(f"    inj {b_} stack group {g_}: Z {zz:.1f} cnt {int(cnt[iy_, ix_])} peak kept {bool(pk[iy_ - 1:iy_ + 2, ix_ - 1:ix_ + 2].any())} n_valid {[int((raw[k_][iy_ + my0[e0] - my0[k_] - 1:iy_ + my0[e0] - my0[k_] + 2, ix_ + mx0[e0] - mx0[k_] - 1:ix_ + mx0[e0] - mx0[k_] + 2] != -128).sum()) for k_ in g_]}")
                idx = torch.nonzero(pk.reshape(-1))[:, 0]
                del pk, cnt
                Zf = Zs.reshape(-1); z0 = Zf[idx]
                flx = z0 / torch.sqrt(den.reshape(-1)[idx])
                W_ = mW[e0]
                def off(a_, b_):
                    dd_ = a_ + b_ - 2 * z0
                    return torch.clamp(torch.where(dd_ < 0, 0.5 * (a_ - b_) / torch.where(dd_ < 0, dd_, -torch.ones_like(dd_)), torch.zeros_like(dd_)), -0.5, 0.5)
                ox = off(Zf[idx - 1], Zf[idx + 1]); oy = off(Zf[idx - W_], Zf[idx + W_])
                xs = ((idx % W_).to(f32) + ox + mx0[e0]) * PS; ys = ((idx // W_).to(f32) + oy + my0[e0]) * PS
                good = torch.ones_like(z0, dtype=torch.bool)
                for k in g_:                       # every member must show its share (kills one-exposure junk)
                    vb, okb = bil_at(k, xs, ys)
                    mu_ = flx * float(wS_np[k])       # (tested only where the member should show clearly)
                    good &= (~okb) | (mu_ < 2.5) | (vb >= 0.45 * mu_)
                    good &= (~okb) | (vb < 12.0)      # (int8 maps saturate at 12.7: plateaus of bright blobs; those are catalogue seeds)
                xs = xs[good].cpu().numpy().astype(np.float64); ys = ys[good].cpu().numpy().astype(np.float64)
                z0 = z0[good].cpu().numpy().astype(np.float64); flx = flx[good].cpu().numpy().astype(np.float64)
                log(f"  stack group {g_}: {n_rim} peaks at mask rims dropped")
                log(f"  stack group {g_}: peaks >= {STACK_THR}: {len(z0)}, >= 4.5: {(z0 >= 4.5).sum()}, >= 5: {(z0 >= 5).sum()}, >= 6: {(z0 >= 6).sum()}")
                del Zs, Zf, den, idx
                # not already a catalogue seed of the anchor exposure
                mm = np.nonzero((EXP == e0) & (SNR >= 4.5))[0]
                if len(mm) and len(xs):
                    dd_, _ = cKDTree(np.column_stack([X[mm], Y[mm]])).query(np.column_stack([xs, ys]), distance_upper_bound=1.2)
                    nw = ~np.isfinite(dd_)
                    xs, ys, z0, flx = xs[nw], ys[nw], z0[nw], flx[nw]
                if len(xs) > 3:                    # crowded peaks (halos, trails, ghosts): a real body is isolated
                    pts = np.column_stack([xs, ys])
                    nnb = cKDTree(pts).query_ball_point(pts, 20.0, return_length=True) - 1
                    exp_nb = len(xs) / AREA * math.pi * 400.0
                    okc_ = nnb < max(2, int(math.ceil(exp_nb + 4 * math.sqrt(exp_nb))))
                    xs, ys, z0, flx = xs[okc_], ys[okc_], z0[okc_], flx[okc_]
                if len(xs) > 20000:
                    o = np.argsort(-z0)[:20000]; xs, ys, z0, flx = xs[o], ys[o], z0[o], flx[o]
                if len(xs):
                    fw0 = float(np.median(FW[EXP == e0])) if (EXP == e0).any() else 1.2
                    X = np.concatenate([X, xs]); Y = np.concatenate([Y, ys]); SNR = np.concatenate([SNR, z0])
                    FC = np.concatenate([FC, flx]); FLUX = np.concatenate([FLUX, flx * 10 ** (0.4 * (zp[e0] - 25.0))])
                    FW = np.concatenate([FW, np.full(len(xs), fw0)]); EXP = np.concatenate([EXP, np.full(len(xs), e0, dtype=EXP.dtype)])
                if os.environ.get("FINDER_DUMP"):
                    np.save(os.path.join(os.environ["FINDER_DUMP"], f"stack_{e0}.npy"), np.column_stack([xs, ys, z0]))
                n_stack.append((e0, len(g_), len(xs)))
            except Exception as ex:
                log("night-stack seeds failed:", type(ex).__name__, ex)
        N = len(X)
        log("night-stack seeds (anchor exposure, members, new seeds):", n_stack)
    MAG = 25.0 - 2.5 * np.log10(np.maximum(FC, 1e-6))
    MTOL = 0.8 + 2.7 / SNR
    # chance-match log-likelihood of each detection
    LLR = np.zeros(N)
    for e in range(E):
        m = np.nonzero(EXP == e)[0]
        o = np.argsort(-SNR[m])
        rank = np.empty(len(m)); rank[o] = np.arange(1, len(m) + 1)
        LLR[m] = np.clip(-np.log(rank * math.pi * RAD ** 2 / AREA), 0.0, 14.0)
    for e in range(E):
        m = EXP == e
        log(f"  exp {e}: n={m.sum()} n(>4.5)={(SNR[m] > 4.5).sum()} n(>5)={(SNR[m] > 5).sum()} n(>7)={(SNR[m] > 7).sum()}")

    # ---- GPU search
    wS = tt(1.0 / sigF)
    # unit vectors of detections
    nrm = np.sqrt(1 + (X / ASEC) ** 2 + (Y / ASEC) ** 2)
    UX, UY, UZ = X / ASEC / nrm, Y / ASEC / nrm, 1 / nrm
    Ot = tt(O); tT = tt(t)
    kecl = tt(R @ np.array([0.0, -math.sin(EPS), math.cos(EPS)]))
    night_t = torch.as_tensor(night, device=dev)

    # seeds
    seed_thr = 4.5
    SEEDMAX = 4000 if dev == 'cuda' else 2500
    K1 = int(os.environ.get('FINDER_K1', 1000 if dev == 'cuda' else 500))    # tier-1 seeds per exposure
    seeds = []
    noseed = np.zeros(N, bool)
    try:
        cs = np.nonzero(SNR >= 4.2)[0]
        prs = cKDTree(np.column_stack([X[cs], Y[cs]])).query_pairs(1.0, output_type="ndarray")
        far = np.abs(t[EXP[cs[prs[:, 0]]]] - t[EXP[cs[prs[:, 1]]]]) >= 4.0
        noseed[cs[prs[far, 0]]] = True; noseed[cs[prs[far, 1]]] = True
        log(f"seeds dropped as static over >= 4 days: {noseed.sum()}")
    except Exception as ex:
        log("static-seed veto failed:", type(ex).__name__, ex)
    for e in range(E):
        m = np.nonzero((EXP == e) & (SNR >= seed_thr) & (SNR < 400) & ~noseed)[0]
        m = m[np.argsort(-SNR[m])]                   # sorted by S/N: the index of a seed is its rank
        if len(m) > SEEDMAX:                         # over the cap: the brightest 40 %, and a random sample of the rest
            nt = int(0.4 * SEEDMAX)                  # (the brightest are mostly junk; faint seeds must keep a chance)
            pick = np.sort(np.random.default_rng(1).choice(len(m) - nt, SEEDMAX - nt, replace=False))
            m = np.concatenate([m[:nt], m[nt:][pick]])
        seeds.append(m)
    log("seeds per exposure:", [len(s) for s in seeds])
    inj_seed = {}
    if inj_all is not None:                  # development: where are the injected bodies in the catalogue?
        for b_ in range(inj_all.shape[1]):
            msg = []
            for e in range(E):
                xi, yi = inj_all[used[e], b_, :2]
                m = np.nonzero((EXP == e) & (np.abs(X - xi) < 1.0) & (np.abs(Y - yi) < 1.0))[0]
                if len(m):
                    d_ = m[np.argmax(SNR[m])]
                    isseed = d_ in set(seeds[e].tolist())
                    if isseed:
                        inj_seed[(e, int(np.nonzero(seeds[e] == d_)[0][0]))] = b_
                    msg.append(f"{SNR[d_]:.1f}{'S' if isseed else ''}")
                else:
                    v_, ok_ = pool_at(e, tt([xi]), tt([yi]))
                    msg.append(f"({float(v_[0]):.1f})" if bool(ok_[0]) else "x")
            log(f"  inj {b_}: {msg}")

    def helio(ux, uy, uz, Oe, r):
        """Position of a body at heliocentric distance r seen along u from Oe. All broadcastable."""
        b = Oe[0] * ux + Oe[1] * uy + Oe[2] * uz
        O2 = float(Oe[0] ** 2 + Oe[1] ** 2 + Oe[2] ** 2)
        rho = -b + torch.sqrt(torch.clamp(b * b - O2 + r * r, min=0))
        return Oe[0] + rho * ux, Oe[1] + rho * uy, Oe[2] + rho * uz, rho

    def predict(Pa, Va, g, ta, rhoa, k):
        """Gnomonic xy (arcsec) at exposure k of bodies with state (Pa, Va) at emission time ta."""
        s = tT[k] - rhoa / C_AU - ta
        for it in range(2):
            px = Pa[0] + Va[0] * s - 0.5 * g * Pa[0] * s * s - Ot[k, 0]
            py = Pa[1] + Va[1] * s - 0.5 * g * Pa[1] * s * s - Ot[k, 1]
            pz = Pa[2] + Va[2] * s - 0.5 * g * Pa[2] * s * s - Ot[k, 2]
            if it == 0:
                rho = torch.sqrt(px * px + py * py + pz * pz)
                s = tT[k] - rho / C_AU - ta
        return px / pz * ASEC, py / pz * ASEC

    def setup_pair(i, j, r, rdot_frac, ai, bj):
        """States for triples (seed ai in exp i, seed bj in exp j, r, rdot as fraction of v_circ)."""
        vc = torch.sqrt(GM / r)
        dt0 = float(t[j] - t[i])
        pax, pay, paz, rhoa = helio(sux[i][ai], suy[i][ai], suz[i][ai], O[i], r)
        rb = r + rdot_frac * vc * dt0
        pbx, pby, pbz, rhob = helio(sux[j][bj], suy[j][bj], suz[j][bj], O[j], rb)
        ta = tT[i] - rhoa / C_AU
        dtau = dt0 - (rhob - rhoa) / C_AU
        g = GM / (r * r * r)
        vx = (pbx - pax) / dtau + 0.5 * g * pax * dtau
        vy = (pby - pay) / dtau + 0.5 * g * pay * dtau
        vz = (pbz - paz) / dtau + 0.5 * g * paz * dtau
        return (pax, pay, paz), (vx, vy, vz), g, ta, rhoa

    sux = [tt(UX[s]) for s in seeds]; suy = [tt(UY[s]) for s in seeds]; suz = [tt(UZ[s]) for s in seeds]
    sl = [tt(LLR[s]) for s in seeds]; smag = [tt(MAG[s]) for s in seeds]; stol = [tt(MTOL[s]) for s in seeds]
    sX = [tt(X[s]) for s in seeds]; sY = [tt(Y[s]) for s in seeds]; sF = [tt(FC[s]) for s in seeds]

    # exposure pairs on different nights, shortest first; use at most 2 exposures per night as seed exposures
    seed_exp = []
    for n in range(NN):
        ee = np.nonzero(night == n)[0]
        ee = ee[np.argsort([-len(seeds[e]) for e in ee])]   # (most seeds first is arbitrary but deterministic)
        seed_exp += list(ee[:2])
    pairs = [(i, j) for i in seed_exp for j in seed_exp if night[i] < night[j] and len(seeds[i]) and len(seeds[j])]
    prim = {n: min((k for k in seed_exp if night[k] == n), key=lambda k: sigF[k]) for n in set(int(night[k]) for k in seed_exp)}
    pairs.sort(key=lambda p: (0 if (prim[int(night[p[0]])] == p[0] and prim[int(night[p[1]])] == p[1]) else 1, t[p[1]] - t[p[0]]))
    link_deadline = min(deadline - 0.12 * budget, T0 + 0.62 * budget)
    GAM_MAX, GAM_MIN = 1 / 25.5, 1 / 400.0
    TOL = 1.6
    results = []      # arrays [score, nn, i, j, a, b, r, rdotfrac]
    pair_step = {}
    n_tr_total = 0
    # two tiers: first every exposure pair with the K1 most significant seeds of each exposure (so that every night
    # pair is covered whatever the time), then the remaining seed pairs, shortest time difference first
    t_link0 = time.time()
    tier1_deadline = t_link0 + 0.6 * (link_deadline - t_link0)
    jobs = [(0, p_) for p_ in pairs] + [(1, p_) for p_ in pairs]
    for ip, (tier, (i, j)) in enumerate(jobs):
        now = time.time()
        if now > link_deadline:
            log(f"link time limit after {ip}/{len(jobs)} jobs"); break
        if tier == 0:
            left = sum(1 for q_ in jobs[ip:] if q_[0] == 0)
            pair_deadline = min(link_deadline, now + 2.0 * max(tier1_deadline - now, 0.0) / left + 3.0)
        else:
            if len(seeds[i]) <= K1 and len(seeds[j]) <= K1:
                continue
            pair_deadline = min(link_deadline, now + 2.0 * (link_deadline - now) / (len(jobs) - ip) + 3.0)
        dt0 = float(t[j] - t[i])
        # grid steps from the numerical sensitivity of the predictions (a central track, r = 30 and 60 au)
        others = [k for k in range(E) if night[k] != night[i] and night[k] != night[j]]
        if not others:
            continue
        z1 = torch.zeros(1, device=dev)
        def cen(r_, f_):
            Pa, Va, g, ta, rhoa = setup_pair_c(i, j, r_, f_)
            return torch.stack([torch.cat(predict(Pa, Va, g, ta, rhoa, k)) for k in others])
        def setup_pair_c(i, j, r_, f_):
            r_ = torch.tensor([r_], device=dev, dtype=f32); f_ = torch.tensor([f_], device=dev, dtype=f32)
            vc = torch.sqrt(GM / r_)
            pax, pay, paz, rhoa = helio(z1, z1, z1 + 1, O[i], r_)
            rb = r_ + f_ * vc * dt0
            # a point displaced by a circular-orbit motion along x
            ang = float(vc / r_) * dt0
            pbx, pby, pbz, rhob = helio(z1 + math.sin(ang), z1, z1 + math.cos(ang), O[j], rb)
            # (direction as seen from the Sun used as topocentric direction: adequate for a sensitivity estimate)
            ta = tT[i] - rhoa / C_AU; dtau = dt0 - (rhob - rhoa) / C_AU; g = GM / r_ ** 3
            V = tuple((b - a_) / dtau + 0.5 * g * a_ * dtau for a_, b in ((pax, pbx), (pay, pby), (paz, pbz)))
            return (pax, pay, paz), V, g, ta, rhoa
        sens_g = 0.0; sens_f = 0.0
        for r_ in (30.0, 60.0):
            dg = 2e-5
            p0 = cen(r_, 0.0); p1 = cen(1 / (1 / r_ - dg), 0.0); p2 = cen(r_, 0.1)
            sens_g = max(sens_g, float((p1 - p0).norm(dim=1).max()) / dg)
            sens_f = max(sens_f, float((p2 - p0).norm(dim=1).max()) / 0.1 * (r_ / 30.0) ** 0.0)
        ng = int(min(max((GAM_MAX - GAM_MIN) * sens_g / TOL, 8), 1500))
        nf = int(min(max(2 * 0.45 * sens_f / TOL, 1), 9)) | 1
        gam = torch.linspace(GAM_MAX, GAM_MIN, ng, device=dev)
        pair_step[(i, j)] = (GAM_MAX - GAM_MIN) / max(ng - 1, 1)
        rgrid = 1.0 / gam
        fgrid = torch.linspace(-0.45, 0.45, nf, device=dev) if nf > 1 else torch.zeros(1, device=dev)
        Ni, Nj = len(seeds[i]), len(seeds[j])
        n_tr = 0
        sep2 = (sX[i][:, None] - sX[j][None, :]) ** 2 + (sY[i][:, None] - sY[j][None, :]) ** 2
        cm = (sep2 < (150.0 * dt0 + 10) ** 2) & ((smag[i][:, None] - smag[j][None, :]).abs() < stol[i][:, None] + stol[j][None, :])
        t1m = (torch.arange(Ni, device=dev)[:, None] < K1) & (torch.arange(Nj, device=dev)[None, :] < K1)
        cm &= t1m if tier == 0 else ~t1m
        cand = torch.nonzero(cm)
        del t1m
        del sep2, cm
        Nc = len(cand)
        if Nc == 0:
            continue
        ca, cb = cand[:, 0], cand[:, 1]
        # coarse-to-fine: the bound / prograde cut is first tested for a block of G distances at once (both ends of
        # the block, rdot = 0, wide margins); only the surviving seed pairs get the full (r, rdot) grid of the block
        G = 8 if ng >= 64 else 1
        best_pair = []
        c_done = 0
        nblk = (ng + G - 1) // G
        nb = max(1, int(4e6 // Nc))                 # distance blocks handled in one batch
        gi = torch.arange(G, device=dev)
        for b0 in range(0, nblk, nb):
            if time.time() > pair_deadline:
                break
            c_done = min((b0 + nb) * G, ng)
            blk = torch.arange(b0, min(b0 + nb, nblk), device=dev)
            if G > 1:
                r_lo = rgrid[blk * G][:, None]; r_hi = rgrid[torch.clamp(blk * G + G - 1, max=ng - 1)][:, None]
                sb = []; sp = []
                for p0 in range(0, Nc, 4000000):
                    ca_ = ca[p0:p0 + 4000000][None, :]; cb_ = cb[p0:p0 + 4000000][None, :]
                    lo = hi_ = cm_ = None
                    for r_end in (r_lo, r_hi):
                        Pa, Va, g, ta, rhoa = setup_pair(i, j, r_end, 0.0, ca_, cb_)
                        v2 = (Va[0] ** 2 + Va[1] ** 2 + Va[2] ** 2) / (GM / r_end)
                        hx = Pa[1] * Va[2] - Pa[2] * Va[1]; hy = Pa[2] * Va[0] - Pa[0] * Va[2]; hz = Pa[0] * Va[1] - Pa[1] * Va[0]
                        cosi = (hx * kecl[0] + hy * kecl[1] + hz * kecl[2]) / torch.sqrt(hx * hx + hy * hy + hz * hz)
                        if lo is None:
                            lo, hi_, cm_ = v2, v2, cosi
                        else:
                            lo = torch.minimum(lo, v2); hi_ = torch.maximum(hi_, v2); cm_ = torch.maximum(cm_, cosi)
                    okc = (lo * 0.8 < 1.9) & (hi_ * 1.2 > 0.05) & (cm_ > 0.42)
                    sel = torch.nonzero(okc)
                    sb.append(blk[sel[:, 0]]); sp.append(sel[:, 1] + p0)
                    del Pa, Va, v2, hx, hy, hz, cosi, lo, hi_, cm_, okc, sel
                sb = torch.cat(sb); sp = torch.cat(sp)
            else:
                sb = blk.repeat_interleave(Nc); sp = torch.arange(Nc, device=dev).repeat(len(blk))
            if len(sp) == 0:
                continue
            per = max(1, 3000000 // (G * nf))
            for s0 in range(0, len(sp), per):
                a_ = ca[sp[s0:s0 + per]]; b_ = cb[sp[s0:s0 + per]]
                ridx = sb[s0:s0 + per][:, None] * G + gi[None, :]
                r3 = rgrid[ridx.clamp(max=ng - 1)][:, :, None]
                Pa, Va, g, ta, rhoa = setup_pair(i, j, r3, fgrid[None, None, :], a_[:, None, None], b_[:, None, None])
                v2 = Va[0] ** 2 + Va[1] ** 2 + Va[2] ** 2
                vc2 = GM / r3
                hx = Pa[1] * Va[2] - Pa[2] * Va[1]; hy = Pa[2] * Va[0] - Pa[0] * Va[2]; hz = Pa[0] * Va[1] - Pa[1] * Va[0]
                hn = torch.sqrt(hx * hx + hy * hy + hz * hz)
                cosi = (hx * kecl[0] + hy * kecl[1] + hz * kecl[2]) / hn
                ok = (v2 < 1.9 * vc2) & (v2 > 0.25 * vc2) & (cosi > 0.5) & (ridx < ng)[:, :, None]
                sel = torch.nonzero(ok)
                del Pa, Va, v2, hx, hy, hz, hn, cosi, ok
                if len(sel) == 0:
                    continue
                for q0 in range(0, len(sel), 3000000):
                    q = sel[q0:q0 + 3000000]
                    rq = r3[q[:, 0], q[:, 1], 0]; fv = fgrid[q[:, 2]]; aq = a_[q[:, 0]]; bq = b_[q[:, 0]]
                    Pa, Va, g, ta, rhoa = setup_pair(i, j, rq, fv, aq, bq)
                    score = sl[i][aq] + sl[j][bq]
                    fq = 0.425 * (sF[i][aq] + sF[j][bq])          # 0.85 x mean seed flux (seeds are biased bright)
                    nn = torch.full_like(score, 2.0)
                    for n in range(NN):
                        seedn = (n == night[i] or n == night[j])
                        hit = torch.zeros_like(score)
                        for k in np.nonzero(night == n)[0]:
                            k = int(k)
                            if k == i or k == j:
                                continue
                            x, y = predict(Pa, Va, g, ta, rhoa, k)
                            v, ok = pool_at(k, x, y)
                            score = score + evid(k, v, ok, fq * wS[k], 0.5)
                            if not seedn:
                                hit = torch.maximum(hit, (ok & (v >= 3.0)).to(f32))
                        if not seedn:
                            nn = nn + hit
                    m = nn >= 3
                    n_tr += len(q)
                    if m.any():
                        mi = torch.nonzero(m)[:, 0]
                        sc = score[mi]
                        if len(mi) > 20000:
                            top = torch.topk(sc, 20000).indices; mi = mi[top]; sc = sc[top]
                        arr = torch.stack([sc, nn[mi], aq[mi].to(f32), bq[mi].to(f32), rq[mi], fv[mi]], 1).cpu().numpy()
                        best_pair.append(arr)
        n_tr_total += n_tr
        if best_pair:
            arr = np.concatenate(best_pair)
            # keep the best hypothesis per seed pair
            key = arr[:, 2].astype(np.int64) * 100000 + arr[:, 3].astype(np.int64)
            o = np.lexsort((-arr[:, 0], key))
            arr = arr[o]; key = key[o]
            first = np.concatenate([[True], key[1:] != key[:-1]])
            arr = arr[first]
            arr = arr[np.argsort(-arr[:, 0])[:3000]]
            results.append(np.column_stack([arr[:, 0], arr[:, 1], np.full(len(arr), i), np.full(len(arr), j), arr[:, 2:]]))
        log(f"job {ip} tier {tier} ({i},{j}) dt={dt0:.2f}d ng={min(c_done, ng)}/{ng} nf={nf} seeds {Ni}x{Nj} cand={Nc} triples={n_tr:.3g} kept={len(results[-1]) if best_pair else 0}")
    log(f"total hypotheses scored: {n_tr_total:.3g}")
    if not results:
        log("no linkages"); return
    res = np.concatenate(results)
    res = res[np.argsort(-res[:, 0])]
    if inj_seed:
        best = {}
        for rk, row in enumerate(res):
            ba = inj_seed.get((int(row[2]), int(row[4]))); bb = inj_seed.get((int(row[3]), int(row[5])))
            if ba is not None and ba == bb and ba not in best:
                best[ba] = (rk, round(float(row[0]), 1), int(row[2]), int(row[3]), round(float(row[6]), 1))
        log("  inj linked (body: rank, score, exp i, exp j, r):", best, "of", len(res))

    # ---- refine on the CPU (float64): orbit fit, re-match, final score
    trees = [cKDTree(np.column_stack([X[EXP == e], Y[EXP == e]])) for e in range(E)]
    eidx = [np.nonzero(EXP == e)[0] for e in range(E)]
    Rinv = R.T

    def grid_state(row):
        _, _, i, j, ai, bj, r, f = row
        i, j = int(i), int(j); da = seeds[i][int(ai)]; db = seeds[j][int(bj)]
        def hel(d, Oe, r_):
            u = np.array([UX[d], UY[d], UZ[d]]); b = Oe @ u
            rho = -b + math.sqrt(max(b * b - Oe @ Oe + r_ * r_, 0)); return Oe + rho * u, rho
        vc = math.sqrt(GM / r); dt0 = t[j] - t[i]
        Pa, rhoa = hel(da, O[i], r); Pb, rhob = hel(db, O[j], r + f * vc * dt0)
        dtau = dt0 - (rhob - rhoa) / C_AU; g = GM / r ** 3
        Va = (Pb - Pa) / dtau + 0.5 * g * Pa * dtau
        return Pa, Va, t[i] - rhoa / C_AU, da, db

    def match(P, V, t0, rad):
        xy = predict_xy(P, V, t0, t, O)
        md = np.full(E, -1)
        for e in range(E):
            c = trees[e].query_ball_point(xy[e], rad)
            if c:
                c = eidx[e][c]; md[e] = c[np.argmax(LLR[c])]
        return md

    def sample(xy):
        """S/N at predicted positions xy [M, E, 2] in every exposure: max(bilinear, 3x3-pooled - 0.8); nan = no data."""
        M = len(xy); outv = np.full((M, E), np.nan)
        for e in range(E):
            x = tt(xy[:, e, 0]); y = tt(xy[:, e, 1])
            vb, okb = bil_at(e, x, y); vp, okp = pool_at(e, x, y)
            v = torch.where(okb, torch.maximum(vb, vp - 0.8), vp - 0.8)
            v = torch.where(okb | okp, v, torch.full_like(v, float("nan")))
            outv[:, e] = v.cpu().numpy()
        return outv

    wS_n = 1.0 / sigF
    dbg_seen = set()
    kecl_n = R @ np.array([0.0, -math.sin(EPS), math.cos(EPS)])

    def bright_score(sv):
        """Brightness-consistency evidence of one track: sv [E] measured S/N (nan = no data).
        Returns (score, flux, Z of the optimal stack, number of nights with a >= 2.5 sigma night stack)."""
        ok = np.isfinite(sv)
        if ok.sum() < 3:
            return -1e9, 0.0, 0.0, 0
        s_ = sv[ok]; w_ = wS_n[ok]
        r_ = s_ / w_; o = np.argsort(r_); cw = np.cumsum(w_[o] ** 2)
        fl = r_[o][np.searchsorted(cw, 0.5 * cw[-1])]
        if fl <= 0:
            return -1e9, 0.0, 0.0, 0
        mu = fl * w_
        pos = postab[np.nonzero(ok)[0], np.clip(np.rint(s_ * 10), 0, 127).astype(int)]
        neg = np.minimum((s_ - mu) ** 2 / (2 * (1 + (0.25 * mu) ** 2)), 15.0)
        Z = float(np.sum(s_ * w_) / math.sqrt(np.sum(w_ * w_)))
        nt = night[ok]; nnights = 0
        for n in np.unique(nt):
            q = s_[nt == n]
            if q.sum() / math.sqrt(len(q)) >= 2.5:
                nnights += 1
        return float(np.sum(pos - neg)), float(fl), Z, nnights

    def cand_json(c):
        md = c["md"]; ds = md[md >= 0]
        P, V = c["P"], c["V"]
        j = {"epoch_mjd": float(c["t0"] + tref), "state": [float(v) for v in np.concatenate([Rinv @ P, Rinv @ V])],
             "p": 0.0, "score": float(c["score"]), "n_det": int(len(ds)), "n_nights": int(c["nn"]), "Z": round(c["Z"], 2),
             "snr": [None if not np.isfinite(v) else round(float(v), 1) for v in c["sv"]]}
        if c["fl"] > 0:
            j["mag"] = float(25.0 - 2.5 * math.log10(c["fl"]))
        return j

    S_MID, S_SCALE = float(os.environ.get("FINDER_SMID", 45.0)), 10.0

    def finalize(cl):
        cl = sorted(cl, key=lambda c: -c["score"])
        used_d = set(); outl = []; tracks = []
        for c in cl:
            ds = [int(d) for d in c["md"] if d >= 0]
            if sum(d in used_d for d in ds) >= max(1, len(ds) // 2):
                continue
            xy = c["xy"]
            if tracks:
                dd_ = np.abs(np.array(tracks) - xy[None]).max(axis=2)        # [n, E] offset in every exposure
                if (np.sum(dd_ < 2.0, axis=1) >= max(2, E // 2)).any():
                    continue
            used_d.update(ds); tracks.append(xy)
            outl.append(cand_json(c))
            if len(outl) >= 200:
                break
        # the top 25 get p >= 0.5; so does anything further down that stands clearly above this scenario's chance
        # alignments (their level: the median score of ranks 25-60), so a rich scenario can use slots a poor one wastes
        tail = [c["score"] for c in outl[25:60]]
        s_clear = max(S_MID, (float(np.median(tail)) + 12.0) if len(tail) >= 10 else 1e9)
        for rk, c in enumerate(outl):
            sg = 1 / (1 + math.exp(-(c["score"] - S_MID) / S_SCALE))
            c["p"] = round((0.5 + 0.49 * sg) if (rk < 25 or c["score"] >= s_clear) else 0.45 * sg * 0.999 ** rk, 5)
        return outl

    def score_batch(batch):
        if not batch:
            return []
        sv = sample(np.array([c["xy"] for c in batch]))
        outb = []
        for c, s_ in zip(batch, sv):
            b_, fl, Z, nnights = bright_score(s_)
            if c.get("inj") is not None and c["inj"] not in dbg_seen:
                dbg_seen.add(c["inj"])
                log(f"  inj {c['inj']} refined: bright {b_:.1f} pos {c['pos']:.1f} nights {nnights} fitted {c['fit']} sv {np.round(s_, 1).tolist()} off {np.round(np.hypot(*(c['xy'] - inj_all[used, c['inj'], :2]).T), 1).tolist()}")
            if nnights < 3:
                continue
            c["score"] = b_ + c["pos"]; c["fl"] = fl; c["Z"] = Z; c["nn"] = nnights; c["sv"] = s_
            outb.append(c)
        return outb

    seen = set(); todo = []
    for row in res:
        k1 = (int(row[2]), int(row[4])); k2 = (int(row[3]), int(row[5]))
        if (k1, k2) in seen:
            continue
        seen.add((k1, k2)); todo.append(row)
        if len(todo) >= 12000:
            break
    # ---- polish (r, rdot) of every kept linkage on the unpooled S/N maps: the search grid is ~1" coarse, and a
    # linkage with only two catalogue detections has nothing else to pin its orbit
    todo = np.array(todo, dtype=np.float64)
    try:
        for (i, j), gstep in pair_step.items():
            if time.time() > deadline - 0.2 * budget:
                break
            m = np.nonzero((todo[:, 2] == i) & (todo[:, 3] == j))[0]
            if not len(m):
                continue
            aq = torch.as_tensor(todo[m, 4].astype(np.int64), device=dev); bq = torch.as_tensor(todo[m, 5].astype(np.int64), device=dev)
            r0 = tt(todo[m, 6]); f0 = tt(todo[m, 7])
            fq = 0.425 * (sF[i][aq] + sF[j][bq])
            best = torch.full((len(m),), -1e9, device=dev); br = r0.clone(); bf = f0.clone()
            for dg in np.linspace(-1.0, 1.0, 9):
                for df in (0.0, -0.15, 0.15):
                    rq = 1.0 / torch.clamp(1.0 / r0 + float(dg) * gstep, min=1 / 600.0, max=1 / 25.3); fv = f0 + df
                    Pa, Va, g, ta, rhoa = setup_pair(i, j, rq, fv, aq, bq)
                    sc = torch.zeros_like(r0) - (0.5 if df else 0.0)
                    for k in range(E):
                        if k == i or k == j:
                            continue
                        x, y = predict(Pa, Va, g, ta, rhoa, k)
                        vb, okb = bil_at(k, x, y); vp, okp = pool_at(k, x, y)
                        sc = sc + evid(k, torch.where(okb, vb, vp - 0.8), okb | okp, fq * wS[k], 0.0)
                    upd = sc > best
                    best = torch.where(upd, sc, best); br = torch.where(upd, rq, br); bf = torch.where(upd, fv, bf)
            todo[m, 6] = br.cpu().numpy().astype(np.float64); todo[m, 7] = bf.cpu().numpy().astype(np.float64)
        log("polished (r, rdot) of the kept linkages")
    except Exception as ex:
        log("polish failed:", type(ex).__name__, ex)
    # preliminary output from the grid states (in case refinement runs out of time)
    prelim = []
    for row in todo[:400]:
        Pa, Va, ta, da, db = grid_state(row)
        r_ = np.linalg.norm(Pa)
        if r_ < 25.2 or Va @ Va / 2 - GM / r_ >= 0:
            continue
        md = np.full(E, -1); md[int(row[2])] = da; md[int(row[3])] = db
        prelim.append({"P": Pa, "V": Va, "t0": ta, "md": md, "pos": 0.0, "xy": predict_xy(Pa, Va, ta, t, O)})
    prelim = score_batch(prelim)
    write_out(a.out, finalize(prelim))
    log(f"preliminary output written ({len(prelim)}); refining up to {len(todo)} linkages")

    final = []; batch = []
    for n_ref, row in enumerate(todo):
        if time.time() > deadline - 6:
            log(f"refine time limit at {n_ref}"); break
        if len(batch) >= 100:
            final += score_batch(batch); batch = []
            write_out(a.out, finalize(final) or finalize(prelim))
        P, V, t0, da, db = grid_state(row)
        md = match(P, V, t0, RAD)
        md[int(row[2])] = da; md[int(row[3])] = db
        fitted = False
        for it in range(3):
            ds = md[md >= 0]
            if len(set(night[EXP[ds]])) < 3:
                break
            xy = np.column_stack([X[ds], Y[ds]])
            sig = np.sqrt(0.12 ** 2 + (0.7 * FW[ds] / SNR[ds]) ** 2)
            P2, V2, resid = fit_orbit(P, V, t0, t[EXP[ds]], O[EXP[ds]], xy, sig)
            md_new = match(P2, V2, t0, 1.0 if it else 1.3)
            if len(set(night[EXP[md_new[md_new >= 0]]])) < 3:
                break
            P, V = P2, V2
            fitted = True
            if np.array_equal(md_new, md):
                break
            md = md_new
        r_ = np.linalg.norm(P)
        if r_ < 25.2 or V @ V / 2 - GM / r_ >= 0:
            if inj_seed.get((int(row[2]), int(row[4]))) is not None:
                log(f"  inj seed row dropped: r={r_:.1f} unbound={V @ V / 2 - GM / r_ >= 0} fitted={fitted}")
            continue
        xyp = predict_xy(P, V, t0, t, O)
        pos = 0.0
        if fitted:
            ds = md[md >= 0]
            resid = np.hypot(*(xyp[EXP[ds]] - np.column_stack([X[ds], Y[ds]])).T)
            if resid.max() > 1.3:
                fitted = False
            else:
                sg2 = 0.15 ** 2 + (0.7 * FW[ds] / SNR[ds]) ** 2
                pos = float(np.sum(np.log(RAD ** 2 / (2 * sg2)) - np.minimum(resid ** 2 / (2 * sg2), 8.0)))
                pos = 0.5 * max(pos, 0.0) * max(len(ds) - 2, 0) / len(ds)
        if not fitted:
            md = np.full(E, -1); md[int(row[2])] = da; md[int(row[3])] = db
        # mild orbit prior: chance alignments fill the allowed velocity space evenly, real bodies prefer low e and i
        hv = np.cross(P, V); hn_ = np.linalg.norm(hv)
        inc = math.degrees(math.acos(max(-1.0, min(1.0, float(hv @ kecl_n) / hn_))))
        ecc = float(np.linalg.norm(np.cross(V, hv) / GM - P / r_))
        pos -= min(0.5 * (ecc / 0.3) ** 2 + 0.5 * (inc / 25.0) ** 2, 5.0)
        batch.append({"P": P, "V": V, "t0": t0, "md": md, "pos": pos, "xy": xyp, "fit": fitted,
                      "inj": inj_seed.get((int(row[2]), int(row[4]))) if inj_seed.get((int(row[2]), int(row[4]))) == inj_seed.get((int(row[3]), int(row[5]))) else None})
    final += score_batch(batch)
    outl = finalize(final)
    if outl or not prelim:
        write_out(a.out, outl)
    for c in outl[:40]:
        r_ = np.linalg.norm(c["state"][:3])
        log(f"  p={c['p']:.3f} score={c['score']:.1f} Z={c['Z']:.1f} ndet={c['n_det']} nights={c['n_nights']} r={r_:.1f} mag={c.get('mag', float('nan')):.2f} snr={c['snr']}")
    log(f"done: {len(outl)} candidates")


if __name__ == "__main__":
    main()
