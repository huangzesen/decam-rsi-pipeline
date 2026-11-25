"""TNO finder: matched-filter detection per CCD -> static-source veto -> seed-pair x (distance, radial velocity)
orbit hypotheses scored on the GPU against a deep (3-sigma) catalogue of every exposure -> orbit fit.  See NOTES.md."""
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


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------------------------- detection
def detect_chunk(task):
    """Worker: detect sources on a few CCDs of one exposure. Returns arrays (ra, dec, snr, flux, fwhm_arcsec)."""
    warnings.simplefilter("ignore")
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy import ndimage as ndi
    img_path, mask_path, hdus, thr, fwhm_fb = task
    out = []
    try:
        hi = fits.open(img_path, memmap=False)
        hm = fits.open(mask_path, memmap=False) if mask_path else None
    except Exception:
        return np.zeros((0, 5))
    for k in hdus:
        try:
            d = np.asarray(hi[k].data, dtype=np.float32)
            hdr = hi[k].header
            if d.ndim != 2:
                continue
            good = np.isfinite(d)
            if hm is not None:
                try:
                    m = hm[hdr.get("EXTNAME", k)].data
                except Exception:
                    m = hm[k].data
                good &= (m == 0)
            d = np.where(good, d, np.float32(0))
            ny, nx = d.shape
            B = 128
            gy, gx = (ny + B - 1) // B, (nx + B - 1) // B
            sub = np.full((gy * B // 4, gx * B // 4), np.nan, np.float32)
            s4 = np.where(good[::4, ::4], d[::4, ::4], np.nan)
            sub[:s4.shape[0], :s4.shape[1]] = s4
            blk = sub.reshape(gy, B // 4, gx, B // 4).transpose(0, 2, 1, 3).reshape(gy, gx, -1)
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
            # bilinear upsample of block maps
            yy = (np.arange(ny) + 0.5) / B - 0.5; xx = (np.arange(nx) + 0.5) / B - 0.5
            coords = np.meshgrid(yy, xx, indexing="ij")
            bg = ndi.map_coordinates(med, coords, order=1, mode="nearest").astype(np.float32)
            sg = ndi.map_coordinates(mad, coords, order=1, mode="nearest").astype(np.float32)
            del coords
            d -= bg
            d *= good
            fw = hdr.get("FWHM", None)
            try:
                fw = float(fw)
            except Exception:
                fw = None
            if fw is None or not (1.5 < fw < 15):
                fw = fwhm_fb / 0.263
            s = min(max(fw / 2.3548, 1.0), 5.0)
            num = ndi.gaussian_filter(d, s, truncate=3.0, mode="constant")
            den = ndi.gaussian_filter(good.astype(np.float32), s / math.sqrt(2), truncate=3.0, mode="constant")
            norm = 1.0 / (4 * math.pi * s * s)
            snr = num / (sg * np.sqrt(np.maximum(den, 0.05) * norm))
            snr[den < 0.8] = 0
            mx = ndi.maximum_filter(snr, 3)
            pk = (snr >= thr) & (snr == mx)
            pk[:8] = False; pk[-8:] = False; pk[:, :8] = False; pk[:, -8:] = False
            py, px = np.nonzero(pk)
            if len(py) == 0:
                continue
            sv = snr[py, px]
            if len(py) > 40000:
                o = np.argsort(-sv)[:40000]; py, px, sv = py[o], px[o], sv[o]
            # veto: wings of bright sources
            ds = 8
            cy, cx = ny // ds, nx // ds
            pool = snr[:cy * ds, :cx * ds].reshape(cy, ds, cx, ds).max(axis=(1, 3))
            pool = ndi.maximum_filter(pool, 5)
            nb = pool[np.minimum(py // ds, cy - 1), np.minimum(px // ds, cx - 1)]
            keep = ~((nb > 40) & (sv < 0.5 * nb))
            # veto: single-pixel spikes (cosmic rays / hot pixels)
            flux = num[py, px] / (np.maximum(den[py, px], 0.05) * norm)
            pkv = d[py, px]
            keep &= ~((pkv - 3 * sg[py, px]) > 0.6 * flux)
            py, px, sv, flux = py[keep], px[keep], sv[keep], flux[keep]
            if len(py) == 0:
                continue
            # sub-pixel centroid from the S/N map
            c0 = snr[py, px]
            def off(a, b):
                dd = a + b - 2 * c0
                return np.clip(np.where(dd < 0, 0.5 * (a - b) / np.where(dd < 0, dd, -1), 0), -0.5, 0.5)
            ox = off(snr[py, px - 1], snr[py, px + 1])
            oy = off(snr[py - 1, px], snr[py + 1, px])
            # off() sign: peak offset = 0.5*(a-b)/(a+b-2c) with a=left, b=right
            w = WCS(hdr)
            ra, dec = w.all_pix2world(px + ox, py + oy, 0)
            out.append(np.stack([ra, dec, sv, flux, np.full(len(ra), fw * 0.263)], 1))
        except Exception as ex:   # a broken CCD must not kill the run
            sys.stderr.write(f"ccd {k}: {type(ex).__name__}: {ex}\n")
    hi.close()
    if hm is not None:
        hm.close()
    return np.concatenate(out) if out else np.zeros((0, 5))


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
    dets = {}
    max_exp = 150
    cache = os.environ.get("FINDER_CACHE")
    cfile = os.path.join(cache, os.path.basename(os.path.normpath(a.scenario)) + ".pkl") if cache else None
    if cfile and os.path.exists(cfile):
        import pickle
        dets = pickle.load(open(cfile, "rb")); order = []
    with ProcessPoolExecutor(ncpu) as pool:
        t_per = None
        for n_done, i in enumerate(order[:max_exp]):
            if t_per is not None and time.time() + 1.3 * t_per > det_deadline and n_done >= 3:
                log(f"detection time limit: {n_done}/{nE_all} exposures used"); break
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
            tasks = [(ip, mp, hd[c::nchunk], THR_LOW, fb) for c in range(nchunk) if hd[c::nchunk]]
            res = list(pool.map(detect_chunk, tasks))
            dets[i] = np.concatenate(res) if res else np.zeros((0, 5))
            t_per = time.time() - t1
            log(f"exp {i} night {night_all[i]} band {r['band']}: {len(dets[i])} detections, {t_per:.1f}s")
    if cfile and not os.path.exists(cfile):
        import pickle
        pickle.dump(dets, open(cfile, "wb"))
    used = sorted(k for k in dets if len(dets[k]) > 0)
    if len(used) < 3:
        log("too few exposures"); return
    E = len(used)
    t = mjd[used] - tref
    nights_u, night = np.unique(night_all[used], return_inverse=True)
    NN = len(nights_u)
    O = np.array([[float(rows[i]["obs_x"]), float(rows[i]["obs_y"]), float(rows[i]["obs_z"])] for i in used]) @ R.T
    magzero = []
    for i in used:
        try:
            magzero.append(float(rows[i]["magzero"]))
        except Exception:
            magzero.append(float("nan"))

    # ---- catalogue in the field frame
    cat = []
    for e, i in enumerate(used):
        d = dets[i]
        rr, dd = np.radians(d[:, 0]), np.radians(d[:, 1])
        u = np.stack([np.cos(dd) * np.cos(rr), np.cos(dd) * np.sin(rr), np.sin(dd)], 1) @ R.T
        ok = u[:, 2] > 0.9
        u, d = u[ok], d[ok]
        cat.append(np.column_stack([u[:, 0] / u[:, 2] * ASEC, u[:, 1] / u[:, 2] * ASEC, d[:, 2], d[:, 3], d[:, 4], np.full(len(d), e)]))
    cat = np.concatenate(cat)
    X, Y, SNR, FLUX, FW, EXP = cat.T
    EXP = EXP.astype(int)
    N = len(cat)
    log(f"{N} detections in {E} exposures, {NN} nights")

    # ---- static-source veto
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([X, Y]))
    pr = tree.query_pairs(2.5, output_type="ndarray")
    i1, i2 = pr[:, 0], pr[:, 1]
    d2 = (X[i1] - X[i2]) ** 2 + (Y[i1] - Y[i2]) ** 2
    dn = night[EXP[i1]] != night[EXP[i2]]
    static = np.zeros(N, bool)
    rad = np.where(np.maximum(SNR[i1], SNR[i2]) > 30, 2.5, 1.0)
    m = dn & (d2 < rad ** 2)
    static[i1[m]] = True; static[i2[m]] = True
    del pr, i1, i2, d2, dn, m
    # anything within 6" of a very bright detection (any exposure) is star/galaxy structure
    bm = np.nonzero(SNR > 150)[0]
    if len(bm) > 1:
        bt = cKDTree(np.column_stack([X[bm], Y[bm]]))
        dd_, ii_ = bt.query(np.column_stack([X, Y]), k=2, distance_upper_bound=6.0)
        other = (np.isfinite(dd_[:, 0]) & (bm[np.minimum(ii_[:, 0], len(bm) - 1)] != np.arange(N))) | np.isfinite(dd_[:, 1])
        static |= other
    log(f"static veto: {static.sum()} of {N}; moving-candidate detections: {(~static).sum()}")
    keep = ~static
    X, Y, SNR, FLUX, FW, EXP = X[keep], Y[keep], SNR[keep], FLUX[keep], FW[keep], EXP[keep]
    N = len(X)
    mz = np.array(magzero); bands = [rows[i]["band"] for i in used]
    for e in range(E):
        if not np.isfinite(mz[e]) or not (20 < mz[e] < 35):
            same = [mz[k] for k in range(E) if bands[k] == bands[e] and np.isfinite(mz[k]) and 20 < mz[k] < 35]
            allz = [mz[k] for k in range(E) if np.isfinite(mz[k]) and 20 < mz[k] < 35]
            mz[e] = np.median(same) if same else (np.median(allz) if allz else 29.5)
    COL = {"g": -0.6, "r": 0.0, "VR": 0.0, "i": 0.25, "z": 0.4, "Y": 0.45}
    MAG = np.array([mz[e] + COL.get(bands[e], 0.0) for e in range(E)])[EXP] - 2.5 * np.log10(np.maximum(FLUX, 1e-3))
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
    import torch
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log("device", dev)
    f32 = torch.float32
    tt = lambda v, dt=f32: torch.as_tensor(np.asarray(v), dtype=dt, device=dev)
    half = float(max(np.abs(X).max(), np.abs(Y).max())) + 10 * CELL
    G = int(2 * half / CELL) + 2
    grid = torch.full((E, G * G), -1, dtype=torch.int32, device=dev)
    cxn = np.floor((X + half) / CELL).astype(np.int64); cyn = np.floor((Y + half) / CELL).astype(np.int64)
    o = np.argsort(SNR)                               # brightest written last (wins a shared cell)
    lin = (cyn * G + cxn)
    for e in range(E):
        m = o[EXP[o] == e]
        grid[e, torch.as_tensor(lin[m], device=dev)] = torch.as_tensor(m.astype(np.int32), device=dev)
    dX = tt(np.append(X, 1e9)); dY = tt(np.append(Y, 1e9)); dL = tt(np.append(LLR, 0.0)); dM = tt(np.append(MAG, 0.0)); dTol = tt(np.append(MTOL, 0.0))
    # unit vectors of detections
    nrm = np.sqrt(1 + (X / ASEC) ** 2 + (Y / ASEC) ** 2)
    UX, UY, UZ = X / ASEC / nrm, Y / ASEC / nrm, 1 / nrm
    Ot = tt(O); tT = tt(t)
    kecl = tt(R @ np.array([0.0, -math.sin(EPS), math.cos(EPS)]))
    night_t = torch.as_tensor(night, device=dev)

    # seeds
    seed_thr = 4.5
    seeds = []
    for e in range(E):
        m = np.nonzero((EXP == e) & (SNR >= seed_thr) & (SNR < 400))[0]
        if len(m) > 2000:                        # keep the faintest-allowed threshold that gives <= 2000 seeds
            m = m[np.argsort(SNR[m])[-2000:]] if False else m[np.argsort(-SNR[m])]
            # drop from the faint end, but never prefer the brightest junk: take a random subset of S/N >= 5.5 first
            hi = m[SNR[m] >= 5.5]; lo = m[SNR[m] < 5.5]
            rng = np.random.default_rng(1)
            if len(hi) > 2000:
                hi = rng.choice(hi, 2000, replace=False); lo = lo[:0]
            m = np.concatenate([hi, lo[:2000 - len(hi)]])
        seeds.append(m)
    log("seeds per exposure:", [len(s) for s in seeds])

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

    def lookup(k, x, y, mag):
        """Best chance-LLR of a support detection of exposure k within RAD of (x, y); and its index."""
        fx = (x + half) / CELL - 0.5; fy = (y + half) / CELL - 0.5
        cx0 = torch.floor(fx).long(); cy0 = torch.floor(fy).long()
        best = torch.zeros_like(x); bidx = torch.full(x.shape, -1, dtype=torch.long, device=dev)
        for ddy in (0, 1):
            for ddx in (0, 1):
                cx = cx0 + ddx; cy = cy0 + ddy
                inb = (cx >= 0) & (cx < G) & (cy >= 0) & (cy < G)
                idx = grid[k, (cy.clamp(0, G - 1) * G + cx.clamp(0, G - 1))].long()
                idx = torch.where(inb, idx, torch.full_like(idx, -1))
                d2 = (dX[idx] - x) ** 2 + (dY[idx] - y) ** 2
                val = torch.where((idx >= 0) & (d2 < RAD * RAD) & ((dM[idx] - mag).abs() < dTol[idx] + 0.3), dL[idx], torch.zeros_like(x))
                upd = val > best
                best = torch.where(upd, val, best); bidx = torch.where(upd, idx, bidx)
        return best, bidx

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
    sX = [tt(X[s]) for s in seeds]; sY = [tt(Y[s]) for s in seeds]

    # exposure pairs on different nights, shortest first; use at most 2 exposures per night as seed exposures
    seed_exp = []
    for n in range(NN):
        ee = np.nonzero(night == n)[0]
        ee = ee[np.argsort([-len(seeds[e]) for e in ee])]   # (most seeds first is arbitrary but deterministic)
        seed_exp += list(ee[:2])
    pairs = [(i, j) for i in seed_exp for j in seed_exp if night[i] < night[j] and len(seeds[i]) and len(seeds[j])]
    pairs.sort(key=lambda p: t[p[1]] - t[p[0]])
    link_deadline = min(deadline - 0.12 * budget, T0 + 0.62 * budget)
    GAM_MAX, GAM_MIN = 1 / 25.5, 1 / 400.0
    TOL = 1.2
    results = []      # arrays [score, nn, i, j, a, b, r, rdotfrac]
    n_tr_total = 0
    for ip, (i, j) in enumerate(pairs):
        if time.time() > link_deadline:
            log(f"link time limit after {ip}/{len(pairs)} exposure pairs"); break
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
        rgrid = 1.0 / gam
        fgrid = torch.linspace(-0.45, 0.45, nf, device=dev) if nf > 1 else torch.zeros(1, device=dev)
        Ni, Nj = len(seeds[i]), len(seeds[j])
        n_tr = 0
        sep2 = (sX[i][:, None] - sX[j][None, :]) ** 2 + (sY[i][:, None] - sY[j][None, :]) ** 2
        cm = (sep2 < (200.0 * dt0 + 10) ** 2) & ((smag[i][:, None] - smag[j][None, :]).abs() < stol[i][:, None] + stol[j][None, :])
        cand = torch.nonzero(cm)
        del sep2, cm
        Nc = len(cand)
        if Nc == 0:
            continue
        ca, cb = cand[:, 0], cand[:, 1]
        rchunk = max(1, int(4e6 // Nc))
        best_pair = []
        for c0 in range(0, ng, rchunk):
            if time.time() > link_deadline:
                break
            rr = rgrid[c0:c0 + rchunk]
            for fi in range(nf):
                r3 = rr[:, None]
                Pa, Va, g, ta, rhoa = setup_pair(i, j, r3, fgrid[fi], ca[None, :], cb[None, :])
                v2 = Va[0] ** 2 + Va[1] ** 2 + Va[2] ** 2
                vc2 = GM / r3
                hx = Pa[1] * Va[2] - Pa[2] * Va[1]; hy = Pa[2] * Va[0] - Pa[0] * Va[2]; hz = Pa[0] * Va[1] - Pa[1] * Va[0]
                hn = torch.sqrt(hx * hx + hy * hy + hz * hz)
                cosi = (hx * kecl[0] + hy * kecl[1] + hz * kecl[2]) / hn
                ok = (v2 < 1.9 * vc2) & (v2 > 0.25 * vc2) & (cosi > 0.5)
                sel = torch.nonzero(ok)
                del Pa, Va, v2, hx, hy, hz, hn, cosi, ok
                if len(sel) == 0:
                    continue
                sel = torch.stack([sel[:, 0], ca[sel[:, 1]], cb[sel[:, 1]]], 1)
                for s0 in range(0, len(sel), 3000000):
                    q = sel[s0:s0 + 3000000]
                    rq = rr[q[:, 0]]; aq = q[:, 1]; bq = q[:, 2]
                    Pa, Va, g, ta, rhoa = setup_pair(i, j, rq, fgrid[fi], aq, bq)
                    score = sl[i][aq] + sl[j][bq]
                    magq = 0.5 * (smag[i][aq] + smag[j][bq])
                    nn = torch.full_like(score, 2.0)
                    for n in range(NN):
                        if n == night[i] or n == night[j]:
                            # other exposures of the seed nights add significance but no new night
                            ks = [k for k in range(E) if night[k] == n and k != i and k != j]
                            for k in ks:
                                x, y = predict(Pa, Va, g, ta, rhoa, k)
                                v, _ = lookup(k, x, y, magq)
                                score = score + v
                            continue
                        hit = torch.zeros_like(score)
                        for k in np.nonzero(night == n)[0]:
                            x, y = predict(Pa, Va, g, ta, rhoa, int(k))
                            v, _ = lookup(int(k), x, y, magq)
                            score = score + v
                            hit = torch.maximum(hit, (v > 0).to(f32))
                        nn = nn + hit
                    m = nn >= 3
                    n_tr += len(q)
                    if m.any():
                        mi = torch.nonzero(m)[:, 0]
                        sc = score[mi]
                        if len(mi) > 20000:
                            top = torch.topk(sc, 20000).indices; mi = mi[top]; sc = sc[top]
                        arr = torch.stack([sc, nn[mi], aq[mi].to(f32), bq[mi].to(f32), rq[mi],
                                           torch.full_like(sc, float(fgrid[fi]))], 1).cpu().numpy()
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
        log(f"pair {ip} ({i},{j}) dt={dt0:.2f}d ng={ng} nf={nf} seeds {Ni}x{Nj} cand={Nc} triples={n_tr:.3g} kept={len(results[-1]) if best_pair else 0}")
    log(f"total hypotheses scored: {n_tr_total:.3g}")
    if not results:
        log("no linkages"); return
    res = np.concatenate(results)
    res = res[np.argsort(-res[:, 0])]

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

    def cand_json(P, V, t0, score, md):
        ds = md[md >= 0]
        mags = [MAG[d] for d in ds]
        c = {"epoch_mjd": float(t0 + tref), "state": [float(v) for v in np.concatenate([Rinv @ P, Rinv @ V])],
             "p": 0.0, "score": float(score), "n_det": int(len(ds)), "n_nights": int(len(set(night[EXP[ds]]))),
             "snr": [round(float(SNR[d]), 1) for d in ds]}
        if mags:
            c["mag"] = float(np.median(mags))
        return c

    def finalize(cl):
        cl.sort(key=lambda c: -c[0])
        used_d = set(); outl = []
        for sc, P, V, t0, md in cl:
            ds = [int(d) for d in md if d >= 0]
            if sum(d in used_d for d in ds) >= max(1, len(ds) // 2):
                continue
            used_d.update(ds)
            outl.append(cand_json(P, V, t0, sc, md))
            if len(outl) >= 200:
                break
        for rk, c in enumerate(outl):
            sg = 1 / (1 + math.exp(-(c["score"] - 55.0) / 8.0))
            c["p"] = round((0.5 + 0.49 * sg) if rk < 25 else 0.45 * sg * 0.999 ** rk, 5)
        return outl

    # preliminary output from the grid states (in case refinement runs out of time)
    seen = set(); todo = []
    for row in res:
        k1 = (int(row[2]), int(row[4])); k2 = (int(row[3]), int(row[5]))
        if (k1, k2) in seen:
            continue
        seen.add((k1, k2)); todo.append(row)
        if len(todo) >= 8000:
            break
    prelim = []
    for row in todo[:300]:
        Pa, Va, ta, da, db = grid_state(row)
        r_ = np.linalg.norm(Pa)
        if r_ < 25 or Va @ Va / 2 - GM / r_ >= 0:
            continue
        md = np.full(E, -1); md[int(row[2])] = da; md[int(row[3])] = db
        prelim.append((row[0], Pa, Va, ta, md))
    write_out(a.out, finalize(prelim))
    log(f"preliminary output written; refining up to {len(todo)} linkages")

    final = []
    for n_ref, row in enumerate(todo):
        if time.time() > deadline - 5:
            log(f"refine time limit at {n_ref}"); break
        P, V, t0, da, db = grid_state(row)
        md = match(P, V, t0, RAD)
        md[int(row[2])] = da; md[int(row[3])] = db
        okc = False
        for it in range(3):
            ds = md[md >= 0]
            if len(set(night[EXP[ds]])) < 3:
                break
            xy = np.column_stack([X[ds], Y[ds]])
            sig = np.sqrt(0.12 ** 2 + (0.7 * FW[ds] / SNR[ds]) ** 2)
            P, V, resid = fit_orbit(P, V, t0, t[EXP[ds]], O[EXP[ds]], xy, sig)
            md_new = match(P, V, t0, 1.0 if it else 1.3)
            okc = True
            if np.array_equal(md_new, md):
                break
            md = md_new
        ds = md[md >= 0]
        if not okc or len(set(night[EXP[ds]])) < 3:
            continue
        xy = np.column_stack([X[ds], Y[ds]])
        pr = predict_xy(P, V, t0, t[EXP[ds]], O[EXP[ds]])
        resid = np.hypot(*(pr - xy).T)
        r_ = np.linalg.norm(P)
        if r_ < 25.2 or V @ V / 2 - GM / r_ >= 0 or resid.max() > 1.0:
            continue
        sig = np.sqrt(0.12 ** 2 + (0.7 * FW[ds] / SNR[ds]) ** 2)
        chi = float(np.sum((resid / sig) ** 2)) / max(2 * len(ds) - 4, 1)
        dm = np.abs(MAG[ds] - np.median(MAG[ds]))
        mpen = float(np.sum(np.maximum(0.0, dm - 0.6 - 2.7 / SNR[ds]) ** 2))
        sg2 = 0.15 ** 2 + (0.7 * FW[ds] / SNR[ds]) ** 2
        pos = float(np.sum(np.log(RAD ** 2 / (2 * sg2)) - np.minimum(resid ** 2 / (2 * sg2), 8.0)))
        score = float(LLR[ds].sum()) + pos - 6.0 * mpen
        final.append((score, P, V, t0, md))
        if n_ref % 100 == 99:
            write_out(a.out, finalize(list(final)))
    outl = finalize(final)
    if outl or not prelim:
        write_out(a.out, outl)
    for c in outl[:30]:
        r_ = np.linalg.norm(c["state"][:3])
        log(f"  p={c['p']:.3f} score={c['score']:.1f} ndet={c['n_det']} nights={c['n_nights']} r={r_:.1f} mag={c.get('mag', float('nan')):.2f} snr={c['snr']}")
    log(f"done: {len(outl)} candidates")


if __name__ == "__main__":
    main()
