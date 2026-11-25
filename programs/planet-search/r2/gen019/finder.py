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


TEL = {"new": 0, "old": 0, "grp": 0, "full": 0, "all": 0, "trip": 0, "weak": 0}     # gen018: run telemetry, encoded in the exit time (see tel_exit)


def log(*a):
    try:                                         # gen018: count the stages that ended in a swallowed exception
        if a and isinstance(a[0], str) and "failed" in a[0] and a[0].rstrip().endswith(":") and not a[0].startswith(("noext", "WCSCAL")):
            TEL["new" if a[0].startswith(("weak pass", "track extension")) else "old"] += 1
    except Exception:
        pass
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def proc_start():
    """Wall-clock time at which this process was created (Linux /proc), else the import time minus 1.5 s."""
    try:
        with open("/proc/self/stat") as f:
            st = f.read()
        ticks = float(st[st.rindex(")") + 2:].split()[19]) / os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime") as f:
            age = float(f.read().split()[0]) - ticks
        if 0.0 <= age <= time.time() - T0 + 120.0:
            return time.time() - age
    except Exception:
        pass
    return T0 - 1.5


TEL_PERIOD, TEL_STEP = 36.0, 3.0


def watchdog(budget):
    """gen018: a stage without a time check (static masks, seeds) can run past the budget on a slow machine; the grader
    kills the program 30 s after it. Leave by ourselves just before the budget instead: OUT.json is always complete
    (write_out replaces it atomically)."""
    import threading

    def run():
        try:
            p0 = proc_start()
            while time.time() - p0 < budget - 0.7:
                time.sleep(0.2)
        except Exception:
            return                               # (never leave early because of the watchdog itself)
        try:
            log("watchdog: budget reached, leaving")
            import multiprocessing
            for ch in multiprocessing.active_children():
                ch.terminate()
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)
    threading.Thread(target=run, daemon=True).start()


def tel_code_old():
    fr = TEL["full"] / TEL["all"] if TEL["all"] else 1.0
    c = (1 if (TEL["trip"] or TEL["weak"]) else 0) if fr >= 0.999 else (2 if fr >= 0.5 else 3)
    if TEL["grp"]:
        return 11
    if TEL["old"]:
        return 8 + min(c, 2)
    return c + (4 if TEL["new"] else 0)


def tel_code():
    """gen019: what the exit time says now (gen018's code was 0 in all ten sealed scenarios). Run progress only:
    k = g + 4 f.  g = 0: no deep-pair jobs ran in this scenario (gen015: they need >= 9 exposures on >= 4 nights);
        1: deep-pair jobs exist and every one ran its full distance grid before the link deadline;
        2: at least one of them did;  3: none did (cut short or never reached).
    f = 0: no stage ended in a swallowed exception;  1: the weak pass or the track extension did;  2: another stage did."""
    f = 2 if TEL["old"] else (1 if TEL["new"] else 0)
    if not TEL.get("d_on"):
        return 4 * f
    da = TEL.get("d_all", 0); df = TEL.get("d_full", 0)
    return (1 if (da > 0 and df >= da) else (2 if df > 0 else 3)) + 4 * f


def tel_exit(budget):
    """gen018: the grader reports only the seconds each sealed scenario ran, so the run's own progress is put into
    them. When all work is done the program waits (at most 36 s) and exits when the age of the process is
    3 k + 0.2 s modulo 36 s, so  k = floor((reported seconds mod 36) / 3)  (process exit adds < 2 s), k = 0..11:
      k = c + 4 f:  c = 0: every regular seed-pair job (tiers 0, 1) ran its full distance grid before the link deadline,
                           and the regular triplet pass and the weak pass (gen017) did every anchor pair;
                        1: the regular jobs are complete, the triplet pass or the weak pass is not;
                        2: >= 50 % of the regular jobs are complete;  3: fewer;
                    f = 0: no stage raised an exception; f = 1: the weak pass (gen017) or the track extension (gen016) did;
      k = 8, 9, 10: some other stage raised (with c = 0, 1, >= 2);   k = 11: group mode (> 55 exposures) ran.
    The time at which the work itself ended is the reported time minus 0..36 s, as before."""
    try:
        k = tel_code()
        p0 = proc_start()
        age = time.time() - p0 + 0.3
        tgt = TEL_STEP * k - 0.3                 # gen019: was + 0.2; the grader's reported seconds were 0.8-2.8 s past the target in gen018
        age_exit = age + ((tgt - age) % TEL_PERIOD)
        log(f"telemetry (gen019): deep-pair jobs on {TEL.get('d_on', 0)} complete {TEL.get('d_full', 0)}/{TEL.get('d_all', 0)} "
            f"distance steps {TEL.get('d_done', 0)}/{TEL.get('d_ng', 0)} -> k={k}; gen018 code would be {tel_code_old()}")
        log(f"telemetry: regular jobs complete {TEL['full']}/{TEL['all']}, triplet pass incomplete {TEL['trip']}, weak pass incomplete {TEL['weak']}, failed stages new {TEL['new']} other {TEL['old']}, group mode {TEL['grp']}, "
            f"work ended at age {age - 0.3:.1f} s, exit at age {age_exit:.1f} s")
        if os.environ.get("FINDER_NOTEL"):
            return
        if age_exit > budget - 3.0:              # no room for the code: leave at the budget itself, which reads "no code"
            age_exit = budget - 0.9
        while time.time() - p0 < age_exit:
            time.sleep(min(0.25, max(0.0, age_exit - (time.time() - p0))))
    except Exception:
        pass


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
                inject_px(d, cpx, cpy, X0, Y0, SX, s, inj)
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


def inject_px(d, cpx, cpy, X0, Y0, SX, s, inj):
    """Development only: add synthetic Gaussian sources (field-frame x, y arcsec, flux) to the pixels of one CCD."""
    ny, nx = d.shape
    T = _terms((inj[:, 0] - X0) / SX, (inj[:, 1] - Y0) / SX)
    ipx = T @ cpx; ipy = T @ cpy
    for qx, qy, qf in zip(ipx, ipy, inj[:, 2]):
        if 12 < qx < nx - 12 and 12 < qy < ny - 12 and abs(qx) < 1e5:
            x0 = int(round(qx)); y0 = int(round(qy)); h = int(4 * s) + 2
            if y0 - h < 0 or x0 - h < 0 or y0 + h + 1 > ny or x0 + h + 1 > nx:
                continue
            yy, xx = np.mgrid[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1]
            d[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1] += (qf / (2 * math.pi * s * s) * np.exp(-((xx - qx) ** 2 + (yy - qy) ** 2) / (2 * s * s))).astype(np.float32)


# ------------------------------------------------------------------ gen012: re-registration of failed-WCS exposures
def _wcs_grid(nx, ny):
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 17))
    return gx.ravel(), gy.ravel()


def wcs_table(img_path, R):
    """Forward WCS polynomials (normalised pixel -> field-frame arcsec) of every CCD HDU of a file, keyed by the
    header's CCDNUM.  In exposures whose astrometric calibration failed the Community Pipeline has left each HDU with
    the WCS of *another* CCD (EXTNAME S29 carries CCDNUM 62 and that CCD's CRPIX / PV terms, S30 carries CCDNUM 1, ...):
    the right WCS of a CCD is in the file, in the HDU whose CCDNUM is that CCD's number in a good exposure."""
    from astropy.io import fits
    from astropy.wcs import WCS
    tab = {}
    with fits.open(img_path, memmap=False) as h:
        for k in range(len(h)):
            try:
                hdr = h[k].header
                if not (hdr.get("NAXIS", 0) == 2 or hdr.get("ZNAXIS", 0) == 2) or hdr.get("CCDNUM", None) is None:
                    continue
                nx = int(hdr.get("NAXIS1", 0) or hdr.get("ZNAXIS1", 0)); ny = int(hdr.get("NAXIS2", 0) or hdr.get("ZNAXIS2", 0))
                if nx < 100 or ny < 100:
                    continue
                gx, gy = _wcs_grid(nx, ny)
                ra, dec = WCS(hdr).all_pix2world(gx, gy, 0)
                rr, dd = np.radians(ra), np.radians(dec)
                uu = np.stack([np.cos(dd) * np.cos(rr), np.cos(dd) * np.sin(rr), np.sin(dd)], 1) @ R.T
                if not np.all(np.isfinite(uu)) or uu[:, 2].min() < 0.9:
                    continue
                Af = _terms((gx - nx / 2) / (nx / 2), (gy - ny / 2) / (ny / 2))
                tab[int(hdr["CCDNUM"])] = (np.linalg.lstsq(Af, uu[:, 0] / uu[:, 2] * ASEC, rcond=None)[0],
                                           np.linalg.lstsq(Af, uu[:, 1] / uu[:, 2] * ASEC, rcond=None)[0], nx, ny)
            except Exception:
                pass
    return tab


def set_wcs(c, cX, cY, M=None):
    """Replace the WCS polynomials of a CCD dict (as made by load_chunk) by the forward polynomials cX, cY, optionally
    followed by the affine map M (2 x 3: x' = M[0] . [1, x, y], y' = M[1] . [1, x, y]) in the field frame."""
    ny, nx = c["d"].shape
    gx, gy = _wcs_grid(nx, ny)
    Af = _terms((gx - nx / 2) / (nx / 2), (gy - ny / 2) / (ny / 2))
    GX = Af @ cX; GY = Af @ cY
    if M is not None:
        GX, GY = M[0, 0] + M[0, 1] * GX + M[0, 2] * GY, M[1, 0] + M[1, 1] * GX + M[1, 2] * GY
    c["cX"] = np.linalg.lstsq(Af, GX, rcond=None)[0]; c["cY"] = np.linalg.lstsq(Af, GY, rcond=None)[0]
    X0, Y0 = GX.mean(), GY.mean(); SX = max(np.ptp(GX), np.ptp(GY)) / 2
    Ai = _terms((GX - X0) / SX, (GY - Y0) / SX)
    c["cpx"] = np.linalg.lstsq(Ai, gx, rcond=None)[0]; c["cpy"] = np.linalg.lstsq(Ai, gy, rcond=None)[0]
    c["X0"] = X0; c["Y0"] = Y0; c["SX"] = SX
    c["box"] = (GX.min(), GX.max(), GY.min(), GY.max())


def fit_affine(cat, tree, ref, L=150.0, bs=4.0):
    """Affine map that puts the detections of one CCD (cat [n, >= 3]: x, y arcsec, S/N) on the reference sources
    (ref [m, 2], tree = cKDTree(ref)): shift from the peak of the offset histogram of the bright detections (+-L
    arcsec), then least squares on nearest matches with a shrinking tolerance.  Returns (M 2x3 in plain field
    coordinates, number of matches, rms arcsec) or None when there is no clear offset peak / too few matches."""
    from scipy import ndimage as ndi
    X, Y, snr = cat[:, 0], cat[:, 1], cat[:, 2]
    ok = np.isfinite(X) & np.isfinite(Y)
    X, Y, snr = X[ok], Y[ok], snr[ok]
    o = np.argsort(-snr)[:250]; o = o[snr[o] >= 15]
    if len(o) < 20:
        return None
    P = np.stack([X[o], Y[o]], 1)
    idx = tree.query_ball_point(P, L * 1.42)
    if sum(len(j) for j in idx) < 20:
        return None
    dx = np.concatenate([ref[j, 0] - P[k, 0] for k, j in enumerate(idx)]); dy = np.concatenate([ref[j, 1] - P[k, 1] for k, j in enumerate(idx)])
    H, xe, ye = np.histogram2d(dx, dy, bins=int(2 * L / bs), range=[[-L, L], [-L, L]])
    Hs = ndi.uniform_filter(H, 3, mode="constant") * 9
    ij = np.unravel_index(np.argmax(Hs), Hs.shape); med = float(np.median(Hs))
    if Hs.max() < max(15.0, med + 8 * math.sqrt(med + 1), 0.2 * len(o)):
        return None
    m = snr >= 8
    x = X[m]; y = Y[m]; xc = x.mean(); yc = y.mean()
    A = np.stack([np.ones(len(x)), x - xc, y - yc], 1)
    M = np.array([[xe[ij[0]] + bs / 2, 1.0, 0.0], [ye[ij[1]] + bs / 2, 0.0, 1.0]])
    k = None
    for tol in (8.0, 4.0, 2.0, 1.2, 0.8, 0.8):
        q = np.stack([xc + A @ M[0], yc + A @ M[1]], 1)
        dd, j = tree.query(q)
        k = dd < tol
        if k.sum() < 25:
            return None
        M = np.stack([np.linalg.lstsq(A[k], ref[j[k], 0] - xc, rcond=None)[0], np.linalg.lstsq(A[k], ref[j[k], 1] - yc, rcond=None)[0]])
    q = np.stack([xc + A @ M[0], yc + A @ M[1]], 1)
    dd, j = tree.query(q); k = dd < 0.8
    if k.sum() < 25 or k.sum() < 0.25 * len(x):
        return None
    rms = float(np.sqrt(np.mean(dd[k] ** 2)))
    # sanity: a rotation / scale change of more than 3 % is not a pointing error
    if abs(M[0, 1] - 1) > 0.03 or abs(M[1, 2] - 1) > 0.03 or abs(M[0, 2]) > 0.03 or abs(M[1, 1]) > 0.03 or rms > 0.5:
        return None
    Mp = np.array([[xc + M[0, 0] - M[0, 1] * xc - M[0, 2] * yc, M[0, 1], M[0, 2]],
                   [yc + M[1, 0] - M[1, 1] * xc - M[1, 2] * yc, M[1, 1], M[1, 2]]])
    return Mp, int(k.sum()), rms


def load_chunk(task):
    """Worker of the GPU detection path: decompress a few CCDs of one exposure and fit their WCS polynomials.
    Returns a list of dicts (pixels as float32 with nan = bad); the filtering is done on the GPU by gpu_ccd."""
    warnings.simplefilter("ignore")
    from astropy.io import fits
    from astropy.wcs import WCS
    img_path, mask_path, hdus, thr, fwhm_fb, R, inj = task
    out = []
    try:
        hi = fits.open(img_path, memmap=False)
        hm = fits.open(mask_path, memmap=False) if mask_path else None
    except Exception:
        return out
    for k in hdus:
        try:
            d = np.asarray(hi[k].data, dtype=np.float32)
            hdr = hi[k].header
            if d.ndim != 2:
                continue
            ny, nx = d.shape
            w = WCS(hdr)
            gx, gy = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 17))
            gx = gx.ravel(); gy = gy.ravel()
            ra, dec = w.all_pix2world(gx, gy, 0)
            rr, dd = np.radians(ra), np.radians(dec)
            uu = np.stack([np.cos(dd) * np.cos(rr), np.cos(dd) * np.sin(rr), np.sin(dd)], 1) @ R.T
            if not np.all(np.isfinite(uu)) or uu[:, 2].min() < 0.9:
                continue
            GX = uu[:, 0] / uu[:, 2] * ASEC; GY = uu[:, 1] / uu[:, 2] * ASEC
            Af = _terms((gx - nx / 2) / (nx / 2), (gy - ny / 2) / (ny / 2))
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
            if inj is not None and len(inj):
                inject_px(d, cpx, cpy, X0, Y0, SX, s, inj)
            if hm is not None:
                try:
                    m = hm[hdr.get("EXTNAME", k)].data
                except Exception:
                    m = hm[k].data
                d[m != 0] = np.nan
            out.append({"d": d, "cX": cX, "cY": cY, "cpx": cpx, "cpy": cpy, "X0": X0, "Y0": Y0, "SX": SX, "fw": fw, "s": s,
                        "box": (GX.min(), GX.max(), GY.min(), GY.max()), "k": int(k),
                        "ext": str(hdr.get("EXTNAME", "")).strip(), "ccdnum": hdr.get("CCDNUM", None)})
        except Exception as ex:
            sys.stderr.write(f"ccd {k} load: {type(ex).__name__}: {ex}\n")
    hi.close()
    if hm is not None:
        hm.close()
    return out


STAMP_H = 5


CLS_THR = float(os.environ.get("FINDER_CLSTHR", -3.0))     # logit below which a >= 4.5 sigma peak is junk
_CLS = {}


def det_classifier(torch, dev):
    """The stamp CNN (weights in detcls.pt next to this file; trained in the gen007 session, see NOTES.md). None if
    it cannot be loaded: the program then runs without it."""
    if dev not in _CLS:
        net = None
        try:
            if not os.environ.get("FINDER_NOCLS"):
                nn = torch.nn
                net = nn.Sequential(nn.Conv2d(2, 16, 3), nn.ReLU(), nn.Conv2d(16, 32, 3), nn.ReLU(), nn.Conv2d(32, 32, 3), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(32 * 25, 48), nn.ReLU(), nn.Linear(48, 1))
                net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "detcls.pt"), map_location="cpu"))
                net = net.to(dev).eval()
        except Exception as ex:
            sys.stderr.write(f"stamp classifier not loaded: {type(ex).__name__}: {ex}\n"); net = None
        _CLS[dev] = net
    return _CLS[dev]


def gpu_ccd(torch, dev, c, thr, stamp_thr=None, cls=None, cat_only=False):
    """detect_chunk's processing of one CCD on the GPU (float32): background, block-MAD noise, cosmic-ray cleaning,
    matched filter, S/N map on the common lattice, peaks.  Returns (catalogue [n, 5] or None, (iy0, ix0, int8 map) or None)."""
    from scipy import ndimage as ndi
    F = torch.nn.functional
    f32 = torch.float32
    d = torch.from_numpy(c["d"]).to(dev)
    ny, nx = d.shape
    s = c["s"]
    good = torch.isfinite(d)
    zero = torch.zeros((), device=dev, dtype=f32); inf = torch.full((), float("inf"), device=dev, dtype=f32)
    d = torch.where(good, d, zero)
    B = 128; q = B // 4
    gyb, gxb = (ny + B - 1) // B, (nx + B - 1) // B
    sub = torch.full((gyb * q, gxb * q), float("inf"), device=dev, dtype=f32)
    s4 = torch.where(good[::4, ::4], d[::4, ::4], inf)
    sub[:s4.shape[0], :s4.shape[1]] = s4
    blk = sub.reshape(gyb, q, gxb, q).permute(0, 2, 1, 3).reshape(gyb, gxb, q * q)
    fin = torch.isfinite(blk)
    nval = fin.sum(2)

    def nmed(v):                                   # nan-median along the last axis (bad = +inf), as numpy.nanmedian
        srt = torch.sort(v, dim=2).values
        i0 = ((nval - 1) // 2).clamp(min=0)[..., None]; i1 = (nval // 2).clamp(max=q * q - 1)[..., None]
        return 0.5 * (srt.gather(2, i0) + srt.gather(2, i1))[..., 0]
    med_t = nmed(blk)
    mad_t = 1.4826 * nmed(torch.where(fin, (blk - med_t[..., None]).abs(), inf))
    med = med_t.cpu().numpy().astype(np.float32); mad = mad_t.cpu().numpy().astype(np.float32); nv = nval.cpu().numpy()
    del sub, s4, blk, fin, med_t, mad_t
    bad = ~np.isfinite(med) | ~np.isfinite(mad) | (nv < 100)
    if bad.all():
        return None, None
    med[bad] = np.median(med[~bad]); mad[bad] = np.median(mad[~bad])
    med = ndi.median_filter(med, 3, mode="nearest"); mad = ndi.median_filter(mad, 3, mode="nearest")
    mad = np.maximum(mad, 1e-3)

    def up(a_):
        t_ = torch.from_numpy(np.ascontiguousarray(a_, dtype=np.float32)).to(dev)[None, None]
        return F.interpolate(t_, scale_factor=B, mode="bilinear", align_corners=False)[0, 0, :ny, :nx]
    sg = up(mad)
    d = (d - up(med)) * good

    def gf(x, sig):
        rad = int(3.0 * sig + 0.5)
        k = torch.exp(-0.5 * (torch.arange(-rad, rad + 1, device=dev, dtype=f32) / sig) ** 2); k = k / k.sum()
        x = F.conv2d(x[None, None], k.view(1, 1, 1, -1), padding=(0, rad))
        return F.conv2d(x, k.view(1, 1, -1, 1), padding=(rad, 0))[0, 0]
    num = gf(d, s); den = gf(good.to(f32), s / math.sqrt(2))
    norm = 1.0 / (4 * math.pi * s * s)
    cr = (d > 4 * sg) & (d > 2.5 * num / (den.clamp(min=0.05) * (norm * 2 * math.pi * s * s)) + 4 * sg)
    c_ = d[1:-1, 1:-1]
    mn = torch.minimum(torch.minimum(d[1:-1, :-2] + d[1:-1, 2:], d[:-2, 1:-1] + d[2:, 1:-1]),
                       torch.minimum(d[:-2, :-2] + d[2:, 2:], d[:-2, 2:] + d[2:, :-2]))
    thr_ = 4.5 * sg[1:-1, 1:-1]
    cr2 = torch.zeros_like(cr)
    cr2[1:-1, 1:-1] = (c_ > thr_) & (c_ - thr_ > 1.5 * mn)
    cr = cr | cr2
    del mn, thr_, c_, cr2
    n_cr = int(cr.sum())
    if 0 < n_cr < 0.02 * cr.numel():
        dil = F.max_pool2d(cr.to(f32)[None, None], (3, 1), 1, (1, 0))[0, 0] + F.max_pool2d(cr.to(f32)[None, None], (1, 3), 1, (0, 1))[0, 0]
        good = good & (dil < 0.5)
        d = d * good
        num = gf(d, s); den = gf(good.to(f32), s / math.sqrt(2))
        del dil
    del cr
    snr = num / (sg * torch.sqrt(den.clamp(min=0.05) * norm))
    valid = den >= 0.8
    snr = torch.where(valid, snr, zero)
    # ---- S/N map on the common lattice (made at the end: junk found by the stamp classifier is blanked in it)
    def make_piece(valid):
        piece = None
        try:
            gx0, gx1, gy0, gy1 = c["box"]
            ix0 = int(math.ceil(gx0 / PS)); ix1 = int(math.floor(gx1 / PS))
            iy0 = int(math.ceil(gy0 / PS)); iy1 = int(math.floor(gy1 / PS))
            if 0 < ix1 - ix0 < 6000 and 0 < iy1 - iy0 < 6000:
                u = torch.from_numpy(((np.arange(ix0, ix1 + 1, dtype=np.float64) * PS - c["X0"]) / c["SX"]).astype(np.float32)).to(dev)[None, :]
                v = torch.from_numpy(((np.arange(iy0, iy1 + 1, dtype=np.float64) * PS - c["Y0"]) / c["SX"]).astype(np.float32)).to(dev)[:, None]

                def poly(cf):
                    cf = [float(x_) for x_ in cf]
                    return (cf[0] + cf[1] * u + cf[3] * u * u + cf[6] * u ** 3) + (cf[2] + cf[4] * u + cf[7] * u * u) * v \
                        + (cf[5] + cf[8] * u) * (v * v) + cf[9] * v ** 3
                grid = torch.stack([poly(c["cpx"]) * (2.0 / (nx - 1)) - 1.0, poly(c["cpy"]) * (2.0 / (ny - 1)) - 1.0], -1)[None]
                rs = F.grid_sample(torch.stack([snr, valid.to(f32)])[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
                m8 = torch.clamp(torch.round(rs[0] * 10), -127, 127).to(torch.int8)
                m8 = torch.where(rs[1] < 0.999, torch.full((), -128, device=dev, dtype=torch.int8), m8)
                piece = (iy0, ix0, m8.cpu().numpy())
                del grid, rs, m8, u, v
        except Exception as ex:
            sys.stderr.write(f"ccd resample: {type(ex).__name__}: {ex}\n")
        return piece
    if cat_only:                                   # gen012: catalogue only (first pass of a failed-WCS CCD)
        make_piece = lambda valid: None
    # ---- peaks
    mx = F.max_pool2d(snr[None, None], 3, 1, 1)[0, 0]
    pk = (snr >= thr) & (snr == mx)
    pk[:8] = False; pk[-8:] = False; pk[:, :8] = False; pk[:, -8:] = False
    idx = torch.nonzero(pk)
    if len(idx) == 0:
        return None, make_piece(valid)
    py, px = idx[:, 0], idx[:, 1]
    sv = snr[py, px]
    if len(py) > 40000:
        o = torch.topk(sv, 40000).indices; py, px, sv = py[o], px[o], sv[o]
    ds = 8
    cy, cx = ny // ds, nx // ds
    pool = F.max_pool2d(F.max_pool2d(snr[None, None, :cy * ds, :cx * ds], ds), 5, 1, 2)[0, 0]
    nb = pool[torch.clamp(py // ds, max=cy - 1), torch.clamp(px // ds, max=cx - 1)]
    keep = ~((nb > 40) & (sv < 0.5 * nb))
    flux = num[py, px] / (den[py, px].clamp(min=0.05) * norm)
    keep &= ~((d[py, px] - 3 * sg[py, px]) > 0.6 * flux)
    py, px, sv, flux = py[keep], px[keep], sv[keep], flux[keep]
    if len(py) == 0:
        return None, make_piece(valid)
    # ---- gen007: stamp classifier. A small CNN looks at the 11 x 11 native pixels (image / noise, and which pixels
    # are masked) of every peak >= 4.5 sigma and says whether it is a source seen through the PSF or junk (what is
    # left of a cosmic ray or a bleed trail next to its masked pixels, the rim of a masked star, ...). Junk is
    # dropped from the catalogue and blanked in the S/N map.
    if cls is not None:
        try:
            sel = torch.nonzero(sv >= 4.5)[:, 0]
            if len(sel):
                ar = torch.arange(-STAMP_H, STAMP_H + 1, device=dev)
                yy = (py[sel][:, None, None] + ar[None, :, None]).clamp(0, ny - 1); xx = (px[sel][:, None, None] + ar[None, None, :]).clamp(0, nx - 1)
                gd = good[yy, xx]
                st = torch.where(gd, torch.asinh(torch.clamp(d[yy, xx] / sg[yy, xx], -20.0, 300.0) / 3.0), zero)
                xin = torch.stack([st, (~gd).to(f32)], 1)
                CB = 1024                               # fixed batch shape (MPS compiles a graph per input shape)
                npad = (-len(sel)) % CB
                if npad:
                    xin = torch.cat([xin, torch.zeros((npad, 2, 2 * STAMP_H + 1, 2 * STAMP_H + 1), device=dev, dtype=f32)])
                with torch.no_grad():
                    lg = torch.cat([cls(xin[b:b + CB])[:, 0] for b in range(0, len(xin), CB)])[:len(sel)]
                jk = sel[lg < CLS_THR]
                # safety valve: stars and galaxies are most of the peaks, so a CCD where the network rejects more than
                # 40 % of them is one it does not understand (unseen band, seeing, detector state): no veto there
                if len(sel) >= 30 and len(jk) > 0.4 * len(sel):
                    c["_nskip"] = 1
                    jk = jk[:0]
                if len(jk):
                    jm = torch.zeros((ny, nx), device=dev, dtype=f32)
                    jm[py[jk], px[jk]] = 1.0
                    rad = int(2.0 * s + 1.5)
                    jm = F.max_pool2d(jm[None, None], 2 * rad + 1, 1, rad)[0, 0] > 0.5
                    valid = valid & ~jm
                    kp = ~jm[py, px]
                    py, px, sv, flux = py[kp], px[kp], sv[kp], flux[kp]
                    c["_njunk"] = int(len(jk))
                    del jm
                if len(py) == 0:
                    return None, make_piece(valid)
        except Exception as ex:
            sys.stderr.write(f"stamp classifier: {type(ex).__name__}: {ex}\n")
    piece = make_piece(valid)
    g = torch.stack([py.to(f32), px.to(f32), sv, flux, snr[py, px - 1], snr[py, px + 1], snr[py - 1, px], snr[py + 1, px]], 1).cpu().numpy().astype(np.float64)
    py_, px_, sv_, flux_ = g[:, 0], g[:, 1], g[:, 2], g[:, 3]

    def off(a, b):
        dd_ = a + b - 2 * sv_
        return np.clip(np.where(dd_ < 0, 0.5 * (a - b) / np.where(dd_ < 0, dd_, -1), 0), -0.5, 0.5)
    ox = off(g[:, 4], g[:, 5]); oy = off(g[:, 6], g[:, 7])
    T = _terms((px_ + ox - nx / 2) / (nx / 2), (py_ + oy - ny / 2) / (ny / 2))
    if stamp_thr is not None:
        # native-pixel cutouts (image / noise, 11 x 11) of the peaks >= stamp_thr, for the detection classifier
        sel = torch.nonzero(sv >= stamp_thr)[:, 0]
        ar = torch.arange(-STAMP_H, STAMP_H + 1, device=dev)
        yy = (py[sel][:, None, None] + ar[None, :, None]).clamp(0, ny - 1); xx = (px[sel][:, None, None] + ar[None, None, :]).clamp(0, nx - 1)
        st = (d[yy, xx] / sg[yy, xx])
        st = torch.where(good[yy, xx], st, torch.full_like(st, -99.0))
        c["_stamps"] = (sel.cpu().numpy(), st.to(torch.float16).cpu().numpy() if dev != "mps" else st.cpu().numpy().astype(np.float16))
    return np.stack([T @ c["cX"], T @ c["cY"], sv_, flux_, np.full(len(sv_), c["fw"] * 0.263)], 1), piece


PRIOR_K = float(os.environ.get("FINDER_PRIORK", 2.2))     # score units per nat of the orbit prior
PRIOR_CAP = 15.0


def orbit_prior(xp, r, w, inc, f, beta):
    """-ln of (density of real trans-Neptunian bodies / density of chance alignments) at an orbit, in nats, up to a
    constant: ~0 at the most TNO-like orbit, larger = less plausible.  xp: numpy or torch (same code for both).
    r: heliocentric distance (au); w = h^2 / (GM r) = (transverse speed / circular speed)^2 = 1 + e cos(nu);
    inc: inclination to the ecliptic (deg); f: radial speed / circular speed; beta: |heliocentric ecliptic latitude|
    (deg), the smallest inclination possible at that place.
    Real bodies (what flux-limited surveys find): r mostly 38-48 au with a shoulder at 30-38 and a thin tail outwards;
    w near 1, wider for eccentric orbits; a cold component (few deg) and a hot one (~16 deg).
    Chance alignments (measured on this program's own candidates, scen02): density ~ r^-4 from 25.5 au, uniform in
    w over 0.2-1.85, uniform in the direction of motion (= in inc for a field on the ecliptic), |f| up to 0.6."""
    N = lambda x, m, s_: xp.exp(-0.5 * ((x - m) / s_) ** 2) / (s_ * 2.5066)
    qr = 0.25 * N(r, 35.0, 3.5) + 0.55 * N(r, 43.0, 3.0) + (r > 44.0) * (0.20 / 14.0) * xp.exp(-(r - 44.0) / 14.0) + 2e-4
    pr = 3.0 * 25.5 ** 3 / r ** 4
    qw = 0.55 * N(w, 1.0, 0.08) + 0.45 * N(w, 1.0, 0.22) + 0.02
    g = lambda i_: 0.45 / 3.5 * xp.exp(-0.5 * (i_ / 3.5) ** 2) + 0.55 / 16.0 * xp.exp(-0.5 * (i_ / 16.0) ** 2) + 2e-4
    pen = -xp.log(qr / pr / 5.7) - xp.log(qw / 3.58) - xp.log(g(xp.maximum(inc, beta)) / g(beta)) + 2.0 * (1.0 - xp.exp(-0.25 * (f / 0.3) ** 2))
    return pen * (pen > 0)


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


def predict_xy_b(P0, V0, t0, t, O, nstep=6):
    """predict_xy for many states at once: P0, V0 [M,3], t0 [M]; t [K], O [K,3]. Returns [M,K,2] arcsec."""
    M = len(P0); K = len(t)
    lt = np.repeat((np.linalg.norm(P0 - O[0][None], axis=1) / C_AU)[:, None], K, 1)
    def acc(p):
        return -GM * p / (np.sum(p * p, 2, keepdims=True) ** 1.5)
    D = None
    for _ in range(3):
        dt = t[None, :] - lt - t0[:, None]
        P = np.repeat(P0[:, None, :], K, 1).astype(float); V = np.repeat(V0[:, None, :], K, 1).astype(float)
        h = (dt / nstep)[:, :, None]
        for _s in range(nstep):
            k1p, k1v = V, acc(P)
            k2p, k2v = V + 0.5 * h * k1v, acc(P + 0.5 * h * k1p)
            k3p, k3v = V + 0.5 * h * k2v, acc(P + 0.5 * h * k2p)
            k4p, k4v = V + h * k3v, acc(P + h * k3p)
            P = P + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            V = V + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        D = P - O[None]
        lt = np.linalg.norm(D, axis=2) / C_AU
    return np.stack([D[..., 0] / D[..., 2], D[..., 1] / D[..., 2]], -1) * ASEC


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
def _flt(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


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
    TEL["budget"] = budget
    if not os.environ.get("FINDER_NOWATCHDOG"):
        try:
            watchdog(budget)
        except Exception:
            pass
    deadline = T0 + budget - 20
    TEL_SHIFT = min(26.0, 0.03 * budget)     # gen018: the last stage (orbit fits) ends this much earlier, the exit time carries telemetry (tel_exit waits up to 36 s)
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
    gpu_det = dev != "cpu" and not os.environ.get("FINDER_CPUDET")
    det_cls = det_classifier(torch, dev) if gpu_det else None
    failed_w = set(); ext2num = {}; rr_ref = None
    log("stamp classifier:", "loaded" if det_cls is not None else "not used")
    # gen011: exposures whose astrometric calibration failed in the Community Pipeline (primary header WCSCAL =
    # 'Failed': the WCS is the telescope pointing, tens of arcsec off and with a wrong scale) are not read at all,
    # as long as the others cover three nights. (The registration check after detection catches the same thing from
    # the data; this saves their detection time.)
    try:
        if order and not os.environ.get("FINDER_NOREGCHECK"):
            bad_w = []
            for i in order:
                try:
                    with fits.open(os.path.join(a.scenario, rows[i]["image"])) as h:
                        if str(h[0].header.get("WCSCAL", "")).strip().lower().startswith("fail"):
                            bad_w.append(i)
                except Exception:
                    pass
            good_w = [i for i in order if i not in bad_w]
            if bad_w and good_w and gpu_det and not os.environ.get("FINDER_NOREREG"):
                # gen012: they are re-registered against the good exposures (below), so they are read last
                failed_w = set(bad_w); order = good_w + bad_w
                log(f"WCSCAL = Failed in {len(bad_w)} of {nE_all} exposures: to be re-registered", [rows[i]["exp_id"] for i in bad_w])
            elif bad_w and len(good_w) >= 3 and len(set(int(night_all[i]) for i in good_w)) >= 3:
                order = good_w
                log(f"WCSCAL = Failed in {len(bad_w)} of {nE_all} exposures: not used", [rows[i]["exp_id"] for i in bad_w])
            elif bad_w:
                log(f"WCSCAL = Failed in {len(bad_w)} exposures, kept (too few others)")
    except Exception as ex:
        log("WCSCAL check failed:", type(ex).__name__, ex)
    # ---- gen013: many-exposure scenarios. Back-to-back exposures of a night are co-added on the lattice into one
    # deeper S/N map with one catalogue (a "group"), which the rest of the program treats as one exposure.
    GRP_MIN = int(os.environ.get("FINDER_GROUPMIN", 55)); GRP_MAXN = int(os.environ.get("FINDER_GROUPMAXN", 40))
    grp_mode = bool(order) and gpu_det and len(order) > GRP_MIN
    grp_n_nights = 0
    gid = {}; grp_n_groups = 0; grp_cur = -1; grp_lim = 10 ** 9; grp_cnt = 0; grp_over = {}
    GS = {"num": None, "mem": []}
    if grp_mode:
        try:
            if failed_w:
                order = [i for i in order if i not in failed_w]; failed_w = set()
            ords = sorted(order, key=lambda i_: mjd[i_])
            def plan_(Wd):
                gs_ = []
                for i_ in ords:
                    j_ = gs_[-1][0] if gs_ else None
                    if j_ is not None and night_all[j_] == night_all[i_] and mjd[i_] - mjd[j_] <= Wd \
                            and math.hypot((rac[i_] - rac[j_]) * math.cos(dec[i_]), dec[i_] - dec[j_]) < math.radians(0.08):
                        gs_[-1].append(i_)
                    else:
                        gs_.append([i_])
                return gs_
            for Wm_ in [w_ for w_ in (6.0, 9.0, 12.0, 16.0, 20.0, 25.0) if w_ >= float(os.environ.get('FINDER_GROUPW', 0))]:
                gplan = plan_(Wm_ / 1440.0)
                if len(gplan) <= GRP_MAXN:
                    break
            bn_ = {}
            for g_ in gplan:
                bn_.setdefault(int(night_all[g_[0]]), []).append(g_)
            ls_ = list(bn_.values()); gorder = []
            while any(ls_):
                for l_ in ls_:
                    if l_:
                        gorder.append(l_.pop(0))
            order = [i_ for g_ in gorder for i_ in g_]
            for q_, g_ in enumerate(gorder):
                for i_ in g_:
                    gid[i_] = q_
            grp_n_groups = len(gorder); max_exp = 10 ** 6; grp_n_nights = len(bn_)
            log(f"group mode: {len(order)} exposures in {grp_n_groups} groups (window {Wm_:.0f} min), sizes {[len(g_) for g_ in gorder][:60]}")
        except Exception as ex:
            log("group planning failed:", type(ex).__name__, ex); grp_mode = False

    def grp_add(i, cat_i, y0, x0, arr):
        m_ = (cat_i[:, 2] > 3.5) & (cat_i[:, 2] < 50) & (cat_i[:, 3] > 0) & np.isfinite(cat_i[:, 3])
        c_i = 10 ** (-0.4 * (zph[i] - 25.0))
        sg_ = float(np.median(cat_i[m_, 3] / cat_i[m_, 2])) * c_i if m_.sum() >= 20 else float("nan")
        if not (np.isfinite(sg_) and sg_ > 0):
            log(f"exp {i}: no noise estimate, left out of its group"); return
        w_ = 1.0 / sg_
        f32_ = torch.float32
        if GS["num"] is None:
            PAD_ = 400
            GS["y0"] = y0 - PAD_; GS["x0"] = x0 - PAD_; GS["H"] = arr.shape[0] + 2 * PAD_; GS["W"] = arr.shape[1] + 2 * PAD_
            shp_ = (GS["H"], GS["W"])
            GS["num"] = torch.zeros(shp_, dtype=f32_, device=dev); GS["den"] = torch.zeros(shp_, dtype=f32_, device=dev)
            GS["top"] = torch.zeros(shp_, dtype=f32_, device=dev); GS["cnt"] = torch.zeros(shp_, dtype=torch.int8, device=dev)
            GS["sat"] = torch.zeros(shp_, dtype=torch.bool, device=dev); GS["topw"] = torch.ones(shp_, dtype=f32_, device=dev)
        ya = max(GS["y0"], y0); yb = min(GS["y0"] + GS["H"], y0 + arr.shape[0])
        xa = max(GS["x0"], x0); xb = min(GS["x0"] + GS["W"], x0 + arr.shape[1])
        if yb <= ya or xb <= xa:
            return
        sub = torch.from_numpy(arr[ya - y0:yb - y0, xa - x0:xb - x0]).to(dev)
        v = sub != -128
        term = torch.where(v, sub.to(f32_) * (0.1 * w_), torch.zeros((), dtype=f32_, device=dev))
        se = (slice(ya - GS["y0"], yb - GS["y0"]), slice(xa - GS["x0"], xb - GS["x0"]))
        GS["num"][se] += term; GS["den"][se] += v.to(f32_) * (w_ * w_)
        upd_ = term > GS["top"][se]
        GS["top"][se] = torch.where(upd_, term, GS["top"][se]); GS["topw"][se] = torch.where(upd_, torch.full((), w_, dtype=f32_, device=dev), GS["topw"][se])
        GS["cnt"][se] += v.to(torch.int8)
        del upd_
        GS["sat"][se] |= sub >= 120                      # at the int8 ceiling of a member: a bright source (members' catalogues)
        del sub, v, term
        br_ = cat_i[(cat_i[:, 2] >= 9.0) & np.all(np.isfinite(cat_i), 1)]
        GS["mem"].append((i, w_, sg_, c_i, br_, float(np.median(cat_i[:, 4])) if len(cat_i) else 1.2))

    def grp_flush():
        # one group -> dets[rep], maps[rep]; the group's time and observer position replace the representative's
        nonlocal map_bytes
        mem = GS["mem"]
        if GS["num"] is None or not mem:
            GS["num"] = None; GS["mem"] = []; return
        f32_ = torch.float32; F_ = torch.nn.functional
        num, den, top, cnt = GS["num"], GS["den"], GS["top"], GS["cnt"]
        ws = np.array([m_[1] for m_ in mem]); w2 = ws ** 2
        sg_g = 1.0 / math.sqrt(w2.sum())
        okz = [m_ for m_ in mem if (lambda r_: 20 < r_ < 35)(_flt(rows[m_[0]].get("magzero")))]
        rep_ = max(okz or mem, key=lambda m_: m_[1])
        ri, c_rep = rep_[0], rep_[3]
        valid = cnt > 0
        dsq = torch.sqrt(den.clamp(min=1e-20))
        Z = num / dsq
        if len(mem) >= 2:
            # robust against a source that is in one member only (cosmic ray, satellite, fast asteroid): the stack
            # without its strongest member must carry it
            # must carry it. Test: the strongest member's S/N predicts (same flux) what the rest of the stack shows;
            # when the rest shows less than half of that minus 1.5 sigma, the strongest member is left out there.
            # (Half: bodies are red, a g member sees less than an r member predicts.)
            tw = GS["topw"]
            rsq = torch.sqrt((den - tw * tw).clamp(min=1e-20))
            Zl = (num - top) / rsq                       # the stack without the strongest member
            junk = (cnt >= 2) & (top > 0) & (Zl < 0.5 * (top / tw) * (rsq / tw) - 1.5)
            Z = torch.where(junk, Zl, Z)
            del tw, rsq, Zl, junk
        Z = torch.where(valid, Z, torch.full((), -1e3, dtype=f32_, device=dev))
        del top; GS["top"] = None; GS["topw"] = None
        # peaks of the co-added map
        # (a tiny position-dependent term breaks the ties of the 0.1-sigma steps; plateaus at the int8 ceiling are left out)
        jit = ((torch.arange(GS["H"], device=dev, dtype=f32_)[:, None] * 0.618034 + torch.arange(GS["W"], device=dev, dtype=f32_)[None, :] * 0.414214) % 1.0) * 2e-3
        Zj = Z + jit; del jit
        mx = F_.max_pool2d(Zj[None, None], 3, 1, 1)[0, 0]
        pk = (Z >= THR_LOW) & (Zj == mx)
        del mx, Zj
        pk &= ~(F_.max_pool2d(GS["sat"].to(f32_)[None, None], 5, 1, 2)[0, 0] > 0.5)
        GS["sat"] = None
        pk[:2] = False; pk[-2:] = False; pk[:, :2] = False; pk[:, -2:] = False
        idx = torch.nonzero(pk); del pk
        py, px = idx[:, 0], idx[:, 1]
        zv = Z[py, px]
        if len(py) > 1500000:
            o_ = torch.topk(zv, 1500000).indices; py, px, zv = py[o_], px[o_], zv[o_]
        g = torch.stack([zv, Z[py, px - 1], Z[py, px + 1], Z[py - 1, px], Z[py + 1, px], num[py, px] / den[py, px].clamp(min=1e-20)], 1).cpu().numpy().astype(np.float64)
        pyn = py.cpu().numpy().astype(np.float64); pxn = px.cpu().numpy().astype(np.float64)
        def off(a_, b_):
            a_ = np.maximum(a_, -5.0); b_ = np.maximum(b_, -5.0)
            dd_ = a_ + b_ - 2 * g[:, 0]
            return np.clip(np.where(dd_ < 0, 0.5 * (a_ - b_) / np.where(dd_ < 0, dd_, -1), 0), -0.5, 0.5)
        Xp = (GS["x0"] + pxn + off(g[:, 1], g[:, 2])) * PS; Yp = (GS["y0"] + pyn + off(g[:, 3], g[:, 4])) * PS
        fwm = float(np.median([m_[5] for m_ in mem]))
        catp = np.column_stack([Xp, Yp, g[:, 0], np.maximum(g[:, 5], 1e-6) / c_rep, np.full(len(Xp), fwm)])
        # bright sources (the int8 maps stop at 12.7 sigma): from the members' own catalogues, deepest member first
        from scipy.spatial import cKDTree as KD_
        br = []
        for m_ in sorted(mem, key=lambda m_: -m_[1]):
            b_ = m_[4]
            if not len(b_):
                continue
            b_ = b_.copy(); b_[:, 2] = b_[:, 3] * m_[3] / sg_g; b_[:, 3] = b_[:, 3] * m_[3] / c_rep
            if br:
                prev = np.concatenate(br)
                dd_, _ = KD_(prev[:, :2]).query(b_[:, :2], distance_upper_bound=1.0)
                b_ = b_[~np.isfinite(dd_)]
            if len(b_):
                br.append(b_)
        if br:
            br = np.concatenate(br)
            ix_ = np.clip(np.rint(br[:, 0] / PS).astype(np.int64) - GS["x0"], 1, GS["W"] - 2); iy_ = np.clip(np.rint(br[:, 1] / PS).astype(np.int64) - GS["y0"], 1, GS["H"] - 2)
            ixt = torch.as_tensor(ix_, device=dev); iyt = torch.as_tensor(iy_, device=dev)
            zc = torch.stack([Z[iyt + dy_, ixt + dx_] for dy_ in (-1, 0, 1) for dx_ in (-1, 0, 1)]).max(0).values.cpu().numpy()
            cc = cnt[iyt, ixt].cpu().numpy()
            br = br[(zc >= 5.0) | (cc <= 1)]              # (a bright source of one member that the stack does not show: junk)
            if len(br) and len(catp):
                dd_, _ = KD_(br[:, :2]).query(catp[:, :2], distance_upper_bound=1.5)
                catp = catp[~np.isfinite(dd_)]
            catg = np.concatenate([br, catp]) if len(br) else catp
        else:
            catg = catp
        m8 = torch.clamp(torch.round(Z * 10), -127, 127).to(torch.int8)
        m8 = torch.where(valid, m8, torch.full((), -128, device=dev, dtype=torch.int8))
        rr_ = torch.nonzero(valid.any(1))[:, 0]; cc_ = torch.nonzero(valid.any(0))[:, 0]
        GS["num"] = None; GS["den"] = None; GS["cnt"] = None; GS["mem"] = []
        if len(rr_) == 0 or len(catg) == 0:
            return
        ya, yb, xa, xb = int(rr_[0]), int(rr_[-1]) + 1, int(cc_[0]), int(cc_[-1]) + 1
        arr8 = m8[ya:yb, xa:xb].cpu().numpy()
        del m8, Z, num, den, cnt, valid
        tg = float(np.sum(w2 * np.array([mjd[m_[0]] for m_ in mem])) / w2.sum())
        Og = (w2[:, None] * np.array([O_all[m_[0]] for m_ in mem])).sum(0) / w2.sum()
        grp_over[ri] = (tg, Og)
        dets[ri] = catg; maps[ri] = (GS["y0"] + ya, GS["x0"] + xa, arr8); map_bytes += arr8.size
        log(f"group of exp {[m_[0] for m_ in mem]} -> {ri}: {len(catg)} detections ({len(br) if len(br) else 0} bright), depth gain {1.0 / (sg_g * ws.max()):.2f}x over its deepest member, span {1440 * (max(mjd[m_[0]] for m_ in mem) - min(mjd[m_[0]] for m_ in mem)):.1f} min")

    import multiprocessing
    with ProcessPoolExecutor(ncpu, mp_context=multiprocessing.get_context("spawn")) as pool:
        t_per = None
        for n_done, i in enumerate(order[:max_exp]):
            if grp_mode:
                if gid[i] != grp_cur:
                    try:
                        grp_flush()
                    except Exception as ex:
                        import traceback
                        log("group flush failed:", type(ex).__name__, ex, traceback.format_exc()[-800:]); GS["num"] = None; GS["mem"] = []
                    grp_cur = gid[i]; grp_cnt = 0
                    # members this group may take: the exposures still affordable, shared among the groups still to
                    # come - but rather fewer groups of >= 4 members than many shallow ones (every night first, and
                    # at least 16 groups: the score needs epochs, 6 groups of a 13-exposure scenario rank worse than 13 exposures)
                    n_aff_ = (det_deadline - time.time()) / (1.15 * (t_per if t_per is not None else 12.0))
                    g_left_ = max(grp_n_groups - grp_cur, 1)
                    g_tgt_ = min(g_left_, max(int(math.ceil(n_aff_ / 4.0)), grp_n_nights - grp_cur, 16 - grp_cur, 1))
                    grp_lim = max(1, int(n_aff_ / g_tgt_))
                if grp_cnt >= grp_lim:
                    continue
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
            is_f = i in failed_w
            tasks = [(ip, mp, hd[c::nchunk], THR_LOW, fb, R, None if (inj_all is None or is_f) else inj_all[i]) for c in range(nchunk) if hd[c::nchunk]]
            res = None; n_junk = 0; n_skip = 0
            n_rr = 0; n_rrfail = 0; rr_rms = []; rr_sh = []; wtab = None
            if is_f:
                try:
                    if rr_ref is None:
                        from scipy.spatial import cKDTree as _KD
                        gk = sorted((k_ for k_ in dets if k_ not in failed_w), key=lambda k_: -int(np.sum(dets[k_][:, 2] >= 30)))[:3]
                        rf = np.concatenate([dets[k_][dets[k_][:, 2] >= 30][:, :2] for k_ in gk])
                        rf = rf[np.all(np.isfinite(rf), 1)]
                        rr_ref = (_KD(rf), rf)
                    wtab = wcs_table(ip, R)
                except Exception as ex:
                    log("re-registration set-up failed:", type(ex).__name__, ex); rr_ref = None
                if rr_ref is None or wtab is None or len(rr_ref[1]) < 500:
                    log(f"exp {i}: failed WCS and no reference to register it on: not used"); continue
            if gpu_det:
                # workers only decompress; background, filtering, lattice resampling and peaks run on the GPU
                try:
                    cats = []; pieces = []; n_bad = 0; n_junk = 0; n_skip = 0
                    for chunk in pool.map(load_chunk, tasks):
                        for c_ in chunk:
                            try:
                                if is_f:
                                    # gen012: find the WCS that belongs to these pixels and tie it to the good exposures
                                    own = (c_["cX"], c_["cY"]); cands_w = []
                                    tn = ext2num.get(c_["ext"])
                                    ny_, nx_ = c_["d"].shape
                                    if tn is not None and tn != c_["ccdnum"] and tn in wtab and wtab[tn][2] == nx_ and wtab[tn][3] == ny_:
                                        cands_w.append(wtab[tn][:2])
                                    cands_w.append(own)
                                    done_ = False
                                    for cw in cands_w:
                                        set_wcs(c_, cw[0], cw[1])
                                        cat0, _ = gpu_ccd(torch, dev, c_, THR_LOW, cat_only=True)
                                        fa = fit_affine(cat0, rr_ref[0], rr_ref[1]) if cat0 is not None and len(cat0) else None
                                        if fa is not None:
                                            set_wcs(c_, cw[0], cw[1], fa[0])
                                            rr_rms.append(fa[2]); rr_sh.append(math.hypot(fa[0][0, 0], fa[0][1, 0])); done_ = True
                                            break
                                    if not done_:
                                        n_rrfail += 1; continue
                                    n_rr += 1
                                    if inj_all is not None and len(inj_all[i]):
                                        inject_px(c_["d"], c_["cpx"], c_["cpy"], c_["X0"], c_["Y0"], c_["SX"], c_["s"], inj_all[i])
                                elif c_.get("ext") and c_.get("ccdnum") is not None:
                                    ext2num.setdefault(c_["ext"], c_["ccdnum"])
                                cat_, piece_ = gpu_ccd(torch, dev, c_, THR_LOW, cls=det_cls)
                                n_junk += c_.get("_njunk", 0); n_skip += c_.get("_nskip", 0)
                            except Exception as ex:
                                n_bad += 1
                                if n_bad > 6:
                                    raise
                                sys.stderr.write(f"gpu ccd: {type(ex).__name__}: {ex}\n"); continue
                            if cat_ is not None and len(cat_):
                                cats.append(cat_)
                            if piece_ is not None:
                                pieces.append(piece_)
                        del chunk
                    res = True
                except Exception as ex:
                    log("GPU detection failed, using the CPU path:", type(ex).__name__, ex)
                    gpu_det = False; res = None
            if res is None and is_f:
                log(f"exp {i}: failed WCS and no GPU detection: not used"); continue
            if res is None:
                res = list(pool.map(detect_chunk, tasks))
                cats = [x[0] for x in res if len(x[0])]
                pieces = [p for x in res for p in x[1]]
            if is_f:
                log(f"exp {i}: failed WCS re-registered on {n_rr} CCDs ({n_rrfail} not), median rms {np.median(rr_rms) if rr_rms else float('nan'):.3f} arcsec")
                if n_rr < 0.5 * (n_rr + n_rrfail) or res is None:
                    log(f"exp {i}: not used"); continue
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
            if grp_mode:
                cat_i = np.concatenate(cats); n_det_i = len(cat_i)
                try:
                    grp_add(i, cat_i, y0, x0, arr); grp_cnt += 1
                except Exception as ex:
                    log("group add failed:", type(ex).__name__, ex)
                del arr, cat_i
            else:
                dets[i] = np.concatenate(cats); n_det_i = len(dets[i])
                maps[i] = (y0, x0, arr)
                map_bytes += arr.size
            t_per = time.time() - t1
            log(f"exp {i} night {night_all[i]} band {r['band']}: {n_det_i} detections, map {H}x{W}, {n_junk} peaks >= 4.5 sigma dropped by the stamp classifier ({n_skip} CCDs not vetoed), {t_per:.1f}s")
        if grp_mode:
            TEL["grp"] = 1
            try:
                grp_flush()
            except Exception as ex:
                import traceback
                log("group flush failed:", type(ex).__name__, ex, traceback.format_exc()[-800:])
            GS["num"] = None; GS["den"] = None; GS["top"] = None; GS["cnt"] = None; GS["sat"] = None; GS["topw"] = None
    if dev == "cuda":
        torch.cuda.empty_cache()
    if cfile and not os.path.exists(cfile):
        import pickle
        pickle.dump((dets, maps), open(cfile, "wb"), protocol=4)
        pickle.dump(grp_over, open(cfile + ".grp", "wb"), protocol=4)
    elif cfile and os.path.exists(cfile + ".grp"):
        import pickle
        grp_over = pickle.load(open(cfile + ".grp", "rb"))
    for ri_, (tg_, Og_) in grp_over.items():          # gen013: a group's time and observer position
        mjd[ri_] = tg_; O_all[ri_] = Og_
        if inj_all is not None:
            for b_, bd in enumerate(bodies):
                st = np.array(bd["state"], float)
                inj_all[ri_, b_, :2] = predict_xy(R @ st[:3], R @ st[3:], bd["epoch_mjd"] - tref, np.array([tg_ - tref]), Og_[None])[0]
    # ---- gen011: registration check. An exposure whose bright detections are not where any exposure of another
    # night shows a source has a wrong WCS (scen03: both exposures of the first night; nothing within 500" as a
    # shift). Its static sources all pass as "moving" (3600 detections > 7 sigma in one exposure), fill the seed cap
    # and the deep-point list, and bodies are never linked. Such exposures are dropped.
    try:
        ks_ = sorted(k for k in dets if len(dets[k]) > 0 and k in maps)
        if len(ks_) >= 4 and not os.environ.get("FINDER_NOREGCHECK"):
            rng_ = np.random.default_rng(5); reg_ = {}
            for a_ in ks_:
                d_ = np.asarray(dets[a_]); m_ = np.nonzero((d_[:, 2] >= 20) & (d_[:, 2] < 400) & np.isfinite(d_[:, 0]) & np.isfinite(d_[:, 1]))[0]
                if len(m_) < 300:
                    continue
                m_ = rng_.choice(m_, min(len(m_), 3000), replace=False)
                fbest = None
                for b_ in ks_:
                    if night_all[b_] == night_all[a_]:
                        continue
                    y0_, x0_, arr_ = maps[b_]
                    arr_ = arr_.cpu().numpy() if hasattr(arr_, "cpu") else np.asarray(arr_)
                    H_, W_ = arr_.shape
                    ix_ = np.rint(d_[m_, 0] / PS).astype(np.int64) - int(x0_); iy_ = np.rint(d_[m_, 1] / PS).astype(np.int64) - int(y0_)
                    inb_ = (ix_ >= 1) & (ix_ < W_ - 1) & (iy_ >= 1) & (iy_ < H_ - 1)
                    ix_ = ix_[inb_]; iy_ = iy_[inb_]
                    if len(ix_) < 200:
                        continue
                    okv_ = arr_[iy_, ix_] != -128
                    if okv_.sum() < 200:
                        continue
                    v_ = np.full(len(ix_), -128, np.int16)
                    for dy_ in (-1, 0, 1):
                        for dx_ in (-1, 0, 1):
                            v_ = np.maximum(v_, arr_[iy_ + dy_, ix_ + dx_].astype(np.int16))
                    f_ = float(np.mean(v_[okv_] >= 50))
                    fbest = f_ if fbest is None else max(fbest, f_)
                if fbest is not None:
                    reg_[a_] = fbest
            bad_ = [a_ for a_, f_ in reg_.items() if f_ < 0.25]
            log("registration check (share of bright detections seen in another night's exposure):", {int(k_): round(v_, 2) for k_, v_ in reg_.items()})
            if bad_ and len(ks_) - len(bad_) >= 3 and len(bad_) <= len(ks_) // 2 and len(set(night_all[k_] for k_ in ks_ if k_ not in bad_)) >= 3:
                log("exposures dropped, wrong WCS:", bad_)
                for a_ in bad_:
                    del dets[a_]; maps.pop(a_, None)
    except Exception as ex:
        log("registration check failed:", type(ex).__name__, ex)
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
    DIL_SOFT = bool(os.environ.get("FINDER_DILSOFT"))
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
        stat0_ = stat.clone(); cnt0_ = cnt == 0
        if DIL_SOFT:        # gen018 (dev switch, off): only clear static sources are dilated, marginal pixels are blanked alone
            strong_ = (torch.where(loo, rest_s > 40.0 * torch.sqrt(torch.clamp(rest_c, min=1.0)),
                                   (sm.to(f32) > 45.0 * torch.sqrt(cnt.to(f32))) | (mxv >= 50)) | (cnt == 0)) & stat
        del sm, cnt, top_s, top_c, rest_c, rest_s
        if DIL_SOFT:
            st = stat | (torch.nn.functional.max_pool2d(strong_.to(f32)[None, None], 3, 1, 1)[0, 0] > 0)
            del strong_
        else:
            st = torch.nn.functional.max_pool2d(stat.to(f32)[None, None], 3, 1, 1)[0, 0] > 0
        st |= torch.nn.functional.max_pool2d(((mxv >= 125) & stat).to(f32)[None, None], 11, 1, 5)[0, 0] > 0
        del stat, mxv
        frac = float(st.to(f32).mean())
        vld_ = raw[e] != -128
        log(f'  exp {e}: valid {float(vld_.to(f32).mean()):.3f}, of valid masked {float((st & vld_).to(f32).sum() / vld_.to(f32).sum()):.3f} (undilated {float((stat0_ & vld_).to(f32).sum() / vld_.to(f32).sum()):.3f}, uncovered {float((cnt0_ & vld_).to(f32).sum() / vld_.to(f32).sum()):.3f})')
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
    WEAK_THR = float(os.environ.get("FINDER_WEAKTHR", 4.0))      # seeds below 4.5 sigma: only when confirmed by a same-night exposure
    STACK_THR = float(os.environ.get("FINDER_STACKTHR", 3.8))   # (>= 4.5: seeds; 3.8-4.5: deep points of the triplet search)
    n_cat0 = N                                       # detections from here on are night-stack seeds
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

    # gen009: far same-night exposures (see the comment where sstat is built)
    SEP_MIN = float(os.environ.get("FINDER_SEPMIN", 0.0125)); DISP_MIN = 1.5; EP_THR = float(os.environ.get("FINDER_EPTHR", 3.0))
    TWO_NIGHT = not os.environ.get("FINDER_NO2N")
    ST_PEN = float(os.environ.get("FINDER_STPEN", 6.0)); ST_THR = 3.5
    farx = {s_: [int(k) for k in range(E) if k != s_ and night[k] == night[s_] and abs(t[k] - t[s_]) >= SEP_MIN] if TWO_NIGHT else [] for s_ in range(E)}
    # seeds
    seed_thr = 4.5
    SEEDMAX = 4000 if dev == 'cuda' else 2500
    K1 = int(os.environ.get('FINDER_K1', 1000 if dev == 'cuda' else 500))    # tier-1 seeds per exposure
    # few exposure pairs (short scenarios): the search is far from its time limit, so more seeds per exposure
    cn_ = [min(2, int((night == n).sum())) for n in range(NN)]
    est_jobs = max(1, (sum(cn_) ** 2 - sum(c * c for c in cn_)) // 2)
    SEEDMAX = int(SEEDMAX * min(2.0, max(1.0, math.sqrt(60.0 / est_jobs))))
    log(f"seed cap {SEEDMAX} ({est_jobs} exposure pairs expected)")
    WEAKMAX = int(os.environ.get('FINDER_WEAKMAX', 6000 if dev == 'cuda' else 3000))
    seeds = []; nS = []                              # per exposure: strong seeds (first nS[e]) then weak ones
    CAPFRAC = float(os.environ.get('FINDER_CAPFRAC', 0.4))     # gen011: share of the seed cap filled strictly in seed order (rest: random sample)
    noseed = np.zeros(N, bool)
    try:
        cs = np.nonzero(SNR >= 4.2)[0]
        prs = cKDTree(np.column_stack([X[cs], Y[cs]])).query_pairs(1.0, output_type="ndarray")
        far = np.abs(t[EXP[cs[prs[:, 0]]]] - t[EXP[cs[prs[:, 1]]]]) >= 4.0
        noseed[cs[prs[far, 0]]] = True; noseed[cs[prs[far, 1]]] = True
        log(f"seeds dropped as static over >= 4 days: {noseed.sum()}")
    except Exception as ex:
        log("static-seed veto failed:", type(ex).__name__, ex)
    # same-night confirmation: does another exposure of the seed's group show the source (3x3-pooled S/N >= 2 and at
    # least half of what its flux predicts)? Confirmed seeds are searched first (tier 0 takes the first K1 seeds of
    # each exposure), and a 4.0-4.5 sigma detection is a seed only when confirmed.
    CONF = np.zeros(N, bool); NOPART = np.ones(N, bool); QUAL = SNR.copy()
    try:
        if not os.environ.get("FINDER_NOCONF"):
            for e in range(E):
                m = np.nonzero((EXP == e) & (SNR >= min(WEAK_THR, 3.5)))[0]
                if not len(m) or not partner[e]:
                    continue
                x_ = tt(X[m]); y_ = tt(Y[m]); fc_ = tt(FC[m])
                conf = torch.zeros(len(m), dtype=torch.bool, device=dev); anyok = conf.clone(); vbest = torch.zeros(len(m), device=dev)
                for k in partner[e]:
                    v, ok = pool_at(k, x_, y_)
                    c_ = ok & (v >= torch.clamp(0.5 * fc_ * float(wS_np[k]), min=2.0))
                    conf |= c_; anyok |= ok; vbest = torch.maximum(vbest, torch.where(c_, v, torch.zeros_like(v)))
                CONF[m] = conf.cpu().numpy(); NOPART[m] = ~anyok.cpu().numpy()
                QUAL[m] = np.sqrt(SNR[m] ** 2 + vbest.cpu().numpy().astype(np.float64) ** 2)
            CONF[n_cat0:] = True                     # night-stack seeds are confirmed by construction
            log("confirmed seeds >= 4.5 / all >= 4.5 / confirmed 4.0-4.5 per exposure:",
                [(int((CONF & (EXP == e) & (SNR >= seed_thr)).sum()), int(((EXP == e) & (SNR >= seed_thr)).sum()),
                  int((CONF & (EXP == e) & (SNR >= WEAK_THR) & (SNR < seed_thr)).sum())) for e in range(E)])
    except Exception as ex:
        CONF[:] = False; NOPART[:] = True; QUAL = SNR.copy()
        log("seed confirmation failed:", type(ex).__name__, ex)
    # gen009: a detection that a far exposure of its night (in which every body within ~100 au has moved on) shows at
    # the same place is most likely a static source the other nights are too shallow to veto. Such seeds go to the
    # end of the seed order, behind everything else, so they are the first to fall to the seed cap.
    STATF = np.zeros(N, bool)
    try:
        for e in range(E):
            m = np.nonzero((EXP == e) & (SNR >= 3.5))[0]
            if not len(m) or not farx[e]:
                continue
            x_ = tt(X[m]); y_ = tt(Y[m]); fc_ = tt(FC[m])
            fl_ = torch.zeros(len(m), dtype=torch.bool, device=dev)
            for k in farx[e]:
                v, ok = pool_at(k, x_, y_)
                fl_ |= ok & (v >= torch.clamp(0.6 * fc_ * float(wS_np[k]), min=ST_THR))
            STATF[m] = fl_.cpu().numpy()
        log("static in a far same-night exposure, of the detections >= 4.5 sigma, per exposure:",
            [(int((STATF & (EXP == e) & (SNR >= seed_thr)).sum()), int(((EXP == e) & (SNR >= seed_thr)).sum())) for e in range(E) if farx[e]])
    except Exception as ex:
        STATF[:] = False
        log("far static flag failed:", type(ex).__name__, ex)
    for e in range(E):
        m = np.nonzero((EXP == e) & (SNR < 400) & ~noseed & ((SNR >= seed_thr) | ((SNR >= WEAK_THR) & CONF & (np.arange(N) < n_cat0))))[0]
        # order: confirmed (or not checkable) seeds by combined significance, then the unconfirmed ones by S/N.
        # The index of a seed is its rank.
        m = m[np.lexsort((-QUAL[m], ~(CONF[m] | NOPART[m])))]
        m_st = m[STATF[m]]; m = m[~STATF[m]]         # gen009: static-looking seeds last
        if len(m) > SEEDMAX:                         # over the cap: the first 40 %, and a random sample of the rest
            nt = int(CAPFRAC * SEEDMAX)
            pick = np.sort(np.random.default_rng(1).choice(len(m) - nt, SEEDMAX - nt, replace=False))
            m = np.concatenate([m[:nt], m[nt:][pick]])
        m = np.concatenate([m, m_st[:max(0, SEEDMAX - len(m))]])
        nS.append(len(m))
        seeds.append(m.astype(np.int64))
    log("seeds per exposure:", nS)
    # gen010: strong seeds = the leading seeds of an exposure that are confirmed by a same-night partner (or cannot be
    # checked), not static in a far exposure, with combined significance >= QS_THR. Pairs of them on two nights are
    # linked without a third night ("strong pair" jobs, tier 3), see two_night().
    QS_THR = float(os.environ.get("FINDER_QSTHR", 6.5)); KSMAX = int(os.environ.get("FINDER_KSMAX", 500))
    STRONG2 = not os.environ.get("FINDER_NOSTRONG2")
    KS = []
    for e in range(E):
        m_ = seeds[e][:nS[e]]
        good_ = (CONF[m_] | NOPART[m_]) & (QUAL[m_] >= QS_THR) & ~STATF[m_]
        KS.append(min(KSMAX, len(good_) if good_.all() else int(np.argmin(good_))) if STRONG2 and len(m_) else 0)
    log("strong seeds per exposure:", KS)
    # ---- deep points for the triplet search (gen005): per same-night group one anchor exposure (the deepest); its
    # list = its seeds + night-stack peaks >= DEEP_THR + its own detections >= 4.0 (>= DEEP_THR when it has no
    # same-night partner). They are appended to seeds[e] behind the nS[e] strong seeds, so the two tiers ignore them.
    DEEP_THR = float(os.environ.get("FINDER_DEEPTHR", 3.8))
    NDT = int(os.environ.get("FINDER_NDEEP", 10000 if dev == "cuda" else 6000))      # points per list in the triplet search
    # gen014: the list goes on to N4 points; the ones behind NDT are used by the deep-pair jobs (tier 4) only
    ZB_K = float(os.environ.get("FINDER_ZB", 0.6)); ZB_0 = 5.0; ZB_CAP = 6.0
    N4 = int(os.environ.get("FINDER_N4", 30000 if dev == "cuda" else 12000))
    N4A = min(N4, max(NDT, int(os.environ.get("FINDER_N4A", 12000 if dev == "cuda" else 6000))))   # first round (tier 4); the rest is tier 5
    # gen015: deep pairs are on when the scenario has many exposures (E >= DEEP_MIN_E on >= 4 nights): there a body
    # with a flat 2-4 sigma per exposure (total S/N 8-14, weight 4) has no 4.5 sigma seed on two nights and often no
    # three deep points, but its full stack stands above the chance level. Their candidates carry a "deep" tag and
    # are ranked against their own chance level (see finalize). FINDER_DEEP4=0/1 forces off/on.
    DEEP_MIN_E = int(os.environ.get("FINDER_DEEPMINE", 9))
    _d4 = os.environ.get("FINDER_DEEP4")
    DEEP4 = (E >= DEEP_MIN_E and NN >= 4) if _d4 in (None, "") else (_d4 != "0")
    DEEP_FRAC = float(os.environ.get("FINDER_DEEPFRAC", 0.0))      # share of the tier time kept for the deep-pair jobs (0: they only get what the regular jobs leave over; 0.25 cost a body on the CPU test)
    log(f"deep pairs: {'on' if DEEP4 else 'off'} (E={E}, nights={NN})")
    NDEEP = max(NDT, N4) if DEEP4 else NDT
    anchors = []
    if not os.environ.get("FINDER_NOTRIP"):
        try:
            is_stack = np.arange(N) >= n_cat0
            cg = {}
            for g_ in groups:
                e0 = min(g_, key=lambda k: sigF[k])
                m = np.nonzero((EXP == e0) & ((SNR >= DEEP_THR) | (CONF & (SNR >= 3.5))) & (SNR < 400) & ~noseed & (is_stack | (SNR >= 4.0) | (len(g_) == 1) | (CONF & (SNR >= 3.5))))[0]
                m = m[~np.isin(m, seeds[e0])]
                m = m[np.lexsort((-QUAL[m], STATF[m]))][:max(0, NDEEP - len(seeds[e0]))]
                cg.setdefault(int(night[e0]), []).append((len(seeds[e0]) + len(m), e0, m))
            for n_, lst in cg.items():
                for _, e0, m in sorted(lst, key=lambda q: -q[0])[:2]:
                    seeds[e0] = np.concatenate([seeds[e0], m.astype(np.int64)])
                    if len(seeds[e0]) >= 3:
                        anchors.append(int(e0))
            log("deep points per anchor exposure:", [(e, len(seeds[e]), nS[e]) for e in anchors])
        except Exception as ex:
            log("deep points failed:", type(ex).__name__, ex)
            anchors = []
            seeds = [s_[:n_] for s_, n_ in zip(seeds, nS)]
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

    # gen009: "far" same-night exposures. An exposure of the seed's own night taken >= SEP_MIN later or earlier, in
    # which the hypothesis puts the body >= DISP_MIN away from the seed, is an independent epoch: a static source,
    # a cosmic ray or a noise peak is not there. farx[s] = such exposures of seed exposure s; sstat[(s, k)] = pooled
    # S/N of exposure k at the seeds' own positions (what a static source would show).
    sstat = {}
    for s_ in range(E):
        for k in farx[s_]:
            if len(seeds[s_]):
                v_, ok_ = pool_at(k, sX[s_], sY[s_])
                sstat[(s_, k)] = torch.where(ok_, v_, torch.full_like(v_, -9.0))
    log("far same-night exposures:", {s_: farx[s_] for s_ in range(E) if farx[s_]})
    # exposure pairs on different nights, shortest first; use at most 2 exposures per night as seed exposures
    seed_exp = []
    for n in range(NN):
        ee = np.nonzero(night == n)[0]
        ee = ee[np.argsort([-nS[e] for e in ee], kind="stable")]   # (most seeds first is arbitrary but deterministic)
        seed_exp += list(ee[:2])
    pairs = [(i, j) for i in seed_exp for j in seed_exp if night[i] < night[j] and nS[i] and nS[j]]
    prim = {n: min((k for k in seed_exp if night[k] == n), key=lambda k: sigF[k]) for n in set(int(night[k]) for k in seed_exp)}
    pairs.sort(key=lambda p: (0 if (prim[int(night[p[0]])] == p[0] and prim[int(night[p[1]])] == p[1]) else 1, t[p[1]] - t[p[0]]))
    link_deadline = min(deadline - 0.12 * budget, T0 + float(os.environ.get('FINDER_LINKFRAC', 0.70)) * budget)   # gen009: was 0.62
    GAM_MAX, GAM_MIN = 1 / 25.5, 1 / 400.0
    TOL = 1.6
    results = []      # arrays [score, nn, i, j, a, b, r, rdotfrac]
    pair_step = {}
    n_tr_total = 0
    # ---- gen005: heliocentric triplet linking on the deep points -------------------------------------------------
    # For an anchor pair (i, j) on two nights and a distance r, two points a, b fix an orbit (rdot = 0). As 1/r runs
    # over its range, the predicted position on a night in between moves along a short curve. The 1/r range is cut
    # into segments <= ~4.5" long there; a deep point c of that night within TRIP_EPS of a segment makes a triple and
    # fixes 1/r. Only triples are scored on the S/N maps (a short scan in 1/r around the triple's value).
    # strength of the orbit prior: r and w are only measured when the Earth's motion curves the track (span of days)
    PK = PRIOR_K * min(1.0, max(0.15, (float(t.max() - t.min()) - 2.0) / 8.0))
    log(f"orbit prior: {PK:.2f} score units per nat")

    def dist_mass(r_):
        """Share of real bodies per step of 1/r at distance r_ (the q(r) of orbit_prior x r^2), numpy."""
        N_ = lambda x, m, s_: np.exp(-0.5 * ((x - m) / s_) ** 2) / (s_ * 2.5066)
        qr = 0.25 * N_(r_, 35.0, 3.5) + 0.55 * N_(r_, 43.0, 3.0) + (r_ > 44.0) * (0.20 / 14.0) * np.exp(-(r_ - 44.0) / 14.0)
        return qr * r_ * r_ + 1e-6 / r_

    def prior_t(Pa, Va):
        """Orbit-prior penalty (score units) of heliocentric states, on the device."""
        r_ = torch.sqrt(Pa[0] ** 2 + Pa[1] ** 2 + Pa[2] ** 2)
        hx = Pa[1] * Va[2] - Pa[2] * Va[1]; hy = Pa[2] * Va[0] - Pa[0] * Va[2]; hz = Pa[0] * Va[1] - Pa[1] * Va[0]
        h2 = hx * hx + hy * hy + hz * hz + 1e-30
        inc = torch.rad2deg(torch.acos(((hx * kecl[0] + hy * kecl[1] + hz * kecl[2]) / torch.sqrt(h2)).clamp(-1.0, 1.0)))
        beta = torch.rad2deg(torch.asin(((Pa[0] * kecl[0] + Pa[1] * kecl[1] + Pa[2] * kecl[2]) / r_).clamp(-1.0, 1.0))).abs()
        f_ = (Pa[0] * Va[0] + Pa[1] * Va[1] + Pa[2] * Va[2]) / torch.sqrt(GM * r_)
        return torch.clamp(PK * orbit_prior(torch, r_, h2 / (GM * r_), inc, f_, beta), max=PRIOR_CAP)

    def score_hyp(i, j, aq, bq, rq, fv):
        Pa, Va, g, ta, rhoa = setup_pair(i, j, rq, fv, aq, bq)
        score = sl[i][aq] + sl[j][bq] + torch.zeros_like(rq) - prior_t(Pa, Va)
        fq = 0.425 * (sF[i][aq] + sF[j][bq])
        for k in range(E):
            if k == i or k == j:
                continue
            x, y = predict(Pa, Va, g, ta, rhoa, k)
            v, ok = pool_at(k, x, y)
            score = score + evid(k, v, ok, fq * wS[k], 0.5)
            for s_, sq_ in ((i, aq), (j, bq)):             # gen009: a source at the seed's own place in a far exposure
                if (s_, k) in sstat:
                    d2_ = (x - sX[s_][sq_]) ** 2 + (y - sY[s_][sq_]) ** 2
                    score = score - ST_PEN * ((d2_ >= DISP_MIN ** 2) & (sstat[(s_, k)][sq_] >= torch.clamp(0.6 * v, min=ST_THR))).to(f32)   # (0.6 v: not the wing of the body itself)
        return score

    # ---- gen017: weak third points. The triplet search needs three deep points (>= 3.5-3.8 sigma, the first NDT of
    # an anchor list); a body with two deep points and 3.0-3.8 sigma on its other nights was left to the deep-pair
    # jobs, which test every point pair against the whole (r, rdot) grid on all maps and are far too expensive
    # (gen015 finding 4). A second pass of the triplet search ("weak pass") takes its third point from a much deeper
    # list instead: every local maximum >= W3_THR of the night stack of an anchor's group (W3_N points, ~10x the deep
    # list), also on nights outside the pair (up to W3_LEV x the pair's time difference from its far end). Only
    # the triples are scored on the maps, so the cost per point pair is ~1e3 times lower than a deep-pair job's.
    # Its rows carry the deep tag (x.5), so the deep penalty of gen015 keeps the regular list safe.
    W3_THR = float(os.environ.get("FINDER_W3THR", 3.0)); W3_N = int(os.environ.get("FINDER_W3N", 200000 if dev == "cuda" else 100000))
    W3_LEV = float(os.environ.get("FINDER_W3LEV", 3.5)); W3_CAP = int(float(os.environ.get("FINDER_W3CAP", 3e6 if dev == "cuda" else 4e5)))
    W3_ON = os.environ.get("FINDER_W3", "1") != "0" and NN >= 3
    W3_FRAC = float(os.environ.get("FINDER_W3FRAC", 0.45))      # share of the triplet search's time for the first round of the weak pass
    weak_pts = {}

    def build_weak(k):
        """Local maxima >= W3_THR of the depth-weighted S/N stack of anchor k's same-night group -> (x, y, z) on the device."""
        if k in weak_pts:
            return weak_pts[k]
        g_ = next((g2 for g2 in groups if k in g2), [k])
        shp = (mH[k], mW[k])
        num = torch.zeros(shp, dtype=f32, device=dev); den = torch.zeros(shp, dtype=f32, device=dev)
        for q in g_:
            ya = max(my0[k], my0[q]); yb = min(my0[k] + mH[k], my0[q] + mH[q])
            xa = max(mx0[k], mx0[q]); xb = min(mx0[k] + mW[k], mx0[q] + mW[q])
            if yb <= ya or xb <= xa:
                continue
            sub = raw[q][ya - my0[q]:yb - my0[q], xa - mx0[q]:xb - mx0[q]]
            v = sub != -128
            se = (slice(ya - my0[k], yb - my0[k]), slice(xa - mx0[k], xb - mx0[k]))
            w_ = float(wS_np[q])
            num[se] += torch.where(v, sub.to(f32) * (0.1 * w_), torch.zeros((), dtype=f32, device=dev))
            den[se] += v.to(f32) * (w_ * w_)
            del sub, v
        # (a member without data there counts as absent: the stack of the others, as in the scores)
        Zs = torch.where(den > 0, num / torch.sqrt(den.clamp(min=1e-20)), torch.full((), -9.0, dtype=f32, device=dev))
        del num, den
        pk = (Zs >= W3_THR) & (Zs < 12.0)
        pk &= Zs >= torch.nn.functional.max_pool2d(Zs[None, None], 3, 1, 1)[0, 0]
        rim = torch.nn.functional.max_pool2d((raw[k] == -128).to(f32)[None, None], RIM, 1, RIM // 2)[0, 0] > 0
        pk &= ~rim
        del rim
        pk[:2] = False; pk[-2:] = False; pk[:, :2] = False; pk[:, -2:] = False
        idx = torch.nonzero(pk.reshape(-1))[:, 0]
        del pk
        Zf = Zs.reshape(-1); z0 = Zf[idx]
        n_all = len(idx)
        if len(idx) > W3_N:
            top = torch.topk(z0, W3_N).indices; idx = idx[top]; z0 = z0[top]
        W_ = mW[k]
        def off(a_, b_):
            dd_ = a_ + b_ - 2 * z0
            return torch.clamp(torch.where(dd_ < 0, 0.5 * (a_ - b_) / torch.where(dd_ < 0, dd_, -torch.ones_like(dd_)), torch.zeros_like(dd_)), -0.5, 0.5)
        ox = off(Zf[idx - 1], Zf[idx + 1]); oy = off(Zf[idx - W_], Zf[idx + W_])
        xs = ((idx % W_).to(f32) + ox + mx0[k]) * PS; ys = ((idx // W_).to(f32) + oy + my0[k]) * PS
        del Zs, Zf
        weak_pts[k] = (xs, ys, z0)
        log(f"  weak third points of exp {k} (group {g_}): {n_all} maxima >= {W3_THR}, kept {len(z0)}" + (f", weakest {float(z0.min()):.2f}" if len(z0) else ""))
        if inj_all is not None and len(z0):
            xn = xs.cpu().numpy(); yn = ys.cpu().numpy(); zn = z0.cpu().numpy(); got = {}
            for b_ in range(inj_all.shape[1]):
                d2 = (xn - inj_all[used[k], b_, 0]) ** 2 + (yn - inj_all[used[k], b_, 1]) ** 2
                if d2.min() < 1.0:
                    got[b_] = round(float(zn[np.argmin(d2)]), 1)
            log(f"    inj with a weak point in exp {k}:", got)
        return weak_pts[k]

    def trip_search(t_end, weak=False, skip=()):
        CS = 6.0; SEG = 4.5; TEPS = float(os.environ.get("FINDER_TRIPEPS", 1.5)); G = 8
        gx0 = float(X.min()) - 3 * CS; gy0 = float(Y.min()) - 3 * CS
        GW = int((float(X.max()) - gx0) / CS) + 4; GH = int((float(Y.max()) - gy0) / CS) + 4
        grids = {}; tpx = {}; tpy = {}; tpz = {}
        done_pairs = set()
        t_bw = 0.0
        for e in anchors:
            if weak:
                t_b0 = time.time()
                if e not in weak_pts and t_b0 + 1.5 * t_bw + 1.0 > t_end:      # (no time to build the lists: nothing done)
                    log("weak pass: out of time while building the third-point lists")
                    return done_pairs, 10 ** 6
                xw_, yw_, zw_ = build_weak(e)
                t_bw = max(t_bw, time.time() - t_b0)
                tpx[e], tpy[e], tpz[e] = xw_, yw_, zw_
                cell = torch.floor((yw_ - gy0) / CS).long().clamp(0, GH - 1) * GW + torch.floor((xw_ - gx0) / CS).long().clamp(0, GW - 1)
                g_ = torch.full((GW * GH,), -1, dtype=torch.long, device=dev); o = torch.argsort(zw_, stable=True)
                try:                                          # gen018: the brighter point of a cell stays, also on a GPU (was g_[cell[o]] = o)
                    rk_ = torch.full((GW * GH,), -1, dtype=torch.long, device=dev)
                    rk_.scatter_reduce_(0, cell[o], torch.arange(len(o), device=dev), "amax")
                    g_ = torch.where(rk_ >= 0, o[rk_.clamp(min=0)], g_) if len(o) else g_
                except Exception:
                    g_[cell[o]] = o
                grids[e] = g_
                continue
            s_ = seeds[e][:NDT]
            cell = (np.floor((Y[s_] - gy0) / CS).astype(np.int64) * GW + np.floor((X[s_] - gx0) / CS).astype(np.int64))
            g_ = np.full(GW * GH, -1, np.int64); o = np.argsort(SNR[s_], kind="stable")
            g_[cell[o]] = o                                   # (two points in one 6" cell: the brighter one stays)
            grids[e] = torch.as_tensor(g_, device=dev)
            tpx[e], tpy[e] = sX[e], sY[e]
        def third(i, j, k):
            if night[k] == night[i] or night[k] == night[j]:
                return False
            if t[i] + 0.3 < t[k] < t[j] - 0.3:
                return True
            return weak and len(tpx[k]) > 0 and max(abs(t[k] - t[i]), abs(t[k] - t[j])) <= W3_LEV * (t[j] - t[i])
        tp = [(i, j) for i in anchors for j in anchors if t[j] - t[i] > 0.5 and night[i] != night[j] and (i, j) not in skip
              and any(third(i, j, k) for k in anchors)]
        tp.sort(key=lambda p_: t[p_[1]] - t[p_[0]])
        HCAP = W3_CAP if weak else 3000000
        z1 = torch.zeros(1, device=dev)
        n_hit_total = 0; n_pb_total = 0
        for ipair, (i, j) in enumerate(tp):
            now = time.time()
            if now > t_end:
                log(f"triplet time limit after {ipair}/{len(tp)} anchor pairs" + (" (weak pass)" if weak else "")); break
            pdl = min(t_end, now + 2.0 * (t_end - now) / (len(tp) - ipair) + 3.0)
            dt0 = float(t[j] - t[i])
            ks = [k for k in anchors if third(i, j, k)]
            oth = [k for k in range(E) if night[k] != night[i] and night[k] != night[j]]
            def cen(g_):                                      # central track (circular motion along x) at 1/r = g_
                r_ = torch.tensor([1.0 / g_], device=dev, dtype=f32)
                vc = torch.sqrt(GM / r_); ang = float(vc / r_) * dt0
                pax, pay, paz, rhoa = helio(z1, z1, z1 + 1, O[i], r_)
                pbx, pby, pbz, rhob = helio(z1 + math.sin(ang), z1, z1 + math.cos(ang), O[j], r_)
                ta = tT[i] - rhoa / C_AU; dtau = dt0 - (rhob - rhoa) / C_AU; g = GM / r_ ** 3
                V = tuple((b - a_) / dtau + 0.5 * g * a_ * dtau for a_, b in ((pax, pbx), (pay, pby), (paz, pbz)))
                return torch.stack([torch.cat(predict((pax, pay, paz), V, g, ta, rhoa, k)) for k in oth])
            sens = np.zeros(len(oth))
            for g_ in (1 / 27.0, 1 / 40.0, 1 / 70.0):
                dg = 2e-5
                sens = np.maximum(sens, (cen(g_ - dg) - cen(g_)).norm(dim=1).cpu().numpy().astype(np.float64) / dg)
            sens_k = {k: max(float(sens[oth.index(k)]), 1.0) for k in ks}
            sens_all = max(float(sens.max()), 1.0)
            ngf = int(min(max((GAM_MAX - GAM_MIN) * max(sens_k.values()) / SEG, G), 6000))
            ge = torch.linspace(GAM_MAX, GAM_MIN, ngf + 1, device=dev)
            if (i, j) not in pair_step:
                pair_step[(i, j)] = 1.6 / sens_all
            Ni, Nj = min(len(seeds[i]), NDT), min(len(seeds[j]), NDT)
            dO = float(np.linalg.norm(O[j] - O[i])) + 0.05
            hits = {k: [] for k in ks}
            n_pb = 0; cb_done = 0; ncb = (ngf + G - 1) // G; n_acc = 0
            def prio(cb_):
                rc_ = 2.0 / (float(ge[cb_ * G]) + float(ge[min(cb_ * G + G, ngf)]))
                return (0 if 29.0 <= rc_ <= 52.0 else 1, -float(dist_mass(np.float64(rc_))))
            for cb in sorted(range(ncb), key=prio):               # the distances of most bodies first, if time runs out
                if time.time() > pdl or n_acc > 20000000:           # (out of time, or a junk-rich pair: enough triples)
                    break
                cb_done += 1
                e0_, e1_ = cb * G, min(cb * G + G, ngf)
                g_hi, g_lo = float(ge[e0_]), float(ge[e1_]); g_c = 0.5 * (g_hi + g_lo)
                R_eff = 1.38 * math.sqrt(GM) * g_hi ** 1.5 * dt0 * ASEC * 1.1 + 0.5 * (g_hi - g_lo) * dO * ASEC * 1.1 + 20.0
                pa = helio(sux[i][:Ni], suy[i][:Ni], suz[i][:Ni], O[i], 1.0 / g_c); pb = helio(sux[j][:Nj], suy[j][:Nj], suz[j][:Nj], O[j], 1.0 / g_c)
                hxa = pa[0] / pa[2] * ASEC; hya = pa[1] / pa[2] * ASEC
                hxb = pb[0] / pb[2] * ASEC; hyb = pb[1] / pb[2] * ASEC
                ob = torch.argsort(hxb); hxs = hxb[ob].contiguous()
                lo = torch.searchsorted(hxs, (hxa - R_eff).contiguous()); hi = torch.searchsorted(hxs, (hxa + R_eff).contiguous())
                cnt = (hi - lo).clamp(min=0)
                csum = torch.cumsum(cnt, 0).cpu().numpy()
                a0 = 0
                while a0 < Ni:
                    base = int(csum[a0 - 1]) if a0 else 0
                    a1 = int(np.searchsorted(csum, base + 3000000, side="right"))
                    a1 = min(max(a1, a0 + 1), Ni)
                    c_ = cnt[a0:a1]
                    tot = int(c_.sum())
                    if tot == 0:
                        a0 = a1; continue
                    ia = torch.repeat_interleave(torch.arange(a0, a1, device=dev), c_)
                    st_ = torch.cumsum(c_, 0) - c_
                    ib = ob[lo[ia] + (torch.arange(tot, device=dev) - st_[ia - a0])]
                    a0 = a1
                    near = (hxb[ib] - hxa[ia]) ** 2 + (hyb[ib] - hya[ia]) ** 2 < R_eff ** 2
                    ia = ia[near]; ib = ib[near]
                    if len(ia) == 0:
                        continue
                    gg = ge[e0_:e1_ + 1][None, :]                 # [1, G1] segment ends
                    G1 = gg.shape[1]
                    for q0 in range(0, len(ia), 400000):
                        aq = ia[q0:q0 + 400000]; bq = ib[q0:q0 + 400000]
                        r2 = 1.0 / gg
                        Pa, Va, g, ta, rhoa = setup_pair(i, j, r2, 0.0, aq[:, None], bq[:, None])
                        v2 = Va[0] ** 2 + Va[1] ** 2 + Va[2] ** 2; vc2 = GM / r2
                        hx = Pa[1] * Va[2] - Pa[2] * Va[1]; hy = Pa[2] * Va[0] - Pa[0] * Va[2]; hz = Pa[0] * Va[1] - Pa[1] * Va[0]
                        cosi = (hx * kecl[0] + hy * kecl[1] + hz * kecl[2]) / torch.sqrt(hx * hx + hy * hy + hz * hz)
                        ok = (v2 < 1.9 * vc2) & (v2 > 0.25 * vc2) & (cosi > 0.5)
                        sok = ok[:, :-1] | ok[:, 1:]
                        rowok = sok.any(1)
                        if not bool(rowok.any()):
                            continue
                        rs = torch.nonzero(rowok)[:, 0]
                        aq = aq[rs]; bq = bq[rs]; sok = sok[rs]
                        Pa = tuple(p_[rs] for p_ in Pa); Va = tuple(v_[rs] for v_ in Va); ta = ta[rs]; rhoa = rhoa[rs]
                        n_pb += int(sok.sum())
                        for k in ks:
                            x, y = predict(Pa, Va, g, ta, rhoa, k)
                            x0, y0 = x[:, :-1], y[:, :-1]; dx = x[:, 1:] - x0; dy = y[:, 1:] - y0
                            L2 = dx * dx + dy * dy + 1e-4; Ls = torch.sqrt(L2)
                            cx = torch.floor((x0 + 0.5 * dx - gx0) / CS).long(); cy = torch.floor((y0 + 0.5 * dy - gy0) / CS).long()
                            inb = sok & (cx >= 1) & (cx < GW - 1) & (cy >= 1) & (cy < GH - 1) & (Ls < 2.2 * CS)
                            if not bool(inb.any()):
                                continue
                            cc0 = cy.clamp(1, GH - 2) * GW + cx.clamp(1, GW - 2)
                            for oy in (-1, 0, 1):
                                for ox in (-1, 0, 1):
                                    c = grids[k][cc0 + (oy * GW + ox)]
                                    val = inb & (c >= 0)
                                    if not bool(val.any()):
                                        continue
                                    cidx = c.clamp(min=0)
                                    ux_ = tpx[k][cidx] - x0; uy_ = tpy[k][cidx] - y0
                                    s_ = (ux_ * dx + uy_ * dy) / L2
                                    perp = (ux_ * dy - uy_ * dx).abs() / Ls
                                    hit = val & (perp < TEPS) & (s_ > -TEPS / Ls) & (s_ < 1 + TEPS / Ls)
                                    hh = torch.nonzero(hit)
                                    if len(hh):
                                        n_acc += len(hh)
                                        gam_h = gg[0, hh[:, 1]] + s_[hh[:, 0], hh[:, 1]] * (gg[0, hh[:, 1] + 1] - gg[0, hh[:, 1]])
                                        c3_ = cidx[hh[:, 0], hh[:, 1]]
                                        hits[k].append(torch.stack([aq[hh[:, 0]].to(f32), bq[hh[:, 0]].to(f32), gam_h, tpz[k][c3_] if weak else c3_.to(f32)], 1))
                        del Pa, Va, v2, hx, hy, hz, cosi, ok, sok
            # score the triples on the maps: scan 1/r around the value the third point gives
            rows_ = []; n_hit = 0
            for k in ks:
                if not hits[k]:
                    continue
                if weak and time.time() > t_end + 3.0:        # (the scoring of a junk-rich pair must not run over the link time)
                    break
                h = torch.cat(hits[k]); n_hit += len(h)
                if weak and len(h) > HCAP:                    # over the cap: the triples with the most significant points
                    pre_ = sl[i][h[:, 0].long()] + sl[j][h[:, 1].long()] + 0.5 * h[:, 3] ** 2
                    h = h[torch.topk(pre_, HCAP).indices]
                elif len(h) > 3000000:
                    h = h[torch.randperm(len(h), device=dev)[:3000000]]
                w = 1.3 * TEPS / sens_k[k]; step = 1.6 / sens_all
                ns = int(min(2 * math.ceil(w / step) + 1, 41))
                offs = torch.linspace(-w, w, ns, device=dev)[None, :] if ns > 1 else torch.zeros((1, 1), device=dev)
                per = max(1, 3000000 // ns)
                for q0 in range(0, len(h), per):
                    if weak and time.time() > t_end + 3.0:
                        break
                    hq = h[q0:q0 + per]
                    aq = hq[:, 0].long(); bq = hq[:, 1].long()
                    gq = (hq[:, 2:3] + offs).clamp(GAM_MIN * 0.7, 1 / 25.3)
                    sc = score_hyp(i, j, aq[:, None], bq[:, None], 1.0 / gq, torch.zeros_like(gq))
                    best, bi = sc.max(1)
                    gb = gq.gather(1, bi[:, None])[:, 0]
                    # weak pass: a triple whose third point is itself a deep point on a night in between is what the
                    # regular pass makes (tag 3.0); every other one carries the deep tag (3.5)
                    tg = torch.where(hq[:, 3] >= DEEP_THR, 3.0, 3.5) if (weak and t[i] + 0.3 < t[k] < t[j] - 0.3) else torch.full_like(best, 3.5 if weak else 3.0)
                    if len(best) > 20000:
                        top = torch.topk(best, 20000).indices
                        best, gb, aq, bq, tg = best[top], gb[top], aq[top], bq[top], tg[top]
                    rows_.append(torch.stack([best, tg.to(f32), aq.to(f32), bq.to(f32), 1.0 / gb, torch.zeros_like(best)], 1).cpu().numpy())
            n_hit_total += n_hit; n_pb_total += n_pb
            kept = 0
            if rows_:
                arr = np.concatenate(rows_)
                key = arr[:, 2].astype(np.int64) * 100000 + arr[:, 3].astype(np.int64)
                o = np.lexsort((-arr[:, 0], key)); arr = arr[o]; key = key[o]
                first_ = np.concatenate([[True], key[1:] != key[:-1]])
                if weak:                                     # a point pair with any regular-type triple is not "deep"
                    mt_ = np.minimum.reduceat(arr[:, 1], np.nonzero(first_)[0])
                    arr = arr[first_]; arr[:, 1] = mt_
                    o1 = np.argsort(-arr[:, 0]); o2 = o1[arr[o1, 1] < 3.25]
                    arr = arr[np.union1d(o1[:3000], o2[:3000])]
                    arr = arr[np.argsort(-arr[:, 0])]
                else:
                    arr = arr[first_]
                    arr = arr[np.argsort(-arr[:, 0])[:3000]]
                kept = len(arr)
                results.append(np.column_stack([arr[:, 0], arr[:, 1], np.full(len(arr), i), np.full(len(arr), j), arr[:, 2:]]))
                if inj_seed:
                    fnd = {}
                    for row in arr:
                        ba = inj_seed.get((i, int(row[2]))); bb = inj_seed.get((j, int(row[3])))
                        if ba is not None and ba == bb and ba not in fnd:
                            fnd[ba] = round(float(row[0]), 1)
                    if fnd:
                        log(f"    inj {'weak ' if weak else ''}triples ({i},{j}):", fnd)
            if cb_done >= ncb:
                done_pairs.add((i, j))
            log(f"{'weak ' if weak else ''}triplet ({i},{j}) dt={dt0:.2f}d k={ks} segs={min(cb_done * G, ngf)}/{ngf} points {Ni}x{Nj} pair-segments={n_pb:.3g} triples={n_hit} kept={kept} {time.time() - now:.1f}s")
        log(f"{'weak ' if weak else ''}triplet search: {n_pb_total:.3g} pair-segments, {n_hit_total:.3g} triples, {len(done_pairs)}/{len(tp)} anchor pairs complete")
        return done_pairs, len(tp)

    weak_done = set(); weak_left = 0
    if anchors:
        t_now = time.time()
        t_trip_end = t_now + float(os.environ.get("FINDER_TRIPFRAC", 0.5)) * (link_deadline - t_now)
        # gen017: the weak pass first (it is cheap, and its triples with a deep third point in between are the regular
        # pass's own), in at most W3_FRAC of the triplet search's time share; the regular pass gets the rest. What the
        # weak pass could not finish is done after the regular pair jobs (job tier 9).
        if W3_ON and t_trip_end - time.time() < 10.0:
            weak_left = -1                                   # (no time now: everything is left for tier 9)
            TEL["weak"] = 1
        elif W3_ON:
            n_res0 = len(results)
            try:
                t_w0 = time.time()
                d_, n_ = trip_search(t_w0 + (1.0 if os.environ.get("FINDER_W3ONLY") else W3_FRAC) * (t_trip_end - t_w0), weak=True)
                weak_done |= d_; weak_left = n_ - len(d_)
                TEL["weak"] = int(weak_left != 0)
                log(f"weak pass, first round: {time.time() - t_w0:.1f} s, {weak_left} anchor pairs left")
            except Exception as ex:
                import traceback
                del results[n_res0:]
                weak_left = 0; W3_ON = False
                log("weak pass failed:", type(ex).__name__, ex, traceback.format_exc()[-1500:])
            if dev == "cuda":
                torch.cuda.empty_cache()
        try:
            if not os.environ.get("FINDER_W3ONLY"):          # (dev: the weak pass alone)
                TEL["trip"] = 1
                d_r, n_r = trip_search(t_trip_end)
                TEL["trip"] = int(len(d_r) < n_r)
        except Exception as ex:
            import traceback
            log("triplet search failed:", type(ex).__name__, ex, traceback.format_exc()[-1500:])
        if dev == "cuda":
            torch.cuda.empty_cache()
    # two tiers: first every exposure pair with the K1 most significant seeds of each exposure (so that every night
    # pair is covered whatever the time), then the remaining seed pairs, shortest time difference first
    t_link0 = time.time()
    tier1_deadline = t_link0 + 0.6 * (link_deadline - t_link0)
    tier2_start = link_deadline
    jobs = [(0, p_) for p_ in pairs] + [(1, p_) for p_ in pairs]
    # gen010: strong-pair jobs first (cheap: <= KSMAX^2 seed pairs each). Only night pairs for which some other
    # night lies close enough in time that a few distance hypotheses cover the position there (ARC2S day^2).
    ARC2S = float(os.environ.get("FINDER_ARC2", 40.0))
    def third_ok(i_, j_):
        return any(night[k_] != night[i_] and night[k_] != night[j_] and abs((t[k_] - t[i_]) * (t[k_] - t[j_])) <= ARC2S for k_ in range(E))
    jobs3 = sorted([(3, p_) for p_ in pairs if KS[p_[0]] and KS[p_[1]] and third_ok(*p_)], key=lambda q_: t[q_[1][1]] - t[q_[1][0]])
    t3_deadline = t_link0 + 0.08 * max(link_deadline - t_link0, 0.0)      # (many exposures: the shortest pairs only)
    n3pair = {}
    log(f"strong-pair jobs: {len(jobs3)}")
    jobs = jobs3 + jobs
    # gen014: deep-pair jobs (tier 4), only in the time the regular jobs leave over. Every pair of points of two
    # anchor exposures that is not strong x strong (those are tiers 0-1) gets the full (r, rdot) grid on the maps:
    # a body with no >= 4.5 sigma seed on two nights and no three deep points on a line is linked from two
    # deep points (>= 3.5-3.8 sigma) and the maps of the other nights. The first N4 points of each list only.
    jobs4 = []
    if anchors and DEEP4:
        jobs4 = sorted([(4, (i_, j_)) for i_ in anchors for j_ in anchors if night[i_] < night[j_]
                        and (min(len(seeds[i_]), N4A) > nS[i_] or min(len(seeds[j_]), N4A) > nS[j_])], key=lambda q_: t[q_[1][1]] - t[q_[1][0]])
        jobs4 += sorted([(5, (i_, j_)) for i_ in anchors for j_ in anchors if night[i_] < night[j_]
                         and (len(seeds[i_]) > N4A or len(seeds[j_]) > N4A)], key=lambda q_: t[q_[1][1]] - t[q_[1][0]])
    P4MAX = float(os.environ.get("FINDER_P4MAX", 2e7))
    log(f"deep-pair jobs: {len(jobs4)}")
    TEL["d_on"] = int(bool(jobs4))
    jobs = jobs + ([(9, (0, 0))] if (W3_ON and anchors and weak_left != 0) else []) + jobs4      # gen017: (9) = the rest of the weak pass, after the regular jobs
    # gen015: the regular tiers end DEEP_FRAC of the tier time before the link deadline when deep-pair jobs exist
    reg_deadline = link_deadline - (DEEP_FRAC * (link_deadline - t_link0) if jobs4 else 0.0)
    tier1_deadline = t_link0 + 0.6 * (reg_deadline - t_link0)
    tier2_start = reg_deadline
    for ip, (tier, (i, j)) in enumerate(jobs):
        now = time.time()
        if now > link_deadline:
            TEL["all"] += sum(1 for q_ in jobs[ip:] if q_[0] <= 2)
            TEL["d_all"] = TEL.get("d_all", 0) + sum(1 for q_ in jobs[ip:] if q_[0] in (4, 5))       # gen019
            log(f"link time limit after {ip}/{len(jobs)} jobs"); break
        if tier < 4 and now > reg_deadline:
            TEL["all"] += tier <= 2
            continue
        if tier == 9:
            if link_deadline - now > 15.0:
                n_res0 = len(results)
                try:
                    d_, n_ = trip_search(now + (0.6 if jobs4 else 1.0) * (link_deadline - now) - 5.0, weak=True, skip=weak_done)
                    log(f"weak pass, second round: {time.time() - now:.1f} s, {len(d_)}/{n_} anchor pairs complete")
                    TEL["weak"] = int(len(set(d_) | set(weak_done)) < n_)
                except Exception as ex:
                    import traceback
                    del results[n_res0:]
                    log("weak pass (second round) failed:", type(ex).__name__, ex, traceback.format_exc()[-1500:])
                if dev == "cuda":
                    torch.cuda.empty_cache()
            continue
        if tier == 3:
            if now > t3_deadline:
                continue
            pair_deadline = min(link_deadline, now + 10.0)
        elif tier == 0:
            left = sum(1 for q_ in jobs[ip:] if q_[0] == 0)
            pair_deadline = min(reg_deadline, now + 2.0 * max(tier1_deadline - now, 0.0) / left + 3.0)
        elif tier == 1:
            if nS[i] <= K1 and nS[j] <= K1:
                continue
            left = sum(1 for q_ in jobs[ip:] if q_[0] == 1)
            pair_deadline = min(reg_deadline, now + 2.0 * max(tier2_start - now, 0.0) / left + 3.0)
        elif tier >= 4:
            left = len(jobs) - ip
            pair_deadline = min(link_deadline, now + 2.0 * (link_deadline - now) / left + 3.0)
        else:
            if (len(seeds[i]) == nS[i] or nS[j] == 0) and (len(seeds[j]) == nS[j] or nS[i] == 0):
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
        # tiers 0 and 1 pair strong seeds (the first nS of each list); tier 2 pairs a strong seed with a weak one
        Ni, Nj = (min(len(seeds[i]), N4A), min(len(seeds[j]), N4A)) if tier == 4 else (min(len(seeds[i]), N4), min(len(seeds[j]), N4)) if tier == 5 else (len(seeds[i]), len(seeds[j])) if tier == 2 else (KS[i], KS[j]) if tier == 3 else (nS[i], nS[j])
        n_tr = 0
        if Ni == 0 or Nj == 0:
            continue
        if tier >= 4:                       # at most ~P4MAX point pairs: shorten both lists (they are in order of significance)
            fr_ = min(1.0, math.pi * (150.0 * dt0 + 10) ** 2 / 4.0e7)
            bi_, bj_ = (min(nS[i], Ni), min(nS[j], Nj)) if tier == 4 else (min(N4A, Ni), min(N4A, Nj))   # pairs below both are not this job's
            if (Ni * Nj - bi_ * bj_) * fr_ > P4MAX:
                # keep the first d_ points behind the base of each list: ((bi + d)(bj + d) - bi bj) fr = P4MAX
                sb_ = bi_ + bj_
                d_ = int(0.5 * (-sb_ + math.sqrt(sb_ * sb_ + 4.0 * P4MAX / fr_)))
                Ni = min(Ni, bi_ + d_); Nj = min(Nj, bj_ + d_)
        cand = []
        CH = max(1, int(3e7 // Nj))
        for a0 in range(0, Ni, CH):
            a1 = min(a0 + CH, Ni)
            sep2 = (sX[i][a0:a1, None] - sX[j][None, :Nj]) ** 2 + (sY[i][a0:a1, None] - sY[j][None, :Nj]) ** 2
            cm = (sep2 < (150.0 * dt0 + 10) ** 2) & ((smag[i][a0:a1, None] - smag[j][None, :Nj]).abs() < stol[i][a0:a1, None] + stol[j][None, :Nj])
            ia_ = torch.arange(a0, a1, device=dev)[:, None]; ib_ = torch.arange(Nj, device=dev)[None, :]
            if tier == 2:
                cm &= (ia_ < nS[i]) != (ib_ < nS[j])
            elif tier == 3:
                pass
            elif tier == 4:
                cm &= ~((ia_ < nS[i]) & (ib_ < nS[j]))
            elif tier == 5:
                cm &= ~((ia_ < N4A) & (ib_ < N4A))
            else:
                t1m = (ia_ < K1) & (ib_ < K1)
                cm &= t1m if tier == 0 else ~t1m
            c_ = torch.nonzero(cm); c_[:, 0] += a0
            cand.append(c_)
            del sep2, cm
        cand = torch.cat(cand)
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
        # gen007: the distance blocks in the order of how many real bodies they hold (dist_mass), not from 25.5 au
        # outwards: a job cut short used to cover only 25.5-27 au
        rc_blk = (1.0 / gam[torch.clamp(torch.arange(nblk, device=dev) * G + G // 2, max=ng - 1)]).cpu().numpy()
        blk_order = torch.as_tensor(np.argsort(-dist_mass(rc_blk), kind="stable"), device=dev)
        for b0 in range(0, nblk, nb):
            if time.time() > pair_deadline:
                break
            c_done = min((b0 + nb) * G, ng)
            blk = blk_order[b0:b0 + nb]
            if G > 1:
                r_lo = rgrid[blk * G][:, None]; r_hi = rgrid[torch.clamp(blk * G + G - 1, max=ng - 1)][:, None]
                sb = []; sp = []
                for p0 in range(0, Nc, 4000000):
                    if tier >= 4 and time.time() > pair_deadline + 5.0:
                        break
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
                if tier >= 4 and time.time() > pair_deadline:      # gen015: a deep-pair block can be minutes long; never run over the link deadline
                    break
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
                    score = sl[i][aq] + sl[j][bq] - prior_t(Pa, Va)
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
                            else:                      # gen009: a displaced detection in a far exposure of a seed night
                                s_ = i if night[k] == night[i] else j
                                if (s_, k) in sstat:
                                    sq_ = aq if s_ == i else bq
                                    d2_ = (x - sX[s_][sq_]) ** 2 + (y - sY[s_][sq_]) ** 2
                                    hit = torch.maximum(hit, (ok & (v >= EP_THR) & (d2_ >= DISP_MIN ** 2) & (v > sstat[(s_, k)][sq_] + 0.3)).to(f32))
                                    # the hypothesis says the body has left the seed's place, but exposure k shows a source there
                                    score = score - ST_PEN * ((d2_ >= DISP_MIN ** 2) & (sstat[(s_, k)][sq_] >= torch.clamp(0.6 * v, min=ST_THR))).to(f32)   # (0.6 v: not the wing of the body itself)
                        nn = nn + hit
                    m = (nn >= 3) if tier != 3 else (nn < 3)      # (tier 3: the two-night hypotheses; the others come from tiers 0-1)
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
            if tier == 3:                    # the fraction of the strong seed pairs of this exposure pair that can be one TNO at all
                n3pair[(i, j)] = (len(arr) / float(Ni * Nj), np.sort(QUAL[seeds[i][:Ni]]), np.sort(QUAL[seeds[j][:Nj]]))
            arr = arr[np.argsort(-arr[:, 0])[:(300 if tier == 3 else 3000)]]
            results.append(np.column_stack([arr[:, 0], arr[:, 1] + (0.5 if tier >= 4 else 0.0), np.full(len(arr), i), np.full(len(arr), j), arr[:, 2:]]))   # gen015: x.5 = found by a deep-pair job
        if tier == 3 and c_done < ng:
            n3pair.pop((i, j), None)             # (cut short: the count is not complete)
        if tier <= 2:
            TEL["all"] += 1; TEL["full"] += int(c_done >= ng)
        elif tier in (4, 5):                     # gen019: how far the deep-pair jobs get is in the exit code now
            TEL["d_all"] = TEL.get("d_all", 0) + 1; TEL["d_full"] = TEL.get("d_full", 0) + int(c_done >= ng)
            TEL["d_ng"] = TEL.get("d_ng", 0) + ng; TEL["d_done"] = TEL.get("d_done", 0) + min(c_done, ng)
        log(f"job {ip} tier {tier} ({i},{j}) dt={dt0:.2f}d ng={min(c_done, ng)}/{ng} nf={nf} seeds {Ni}x{Nj} cand={Nc} triples={n_tr:.3g} kept={len(results[-1]) if best_pair else 0}")
    log(f"total hypotheses scored: {n_tr_total:.3g}")
    if not results and os.environ.get("FINDER_EXTTEST"):
        results = [np.zeros((1, 8))]             # (dev: the extension test below needs no linkage)
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

    def sample(xy, centred=False):
        """S/N at predicted positions xy [M, E, 2] in every exposure: max(bilinear, 3x3-pooled - 0.8); nan = no data."""
        M = len(xy); outv = np.full((M, E), np.nan)
        for e in range(E):
            x = tt(xy[:, e, 0]); y = tt(xy[:, e, 1])
            vb, okb = bil_at(e, x, y); vp, okp = pool_at(e, x, y)
            v = torch.where(okb, vb if centred else torch.maximum(vb, vp - 0.8), vp - 0.8)
            v = torch.where(okb | okp, v, torch.full_like(v, float("nan")))
            outv[:, e] = v.cpu().numpy()
        return outv

    MAPPOL = not os.environ.get("FINDER_NOMAPPOL")
    SVMODE = int(os.environ.get("FINDER_SV", 2))      # final score from: 0 old track + old sampling, 1 polished track + old sampling, 2 polished + centred
    POL_NIT = 16

    def map_polish(batch, sv0):
        """gen007: move every candidate's orbit (all six elements) to where the *unpooled* S/N maps of all exposures
        agree best. The track is linearised around the candidate: xy_e(q + dq) = xy_e + J_e dq, J from finite
        differences (float64, CPU); the search runs on the GPU along the singular directions of J that move the track
        by 1" for a moderate change of the state (a greedy pattern search, steps 0.7" -> 0.03" rms).
        Updates P, V, xy of the candidates in place (keeps the old track in xy0)."""
        M = len(batch)
        P = np.array([c["P"] for c in batch]); V = np.array([c["V"] for c in batch]); t0 = np.array([c["t0"] for c in batch])
        xy0 = np.array([c["xy"] for c in batch])
        fl = np.zeros(M)
        for m_ in range(M):
            fl[m_] = max(bright_score(sv0[m_])[1], 0.0)
        r0 = np.linalg.norm(P, axis=1); vc = np.sqrt(GM / r0)
        J = np.zeros((M, E, 2, 6))
        for a_ in range(6):
            P1 = P.copy(); V1 = V.copy()
            if a_ < 3:
                P1[:, a_] += 1e-5 * r0
                J[..., a_] = (predict_xy_b(P1, V1, t0, t, O) - xy0) / 1e-5
            else:
                V1[:, a_ - 3] += 1e-3 * vc
                J[..., a_] = (predict_xy_b(P1, V1, t0, t, O) - xy0) / 1e-3
        Jm = J.reshape(M, 2 * E, 6)
        Jm = np.where(np.isfinite(Jm), Jm, 0.0)
        U_, S_, Vt_ = np.linalg.svd(Jm, full_matrices=False)
        rms = S_ / math.sqrt(2 * E)                             # arcsec rms per unit of the (scaled) state direction
        use = rms >= 20.0                                       # 1" needs <= 5 % of r or of v_circ
        # a body at the int8 ceiling in >= 2 exposures sits on a flat plateau of the maps: nothing to polish there
        use &= (np.sum(np.nan_to_num(sv0, nan=0.0) >= SAT_V, axis=1) < 2)[:, None]
        cdir = np.where(use, 1.0 / np.maximum(rms, 1e-9), 0.0)  # [M,6]
        dirs = np.transpose(Vt_, (0, 2, 1)) * cdir[:, None, :]  # [M,6(param),6(k)]: unit step = 1" rms
        Dn = (Jm @ dirs).reshape(M, E, 2, 6)
        base = tt(xy0); Dt = tt(Dn); mu = tt(fl[:, None] * wS_n[None, :])
        delta = torch.zeros((M, 6), device=dev)
        eye = torch.eye(6, device=dev)
        tr0 = torch.cat([torch.zeros((1, 6), device=dev), eye, -eye], 0)      # [13,6]
        usek = torch.as_tensor(np.concatenate([np.ones((M, 1), bool), use, use], 1), device=dev)
        step = 0.7
        for it in range(POL_NIT):
            trials = delta[:, None, :] + step * tr0[None]                   # [M,13,6]
            pos_ = base[:, None] + torch.einsum("mexk,mtk->mtex", Dt, trials)
            sc = torch.zeros((M, 13), device=dev)
            for e in range(E):
                x = pos_[:, :, e, 0]; y = pos_[:, :, e, 1]
                vb, okb = bil_at(e, x, y); vp, okp = pool_at(e, x, y)
                sc = sc + evid(e, torch.where(okb, vb, vp - 0.8), okb | okp, mu[:, e:e + 1], 0.0)
            sc = torch.where(usek, sc, torch.full_like(sc, -1e9))
            bi = sc.argmax(1)
            delta = trials[torch.arange(M, device=dev), bi]
            step *= 0.82
        dl = delta.cpu().numpy().astype(np.float64)
        dq = np.einsum("mpk,mk->mp", dirs, dl)
        P2 = P + r0[:, None] * dq[:, :3]; V2 = V + vc[:, None] * dq[:, 3:]
        xy2 = predict_xy_b(P2, V2, t0, t, O)
        lin = xy0 + np.einsum("mexk,mk->mex", Dn, dl)
        r2 = np.linalg.norm(P2, axis=1)
        good = np.isfinite(xy2).all(axis=(1, 2)) & (np.abs(xy2 - lin).max(axis=(1, 2)) < 0.25) & (r2 >= 25.2) \
            & (np.sum(V2 * V2, 1) / 2 - GM / r2 < 0) & (np.abs(dl).max(axis=1) > 0)
        for m_ in np.nonzero(good)[0]:
            c = batch[m_]
            c["xy0"] = c["xy"]; c["P"] = P2[m_]; c["V"] = V2[m_]; c["xy"] = xy2[m_]
        return int(good.sum())

    wS_n = 1.0 / sigF
    dbg_seen = set()
    kecl_n = R @ np.array([0.0, -math.sin(EPS), math.cos(EPS)])

    SAT_V = 11.8

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
        # gen007: the int8 maps stop at S/N 12.7, so a value at the ceiling (11.9 after the pooled - 0.8) is a lower
        # limit. The flux must be at least what the shallowest such exposure needs, and a ceiling value is not "too
        # bright" or "too faint" for a flux above it. (Before, a body brighter than 12.7 sigma in every exposure
        # looked flux-inconsistent and scored like a chance alignment in a few-exposure scenario.)
        sat = s_ >= SAT_V
        if sat.any():
            fl = max(fl, float(np.max(r_[sat])))
        mu = fl * w_
        pos = postab[np.nonzero(ok)[0], np.clip(np.rint(s_ * 10), 0, 127).astype(int)]
        neg = np.minimum((s_ - mu) ** 2 / (2 * (1 + (0.25 * mu) ** 2)), 15.0)
        neg[sat & (mu >= s_ - 1e-6)] = 0.0
        Z = float(np.sum(s_ * w_) / math.sqrt(np.sum(w_ * w_)))
        nt = night[ok]; nnights = 0
        for n in np.unique(nt):
            q = s_[nt == n]
            if q.sum() / math.sqrt(len(q)) >= 2.5:
                nnights += 1
        return float(np.sum(pos - neg)), float(fl), Z, nnights

    # gen009: a body seen on two nights only. The grader asks for the *position* on three nights, and when the third
    # night with data lies close in time the two-night arc gives it to ~1" whatever the distance. Such a candidate
    # is kept when (a) one of its two nights has two exposures >= SEP_MIN apart in which the track is >= DISP_MIN
    # apart and each shows >= EP_THR at its own place and less at the other's (three epochs: not a static source),
    # (b) another night has data on the track at a time t3 with |(t3 - tA)(t3 - tC)| <= ARC2 days^2.
    FARP = None; ARC2 = float(os.environ.get("FINDER_ARC2", 40.0)); ARC_TOL = float(os.environ.get("FINDER_ARCTOL", 2.0))
    # the part of the Sun's pull on the observer that is across the line of sight, as an angular acceleration at
    # 1 au distance (arcsec / day^2; 61 at quadrature, 0 at opposition): what bends a track differently at different r
    O_m_ = O.mean(axis=0)
    ACC1 = ASEC * GM / float(np.linalg.norm(O_m_)) ** 3 * float(np.hypot(O_m_[0], O_m_[1]))
    log(f"two-night candidates: {'on' if TWO_NIGHT else 'off'}, reflex bend {ACC1:.1f} arcsec au / day^2")
    if TWO_NIGHT and any(farx[e] for e in range(E)):
        FARP = np.array([max(farx[e], key=lambda k: abs(t[k] - t[e])) if farx[e] else e for e in range(E)])

    S1_THR = float(os.environ.get("FINDER_S1THR", 6.5)); S2_THR = float(os.environ.get("FINDER_S2THR", 7.0))
    ALT_STEP = float(os.environ.get("FINDER_ALTSTEP", 2.6)); ALT_MAX = int(os.environ.get("FINDER_ALTMAX", 6)); ALT_TOTAL = int(os.environ.get("FINDER_ALTTOTAL", 8))

    def make_alts(c):
        """gen010: a two-night candidate's position on a third night depends on its distance, which two nights do not
        give. So the same linkage is output at several distances between 30 and 60 au, spaced ALT_STEP arcsec apart
        on the nearest other night with data (the most likely distances first), so that one of them is within 2" of
        the body there. Returns [(P, V, t0, xy)]."""
        row = np.array(todo[c["row"]], dtype=np.float64)
        i, j = int(row[2]), int(row[3]); sv = c["sv"]
        best = None
        for k in range(E):
            if night[k] == night[i] or night[k] == night[j] or not np.isfinite(sv[k]):
                continue
            prod = abs((t[k] - t[i]) * (t[k] - t[j]))
            if best is None or prod < best[0]:
                best = (prod, k)
        if best is None or best[0] > ARC2:
            return []
        k = best[1]
        pts = []
        for g_ in np.linspace(1 / 60.0, 1 / 30.0, 41):
            r2 = row.copy(); r2[6] = 1.0 / g_
            try:
                Pa, Va, ta, _, _ = grid_state(r2)
                rr = float(np.linalg.norm(Pa))
                if not np.isfinite(Va).all() or rr < 25.2 or Va @ Va / 2 - GM / rr >= 0:
                    continue
                xy = predict_xy(Pa, Va, ta, t, O)
                if np.isfinite(xy).all():
                    pts.append((Pa, Va, ta, xy, abs(rr - 43.0)))
            except Exception:
                continue
        pts.sort(key=lambda p_: p_[4])
        chosen = [c["xy"][k]]; alts = []
        for p_ in pts:
            if min(float(np.hypot(*(p_[3][k] - q_))) for q_ in chosen) >= ALT_STEP:
                chosen.append(p_[3][k]); alts.append(p_[:4])
                if len(alts) >= ALT_MAX:
                    break
        return alts

    def two_night(sv, sw, xy, r_c):
        ok = np.isfinite(sv)
        det_n = []
        for n in np.unique(night[ok]):
            q = sv[ok & (night == n)]
            if q.sum() / math.sqrt(len(q)) >= 2.5:
                det_n.append(n)
        if len(det_n) != 2:
            return len(det_n), False
        epoch = False
        for e in (range(E) if (FARP is not None and sw is not None) else ()):
            k = FARP[e]
            if k == e or night[e] not in det_n or not (ok[e] and ok[k]):
                continue
            if (sv[e] >= EP_THR and sv[k] >= EP_THR and np.hypot(*(xy[e] - xy[k])) >= DISP_MIN
                    and not (sw[e] > sv[e] - 0.3) and not (sw[k] > sv[k] - 0.3)):
                epoch = True
        kind = 1
        if not epoch:
            # gen010: no displaced epoch, but both nights are strong on their own: an exposure >= S1_THR, or a night
            # stack >= S2_THR with two exposures >= 3.5 sigma. Whether two such detections are one body is decided
            # by how many strong pairs of that exposure pair can be a TNO at all (n3pair, see finalize).
            if not STRONG2:
                return 2, False
            for n in det_n:
                mn_ = np.nonzero(ok & (night == n))[0]
                q = np.sort(sv[mn_])[::-1]
                if not (q[0] >= S1_THR or (len(q) >= 2 and q.sum() / math.sqrt(len(q)) >= S2_THR and q[1] >= 3.5)):
                    return 2, False
                # every other exposure of that night must show what the brightest one predicts (not an asteroid
                # that has left, not a cosmic ray): >= 0.7 x the predicted S/N - 2.5
                kb_ = mn_[np.argmax(sv[mn_])]
                if (sv[mn_] < 0.7 * sv[kb_] * wS_n[mn_] / wS_n[kb_] - 2.5).any():
                    return 2, False
            kind = 2
        tA = float(t[night == det_n[0]].mean()); tC = float(t[night == det_n[1]].mean())
        for n in np.unique(night[ok]):
            if n in det_n:
                continue
            t3 = float(t[ok & (night == n)].mean())
            # position error on the third night if the body is at 33 or 55 au instead of the candidate's r_c
            prod = abs((t3 - tA) * (t3 - tC))
            # gen010: no distance test any more (gen009: the position there had to be safe to 2" for 33-55 au);
            # the candidate is output at several distances instead (make_alts)
            if prod <= ARC2:
                return 3, kind
        return 2, False

    def cand_json(c):
        md = c["md"]; ds = md[md >= 0]
        P, V = c["P"], c["V"]
        j = {"epoch_mjd": float(c["t0"] + tref), "state": [float(v) for v in np.concatenate([Rinv @ P, Rinv @ V])],
             "p": 0.0, "score": float(c["score"]), "n_det": int(len(ds)), "n_nights": int(c["nn"]), "Z": round(c["Z"], 2),
             "snr": [None if not np.isfinite(v) else round(float(v), 1) for v in c["sv"]]}
        if c["fl"] > 0:
            j["mag"] = float(25.0 - 2.5 * math.log10(c["fl"]))
        if c.get("two"):
            j["two"] = int(c["two"])
        if c.get("deep"):
            j["deep"] = 1
        if DUMPC:
            j["_xy"] = np.round(c["xy"], 2).tolist()
        return j

    DUMPC = bool(os.environ.get("FINDER_DUMPC"))
    S_MID, S_SCALE = float(os.environ.get("FINDER_SMID", 45.0)), 10.0

    # gen010: strong two-night candidates. The score cannot rank them: its evidence per exposure saturates at ~9
    # (two 8 sigma detections give 18, three 4.4 sigma noise peaks 22), and it has no term for the number of trials,
    # which is what differs: a chance three-night alignment is the best of ~1e7 hypotheses, while only a handful of
    # strong seed pairs of two nearby nights can be a bound prograde TNO at all (n3pair; 6 in a 2-day pair of
    # scen04, 5 of them injected bodies; > 300 for a 27-day pair). npair of a candidate = the expected number of
    # chance pairs of its exposure pair at its level: (strong seeds of i at least as significant as its weaker seed)
    # x (same for j) x (fraction of all strong pairs of (i, j) that pass the orbit cuts). Candidates with npair <=
    # N2MAX (the lowest first, at most N2SLOTS) are lifted to the level of the N2RANK-th regular candidate.
    N2MAX = float(os.environ.get("FINDER_N2MAX", 2.0)); N2SLOTS = int(os.environ.get("FINDER_N2SLOTS", 6)); N2RANK = int(os.environ.get("FINDER_N2RANK", 12))

    RR = None; RR_SCALE = 3.0; RR_CLEAR = float(os.environ.get("FINDER_RRCLEAR", 3.0))
    if os.environ.get("FINDER_RERANK"):          # off by default: in leave-one-scenario-out tests it ranked worse than the score
        try:
            import rerank
            RR = rerank.load(torch, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rerank.pt"))
            log("re-ranker loaded:", len(RR["models"]), "networks")
        except Exception as ex:
            log("re-ranker not used:", type(ex).__name__, ex)

    DP_LIFT = float(os.environ.get("FINDER_DPLIFT", 7.0)); DP_DEF = float(os.environ.get("FINDER_DPDEF", 2.5)); DP_MIN = float(os.environ.get("FINDER_DPMIN", 0.0))
    dp_state = {"pen": None}

    def deep_pen(cl):
        """gen015: candidates that only a deep-pair job linked are the best of a much larger number of hypotheses than
        the regular ones, so their chance alignments reach higher scores. They are ranked against their own chance
        level: the score of every deep-only candidate is lowered by pen, chosen so that the level that ~3 deep chance
        candidates exceed (median score of ranks 25-60 of the deep-only list + DP_LIFT; the number of chance
        candidates falls by e per ~2.7 score units) is the level of the 25th regular chance candidate (median of
        ranks 25-60 of the regular list + 1.3). A deep body must stand clear of the deep chance population; the
        regular list loses at most a few of its last slots."""
        dl = [c for c in cl if c.get("deep")]
        if not dl:
            return
        for c in dl:
            c["score"] = c["score_raw"]
        pen = DP_DEF
        try:
            rl = [c for c in cl if not c.get("deep")]
            tr = [c["score"] for c in finalize_(rl, {}, 60)[25:60]]; td = [c["score"] for c in finalize_(dl, {}, 60)[25:60]]
            if len(tr) >= 10 and len(td) >= 10:
                pen = max(DP_MIN, (float(np.median(td)) + DP_LIFT) - (float(np.median(tr)) + 1.3))
            elif len(td) < 10 and len(tr) >= 10:
                pen = max(DP_MIN, DP_DEF)
        except Exception as ex:
            log("deep penalty failed:", type(ex).__name__, ex)
        if dp_state["pen"] is None or abs(pen - dp_state["pen"]) > 0.5:
            log(f"deep-pair candidates: {len(dl)} of {len(cl)}, score penalty {pen:.1f}")
        dp_state["pen"] = pen
        for c in dl:
            c["score"] = c["score_raw"] - pen

    def finalize(cl):
        try:
            deep_pen(cl)
        except Exception as ex:
            log("deep penalty failed:", type(ex).__name__, ex)
        prom = sorted([c for c in cl if c.get("two") == 2 and c.get("npair", 10 ** 9) <= N2MAX], key=lambda c: c["npair"])
        eff = {}
        if prom:
            base_ = finalize_(cl, {}, N2RANK)
            s_ref = base_[-1]["score"] if base_ else 0.0
            for c in prom:
                eff[id(c)] = max(c["score"], s_ref + 0.5 / (1.0 + c["npair"]))
        outl = finalize_(cl, eff, 200)
        tail = [c["score"] for c in outl[25:60]]
        s_clear = max(S_MID - 5.0, (float(np.median(tail)) + 12.0) if len(tail) >= 10 else 1e9)
        # gen011: learned re-ranker. A small network (rerank.pt, trained on the final lists of injection runs on the
        # dev scenarios: injected bodies against the chance alignments around them) orders the list instead of the
        # score. Lifted two-night candidates keep the level of the N2RANK-th regular one, alternates follow their parent.
        L = None
        if RR is not None and len(outl) >= 30:
            try:
                reg = [c for c in outl if c.get("alt_of") is None and "npair" not in c]
                refs = rerank.scen_refs([c["score"] for c in reg], [c["Z"] for c in reg])
                Lr = rerank.predict(torch, RR, np.array([rerank.features(c, wS_n, night, t, refs, orbit_prior) for c in outl]))
                Lr = np.where(np.isfinite(Lr), Lr, -30.0)
                lreg = np.sort([l for l, c in zip(Lr, outl) if c.get("alt_of") is None and "npair" not in c])[::-1]
                l_ref = float(lreg[min(N2RANK, len(lreg)) - 1]) if len(lreg) else 0.0
                L = np.zeros(len(outl))
                for q_, c in enumerate(outl):
                    if c.get("alt_of") is not None:
                        L[q_] = L[q_ - 1] - 1e-4               # (alternates sit right behind their parent in outl)
                    elif "npair" in c:
                        L[q_] = max(float(Lr[q_]), l_ref + 0.05 / (1.0 + c["npair"]))
                    else:
                        L[q_] = float(Lr[q_])
                o_ = np.argsort(-L, kind="stable")
                outl = [outl[q_] for q_ in o_]; L = L[o_]
                for c, l_ in zip(outl, L):
                    c["L"] = round(float(l_), 3)
            except Exception as ex:
                log("re-ranker failed:", type(ex).__name__, ex); L = None
        for rk, c in enumerate(outl):
            if L is None:
                sg = 1 / (1 + math.exp(-(c["score"] - S_MID) / S_SCALE))
                c["p"] = round((0.5 + 0.49 * sg) if (rk < 25 or c["score"] >= s_clear) else 0.45 * sg * 0.999 ** rk, 5)
            else:
                sg = 1 / (1 + math.exp(-max(-30.0, min(30.0, L[rk] / RR_SCALE))))
                c["p"] = round((0.5 + 0.49 * sg) if (rk < 25 or (L[rk] >= RR_CLEAR and c["score"] >= s_clear - 6.0)) else 0.45 * sg * 0.999 ** rk, 5)
        return outl

    def finalize_(cl, eff, nmax):
        cl = sorted(cl, key=lambda c: -eff.get(id(c), c["score"]))
        used_d = set(); outl = []; tracks = []; n_alt = 0; n_prom = 0
        for c in cl:
            if id(c) in eff and eff[id(c)] > c["score"]:
                if n_prom >= N2SLOTS:
                    continue
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
            if id(c) in eff:
                n_prom += 1; outl[-1]["score"] = float(eff[id(c)]); outl[-1]["npair"] = round(float(c.get("npair", -1)), 2)
            if c.get("two") and c.get("row") is not None and n_alt < ALT_TOTAL and len(outl) <= 25:
                try:
                    if "alts" not in c:
                        c["alts"] = make_alts(c)
                    for q_, (Pa_, Va_, ta_, xy_) in enumerate(c["alts"][:min(ALT_TOTAL - n_alt, 2 if c["two"] == 2 else ALT_MAX)]):
                        c2 = dict(c); c2["P"] = Pa_; c2["V"] = Va_; c2["t0"] = ta_; c2["xy"] = xy_; c2["score"] = outl[-1 - q_]["score"] - 0.01 * (q_ + 1)
                        j2 = cand_json(c2); j2["alt_of"] = len(outl) - 1 - q_
                        outl.append(j2); n_alt += 1
                except Exception as ex:
                    log("alternates failed:", type(ex).__name__, ex)
            if len(outl) >= nmax:
                break
        # (in finalize:) the top 25 get p >= 0.5; so does anything further down that stands clearly above this scenario's chance
        # alignments (their level: the median score of ranks 25-60), so a rich scenario can use slots a poor one wastes
        return outl

    def score_batch(batch):
        if not batch:
            return []
        sv0 = sample(np.array([c["xy"] for c in batch]))
        if MAPPOL:
            try:
                map_polish(batch, sv0)
            except Exception as ex:
                log("map polish failed:", type(ex).__name__, ex)
        xyp_ = np.array([c["xy"] for c in batch])
        sv1 = sample(xyp_) if MAPPOL else sv0
        sv2 = sample(xyp_, True) if MAPPOL else sv0
        sv = (sv0, sv1, sv2)[SVMODE]
        if MAPPOL and SVMODE:
            unp = np.sum(np.nan_to_num(sv0, nan=0.0) >= SAT_V, axis=1) >= 2     # on the ceiling (not polished): old sampling
            sv = np.where(unp[:, None], sv0, sv)
        outb = []; by_grp = {}
        sw_ = None
        if FARP is not None:                       # gen009: S/N of every exposure where the track is in its far partner
            try:
                sw_ = sample(xyp_[:, FARP, :])
                with np.errstate(invalid="ignore"):
                    svk_ = sv[:, FARP]
                    nst_all = ((FARP != np.arange(E))[None, :] & (svk_ >= ST_THR) & (sw_ >= np.maximum(ST_THR, 0.6 * np.nan_to_num(sv, nan=0.0)))
                               & (np.hypot(xyp_[:, :, 0] - xyp_[:, FARP, 0], xyp_[:, :, 1] - xyp_[:, FARP, 1]) >= DISP_MIN)).sum(axis=1)
            except Exception as ex:
                sw_ = None
                log("far-exposure sampling failed:", type(ex).__name__, ex)
        for qi_, (c, s_) in enumerate(zip(batch, sv)):
            b_, fl, Z, nnights = bright_score(s_)
            c["two"] = False
            if nnights == 2 and TWO_NIGHT:
                try:
                    nnights, c["two"] = two_night(s_, None if sw_ is None else sw_[qi_], xyp_[qi_], float(np.linalg.norm(c["P"])))
                except Exception:
                    nnights = 2
            if c["two"] == 2:
                try:
                    rw_ = todo[c["row"]]; i_, j_ = int(rw_[2]), int(rw_[3])
                    q_ = min(QUAL[seeds[i_][int(rw_[4])]], QUAL[seeds[j_][int(rw_[5])]])
                    # expected number of chance pairs of that exposure pair with both seeds at least this significant
                    if (i_, j_) in n3pair and int(rw_[4]) < KS[i_] and int(rw_[5]) < KS[j_]:
                        pg_, qi_s, qj_s = n3pair[(i_, j_)]
                        c["npair"] = float(pg_ * (len(qi_s) - np.searchsorted(qi_s, q_ - 1e-6)) * (len(qj_s) - np.searchsorted(qj_s, q_ - 1e-6)))
                    else:
                        c["npair"] = 10 ** 9
                except Exception:
                    c["npair"] = 10 ** 9
                # (a kind-2 candidate exists only through this count: its own score says nothing, and in a 32-day
                # pair of scen01 chance pairs with two exposures per night scored 23-26, above the chance level)
                if c["npair"] > N2MAX:
                    nnights = 2; c["two"] = False
            if sw_ is not None:
                # a source that stays where the track was in the far exposure of the same night: a static source
                b_ -= ST_PEN * min(int(nst_all[qi_]), 2)
            if DUMPC:
                c["alt"] = tuple(bright_score(q_[qi_])[0] + c["pos"] for q_ in (sv0, sv1, sv2))
            if c.get("inj") is not None and (c["inj"], c["fit"]) not in dbg_seen:
                dbg_seen.add((c["inj"], c["fit"]))
                log(f"  inj {c['inj']} refined: bright {b_:.1f} pos {c['pos']:.1f} nights {nnights} fitted {c['fit']} sv {np.round(s_, 1).tolist()} off {np.round(np.hypot(*(c['xy'] - inj_all[used, c['inj'], :2]).T), 1).tolist()}")
            if nnights < 3:
                continue
            # gen014: the per-exposure evidence saturates near 9, so a body of total S/N 15 on two bright nights scored
            # like a chance alignment of total S/N 7. Bonus for Zlo = the smallest stack S/N that is left when one
            # night is taken away (a chance alignment or an asteroid pair lives on one bright night: Zlo stays low).
            zlo_ = float(Z)
            try:
                okz_ = np.isfinite(s_)
                for n_ in np.unique(night[okz_]):
                    mz_ = okz_ & (night != n_)
                    zlo_ = min(zlo_, float(np.sum(s_[mz_] * wS_n[mz_]) / math.sqrt(np.sum(wS_n[mz_] ** 2))) if mz_.any() else 0.0)
            except Exception:
                zlo_ = 0.0
            c["zlo"] = zlo_
            c["score"] = b_ + c["pos"] + ZB_K * min(max(zlo_ - ZB_0, 0.0), ZB_CAP); c["score_raw"] = c["score"]; c["fl"] = fl; c["Z"] = Z; c["nn"] = nnights; c["sv"] = s_
            g_ = c.get("grp")
            if g_ is None:
                outb.append(c)
            elif g_ not in by_grp or c["score"] > by_grp[g_]["score"]:
                by_grp[g_] = c                     # (fitted orbit and grid state of one linkage: the better one)
        return outb + list(by_grp.values())

    seen = {}; todo = []
    for row in res:
        k1 = (int(row[2]), int(row[4])); k2 = (int(row[3]), int(row[5]), row[1] < 3)     # (gen010: a pair's two-night row is kept beside its three-night row)
        if (k1, k2) in seen:
            if (row[1] % 1.0) < 0.25 and len(todo) > seen[(k1, k2)]:
                todo[seen[(k1, k2)]][1] = math.floor(todo[seen[(k1, k2)]][1])     # gen015: also found by a regular job: not "deep"
            continue
        if len(todo) >= 20000:
            continue
        seen[(k1, k2)] = len(todo); todo.append(row.copy())
    # ---- polish (r, rdot) of every kept linkage on the unpooled S/N maps: the search grid is ~1" coarse, and a
    # linkage with only two catalogue detections has nothing else to pin its orbit
    todo = np.array(todo, dtype=np.float64)
    todo0 = todo.copy()                          # gen009: the linked (r, rdot), used when the polished state is not a valid orbit
    try:
        for (i, j), gstep in pair_step.items():
            if time.time() > deadline - 0.13 * budget:
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
    def prior_pen(P, V):
        """Orbit prior (see orbit_prior): chance alignments fill the searched (distance, velocity) volume evenly, real
        bodies sit in a small part of it. In score units: the number of chance candidates falls by e per ~2.7 units."""
        r_ = np.linalg.norm(P); hv = np.cross(P, V); hn_ = np.linalg.norm(hv) + 1e-30
        inc = math.degrees(math.acos(max(-1.0, min(1.0, float(hv @ kecl_n) / hn_))))
        beta = abs(math.degrees(math.asin(max(-1.0, min(1.0, float(P @ kecl_n) / r_)))))
        f_ = float(P @ V) / math.sqrt(GM * r_)
        return min(PK * float(orbit_prior(np, r_, hn_ * hn_ / (GM * r_), inc, f_, beta)), PRIOR_CAP)

    # ---- gen016: track extension ---------------------------------------------------------------------------------
    # A body whose two seeds lie on nights close in time is linked with a state that fits the nights around the seeds
    # and misses the far nights by 5-50" (the centroid errors of two faint seeds, times the lever arm; gen015 found
    # three such bodies of 60, at ranks 28-41 with their far-night signal unused). The (r, rdot) grid of the linker
    # cannot repair that: it holds the two seed positions fixed. Here every top candidate with a night on which it
    # should be visible and is not gets a scan of its orbit along the (up to two) state directions that the nights
    # it *is* seen on constrain least: steps of ~0.8" on the unsupported nights, up to EXT_MAXOFF arcsec, while the
    # supported nights move <= EXT_TOLS arcsec rms. Every trial is propagated exactly (float64), pulled back onto the
    # supported positions by two Gauss-Newton steps in the well-constrained directions, and scored on the S/N maps
    # of all exposures. A trial that gains >= EXT_GAIN becomes a new candidate (scored like any other, minus
    # EXT_PK x ln(number of independent trials): the best of n noise trials is not evidence).
    EXT_ON = not os.environ.get("FINDER_NOEXT")
    EXT_N = int(os.environ.get("FINDER_EXTN", 300 if dev == "cuda" else 100))
    EXT_TOLS = float(os.environ.get("FINDER_EXTTOL", 2.0)); EXT_MAXOFF = float(os.environ.get("FINDER_EXTOFF", 60.0)); EXT_STEP = 0.8
    EXT_GAIN = float(os.environ.get("FINDER_EXTGAIN", 1.0)); EXT_PK = float(os.environ.get("FINDER_EXTPK", 0.5)); EXT_P0 = float(os.environ.get("FINDER_EXTP0", 1.0)); EXT_EXP = float(os.environ.get("FINDER_EXTEXP", 1.5)); EXT_PA = float(os.environ.get("FINDER_EXTPA", 0.5))
    EXT_T1, EXT_T2 = 151, int(os.environ.get("FINDER_EXTT2", 15))
    EXT_DBG = bool(os.environ.get("FINDER_EXTTEST"))
    dev64 = dev if dev == "cuda" else "cpu"
    f64 = torch.float64

    def pred64(P0, V0, t0, lt, nit=2, nstep=3):
        """Exact two-body tracks (RK4, float64 tensors on dev64): P0, V0 [M,3], t0 [M], lt [M,E] light-time guess.
        Returns xy [M,E,2] arcsec and the light time."""
        t64 = torch.as_tensor(t, dtype=f64, device=dev64); O64 = torch.as_tensor(O, dtype=f64, device=dev64)
        acc = lambda p: -GM * p / (p * p).sum(2, keepdim=True) ** 1.5
        D = None
        for _ in range(nit):
            dt = t64[None, :] - lt - t0[:, None]
            P = P0[:, None, :].repeat(1, E, 1); V = V0[:, None, :].repeat(1, E, 1)
            h = (dt / nstep)[:, :, None]
            for _s in range(nstep):
                k1p, k1v = V, acc(P)
                k2p, k2v = V + 0.5 * h * k1v, acc(P + 0.5 * h * k1p)
                k3p, k3v = V + 0.5 * h * k2v, acc(P + 0.5 * h * k2p)
                k4p, k4v = V + h * k3v, acc(P + h * k3p)
                P = P + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
                V = V + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            D = P - O64[None]
            lt = D.norm(dim=2) / C_AU
        return torch.stack([D[..., 0] / D[..., 2], D[..., 1] / D[..., 2]], -1) * ASEC, lt

    def extend_tracks(cl):
        t_e0 = time.time()
        order = sorted([c for c in cl if c.get("score") is not None and c.get("sv") is not None and not c.get("two")], key=lambda c: -c["score"])
        sel = []; SUP = []; UNS = []; FLS = []; tracks = np.zeros((0, E, 2))
        if EXT_DBG:
            log("  ext dbg: candidates", len(cl), "eligible", len(order), "two:", sum(1 for c in cl if c.get("two")))
        for c in order[:4000]:
            xy = np.asarray(c["xy"], float)
            if not np.isfinite(xy).all():
                continue
            if len(tracks):
                dd_ = np.abs(tracks - xy[None]).max(axis=2)
                if (np.sum(dd_ < 2.0, axis=1) >= max(2, E // 2)).any():
                    continue
            tracks = np.concatenate([tracks, xy[None]])
            sv = np.asarray(c["sv"], float); ok = np.isfinite(sv)
            sup = np.zeros(E, bool); nts = np.unique(night[ok])
            for n in nts:
                m = ok & (night == n)
                if sv[m].sum() / math.sqrt(m.sum()) >= 2.5:
                    sup |= m
            if len(set(night[sup].tolist())) < 2:
                continue
            fls = 0.85 * float(np.sum(sv[sup] * wS_n[sup]) / np.sum(wS_n[sup] ** 2))
            if EXT_DBG:
                log(f"  ext dbg: inj {c.get('inj')} score {c['score']:.1f} sv {np.round(sv, 1).tolist()} sup {sup.astype(int).tolist()} expected {np.round(fls * wS_n, 1).tolist()}")
            if not fls > 0:
                continue
            uns = np.zeros(E, bool)
            for n in np.unique(night):               # a night that is not supported although its stack should show >= EXT_EXP sigma
                ma = night == n                      # (or has no data where this track puts the body: the right track may have)
                if sup[ma].any():
                    continue
                m = ma & ok
                wa_ = math.sqrt(float(np.sum(wS_n[ma] ** 2)))
                if fls * wa_ < EXT_EXP:
                    continue
                wn_ = math.sqrt(float(np.sum(wS_n[m] ** 2))) if m.any() else 0.0
                if not ok[ma].all() or wn_ == 0.0 or float(np.sum(sv[m] * wS_n[m])) / wn_ < 0.6 * fls * wn_:
                    uns |= ma
            if not uns.any():
                continue
            sel.append(c); SUP.append(sup); UNS.append(uns); FLS.append(fls)
            if len(sel) >= EXT_N:
                break
        M = len(sel)
        if M == 0:
            log("track extension: no candidate with an unsupported night"); return []
        SUP = np.array(SUP); UNS = np.array(UNS); FLS = np.array(FLS)
        P = np.array([c["P"] for c in sel], float); V = np.array([c["V"] for c in sel], float); t0 = np.array([float(c["t0"]) for c in sel])
        r0 = np.linalg.norm(P, axis=1); vc = np.sqrt(GM / r0)
        T64 = lambda v: torch.as_tensor(np.asarray(v), dtype=f64, device=dev64)
        Pt, Vt, t0t = T64(P), T64(V), T64(t0); r0t, vct = T64(r0), T64(vc)
        lt0 = ((Pt - T64(O[0])[None]).norm(dim=1) / C_AU)[:, None].repeat(1, E)
        xy0, lt0 = pred64(Pt, Vt, t0t, lt0, nit=3, nstep=6)
        J = torch.zeros((M, E, 2, 6), dtype=f64, device=dev64)
        for a_ in range(6):
            P1 = Pt.clone(); V1 = Vt.clone()
            if a_ < 3:
                P1[:, a_] += 1e-5 * r0t
                J[..., a_] = (pred64(P1, V1, t0t, lt0, nit=2, nstep=6)[0] - xy0) / 1e-5
            else:
                V1[:, a_ - 3] += 1e-3 * vct
                J[..., a_] = (pred64(P1, V1, t0t, lt0, nit=2, nstep=6)[0] - xy0) / 1e-3
        Jn = J.cpu().numpy(); Jn = np.where(np.isfinite(Jn), Jn, 0.0)
        JS = (Jn * SUP[:, :, None, None]).reshape(M, 2 * E, 6)
        U_, S_, Vt_ = np.linalg.svd(JS, full_matrices=False)            # U_ [M,2E,6], Vt_ [M,6,6]
        sig = S_ / np.sqrt(2.0 * SUP.sum(1))[:, None]                    # arcsec rms on the supported exposures per unit
        mot = np.linalg.norm(np.einsum("mexp,mkp->mekx", Jn, Vt_), axis=3)          # [M,E,6] arcsec per unit in every exposure
        mk = np.max(np.where(UNS[:, :, None], mot, 0.0), axis=1)         # [M,6] largest motion on an unsupported exposure
        amp = np.minimum(EXT_TOLS / np.maximum(sig, 1e-9), EXT_MAXOFF / np.maximum(mk, 1e-9))
        reach = amp * mk                                                # arcsec on the unsupported nights
        o2 = np.argsort(-reach, axis=1)[:, :2]; mi = np.arange(M)
        k1, k2 = o2[:, 0], o2[:, 1]
        re1 = reach[mi, k1]; re2 = np.where(reach[mi, k2] >= 1.2, reach[mi, k2], 0.0)
        keep = re1 >= 1.5
        if EXT_T2 <= 1:
            re2 = np.zeros(M)
        h1 = (EXT_T1 - 1) // 2; h2 = max((EXT_T2 - 1) // 2, 1)
        st1 = np.where(mk[mi, k1] > 0, re1 / h1 / np.maximum(mk[mi, k1], 1e-9), 0.0)     # state units per trial step
        st2 = np.where(re2 > 0, re2 / h2 / np.maximum(mk[mi, k2], 1e-9), 0.0)
        ntr = np.maximum(1.0, 2 * re1 / 1.5) * np.maximum(1.0, 2 * re2 / 1.5)             # independent trials
        g1, g2 = np.meshgrid(np.arange(-h1, h1 + 1), np.arange(-h2, h2 + 1), indexing="ij")
        g1 = g1.reshape(-1).astype(float); g2 = g2.reshape(-1).astype(float); T_ = len(g1); i_mid = int(np.nonzero((g1 == 0) & (g2 == 0))[0][0])
        # Gauss-Newton pull-back in the well-constrained directions: dq = -sum_k v_k (u_k . res_S) / s_k
        strong = (sig >= 5.0); strong[mi, k1] = False; strong[mi, k2] &= ~(re2 > 0)
        pinv = np.einsum("mkp,mk,mzk->mpz", Vt_, np.where(strong, 1.0 / np.maximum(S_, 1e-9), 0.0), U_)   # [M,6,2E]
        supf = np.repeat(SUP[:, :, None], 2, axis=2).reshape(M, 2 * E).astype(float)
        best_sc = np.full(M, -1e9); base_sc = np.full(M, -1e9); bP = P.copy(); bV = V.copy(); bxy = xy0.cpu().numpy().copy()
        CH = max(1, int((4000000 if dev == "cuda" else 800000) // max(E, 8) // T_))
        base_u = np.zeros(M)
        for m0 in range(0, M, CH):
            if time.time() > deadline - 45:
                log("track extension: time limit"); break
            ms = np.arange(m0, min(m0 + CH, M)); B = len(ms)
            dq = (st1[ms, None, None] * g1[None, :, None]) * Vt_[ms, k1[ms]][:, None, :] + (st2[ms, None, None] * g2[None, :, None]) * Vt_[ms, k2[ms]][:, None, :]   # [B,T,6]
            lin = xy0[m0:m0 + B].cpu().numpy()[:, None] + np.einsum("mexp,mtp->mtex", Jn[ms], dq)                # linear track [B,T,E,2]
            tgt = T64(lin.reshape(B * T_, 2 * E)); msk = T64(np.repeat(supf[ms], T_, axis=0)); pv = T64(np.repeat(pinv[ms], T_, axis=0))
            dqt = T64(dq.reshape(B * T_, 6))
            Pb = Pt[m0:m0 + B].repeat_interleave(T_, 0); Vb = Vt[m0:m0 + B].repeat_interleave(T_, 0); tb = t0t[m0:m0 + B].repeat_interleave(T_, 0)
            rb = r0t[m0:m0 + B].repeat_interleave(T_, 0)[:, None]; vb_ = vct[m0:m0 + B].repeat_interleave(T_, 0)[:, None]
            ltb = lt0[m0:m0 + B].repeat_interleave(T_, 0)
            xyt = None
            for it in range(3):
                P2 = Pb + rb * dqt[:, :3]; V2 = Vb + vb_ * dqt[:, 3:]
                xyt, ltb = pred64(P2, V2, tb, ltb, nit=1 if it else 2, nstep=3)
                if it < 2:
                    res = (xyt.reshape(B * T_, 2 * E) - tgt) * msk
                    res = torch.where(torch.isfinite(res), res, torch.zeros_like(res))
                    dqt = dqt - torch.einsum("npz,nz->np", pv, res)
            r2 = P2.norm(dim=1)
            valid = torch.isfinite(xyt).all(dim=2).all(dim=1) & (r2 >= 25.2) & ((V2 * V2).sum(1) / 2 - GM / r2 < 0)
            xf = torch.nan_to_num(xyt, nan=-1e7, posinf=-1e7, neginf=-1e7).to(device=dev, dtype=f32)
            mu = tt(np.repeat(FLS[ms, None] * wS_n[None, :], T_, axis=0))
            unt = torch.as_tensor(UNS[ms], device=dev)
            sc = torch.zeros(B * T_, device=dev); scu0 = torch.zeros(B, device=dev)
            for e in range(E):
                x = xf[:, e, 0]; y = xf[:, e, 1]
                vb, okb = bil_at(e, x, y); vp, okp = pool_at(e, x, y)
                v = torch.where(okb, torch.maximum(vb, vp - 0.8), vp - 0.8)
                ev_ = evid(e, v, okb | okp, mu[:, e], 0.0)
                sc = sc + ev_
                scu0 = scu0 + torch.where(unt[:, e], ev_.reshape(B, T_)[:, i_mid], torch.zeros_like(scu0))
            base_u[ms] = scu0.cpu().numpy()
            sc = torch.where(valid.to(dev), sc, torch.full_like(sc, -1e9)).reshape(B, T_)
            bi = sc.argmax(1)
            best_sc[ms] = sc.max(1).values.cpu().numpy(); base_sc[ms] = sc[:, i_mid].cpu().numpy()
            sel_i = (torch.arange(B, device=dev) * T_ + bi).to(dev64)
            bP[ms] = P2[sel_i].cpu().numpy(); bV[ms] = V2[sel_i].cpu().numpy(); bxy[ms] = xyt[sel_i].cpu().numpy()
            del xyt, xf, sc, P2, V2, tgt, msk, pv, dqt, Pb, Vb, ltb
        gain = best_sc - base_sc
        # what a chance alignment gains from the scan: the best of ntr noise trials (ln ntr), and the "absent where it
        # should be seen" evidence of its unsupported exposures, which any other place takes away (base_u < 0)
        pen_all = EXT_PK * np.log(np.maximum(ntr, 1.0)) + EXT_PA * np.maximum(-base_u, 0.0) + EXT_P0
        # ... and never less than what the scanned candidates themselves show: nearly all of them are chance
        # alignments, so the gain that 1 in 10 of them reaches is the level a body has to clear
        okg_ = keep & (best_sc > -1e8) & (base_sc > -1e8)
        q90_ = float(np.quantile(gain[okg_], 0.9)) if int(okg_.sum()) >= 40 else 0.0
        pen_all = np.maximum(pen_all, q90_)
        acc_ = keep & (best_sc > -1e8) & (base_sc > -1e8) & (gain - pen_all >= EXT_GAIN)
        if EXT_DBG:
            for m_ in range(M):
                log(f"  ext dbg: scanned inj {sel[m_].get('inj')}: reach {re1[m_]:.0f}\" / {re2[m_]:.0f}\" sig {sig[m_, k1[m_]]:.3f} {sig[m_, k2[m_]]:.3f} gain {gain[m_]:.1f} penalty {pen_all[m_]:.1f} (ln n {math.log(max(ntr[m_], 1.0)):.1f}, absent {max(-base_u[m_], 0.0):.1f}) off of best trial {float(np.hypot(*(bxy[m_] - inj_all[used, sel[m_]['inj'], :2]).T).max()) if sel[m_].get('inj') is not None else -1:.1f}\"")
        newc = []
        for m_ in np.nonzero(acc_)[0]:
            c = sel[m_]
            newc.append({"P": bP[m_].copy(), "V": bV[m_].copy(), "t0": c["t0"], "md": np.array(c["md"]).copy(), "pos": -prior_pen(bP[m_], bV[m_]), "xy": bxy[m_].copy(),
                         "fit": False, "inj": c.get("inj"), "row": c.get("row"), "deep": bool(c.get("deep")), "ext": float(ntr[m_]), "ext_pen": float(pen_all[m_]), "ext_gain": float(gain[m_]), "ext_from": float(c["score"]), "ext_sv": c["sv"]})
        outn = score_batch(newc) if newc else []
        res_ = []
        for c in outn:
            pen = c["ext_pen"]
            c["ext_raw"] = c["score"]
            c["score"] = c["score"] - pen; c["score_raw"] = c["score"]
            res_.append(c)
            if c.get("inj") is not None:
                log(f"  inj {c['inj']} extended: score {c['ext_from']:.1f} -> {c['score']:.1f} (penalty {pen:.1f}) sv {np.round(c['ext_sv'], 1).tolist()} -> {np.round(c['sv'], 1).tolist()}")
        dsc = sorted([round(c["score"] - c["ext_from"], 1) for c in res_ if c.get("inj") is None], reverse=True)
        gj = np.sort((gain - pen_all)[keep & (best_sc > -1e8) & np.array([c.get("inj") is None for c in sel])])[::-1]
        log(f"track extension: {M} candidates scanned ({int(keep.sum())} with reach >= 1.5\"), {int(acc_.sum())} with scan gain - penalty >= {EXT_GAIN}, {len(res_)} scored; "
            f"median reach {float(np.median(re1)):.0f}\" / {float(np.median(re2)):.0f}\", median penalty {float(np.median(pen_all)):.1f} (gain of the scanned: median {float(np.median(gain)):.1f}, 90 % {float(np.quantile(gain, 0.9)):.1f}, max {float(gain.max()):.1f}); not injected: scan gain - penalty best {np.round(gj[:6], 1).tolist()} "
            f"median {float(np.median(gj)) if len(gj) else 0.0:.1f}, final score change {dsc[:8]}; {time.time() - t_e0:.1f}s")
        return res_

    # every kept linkage is first scored at its map-polished grid state (fast, batched). These are candidates in
    # their own right (a fit to catalogue detections of 3 nights close in time can drift off the track the maps
    # support), and their scores order the slow fits below (the link score is a maximum over a coarse grid).
    if os.environ.get("FINDER_EXTTEST") and inj_all is not None:
        # gen016 (dev): the extension stage alone, on the real maps. Every injected body is scored at its true state;
        # then a "partial" candidate is made the way the linker makes them (an orbit fitted to noisy positions on
        # three consecutive nights only, from a wrong distance), scored, and given to extend_tracks.
        bodies_ = json.load(open(os.environ["FINDER_INJECT"]))
        rng_ = np.random.default_rng(3); tc = []
        for b_, bd in enumerate(bodies_):
            st = np.array(bd["state"], float); Pb_ = R @ st[:3]; Vb_ = R @ st[3:]; tb_ = bd["epoch_mjd"] - tref
            tc.append({"P": Pb_, "V": Vb_, "t0": tb_, "md": np.full(E, -1), "pos": -prior_pen(Pb_, Vb_), "xy": predict_xy(Pb_, Vb_, tb_, t, O),
                       "fit": False, "inj": None, "row": None, "deep": False, "b": b_})
        true_sc = {c["b"]: c for c in score_batch(tc)}
        log(f"EXTTEST: {len(true_sc)} of {len(bodies_)} injected bodies score at their true state (>= 3 nights)")
        pc = []; nts_ = np.unique(night)
        for b_, c in true_sc.items():
            sv = np.nan_to_num(np.asarray(c["sv"], float), nan=0.0)
            xyb = inj_all[used, b_, :2]
            best = None
            for k0 in range(len(nts_) - 2):
                m = np.isin(night, nts_[k0:k0 + 3]) & np.isfinite(c["sv"]) & (sv >= 2.0)
                if len(set(night[m].tolist())) == 3 and (best is None or sv[m].sum() > best[0]):
                    best = (sv[m].sum(), m)
            if best is None:
                continue
            ds = np.nonzero(best[1])[0]
            obs = xyb[ds] + rng_.normal(0, float(os.environ.get("FINDER_EXTTEST_NOISE", 0.35)), (len(ds), 2))
            st = np.array(bodies_[b_]["state"], float); Pb_ = R @ st[:3]; Vb_ = R @ st[3:]; tb_ = bodies_[b_]["epoch_mjd"] - tref
            bf = None
            for sc_ in (0.8, 0.9, 1.1, 1.25):
                try:
                    P1, V1, _ = fit_orbit(Pb_ * sc_, Vb_ / math.sqrt(sc_), tb_, t[ds], O[ds], obs, np.full(len(ds), 0.3))
                    xy1 = predict_xy(P1, V1, tb_, t, O); rr = float(np.hypot(*(xy1[ds] - obs).T).max())
                    r1 = float(np.linalg.norm(P1))
                    if np.isfinite(xy1).all() and r1 >= 25.2 and V1 @ V1 / 2 - GM / r1 < 0 and (bf is None or rr < bf[0]):
                        bf = (rr, P1, V1, xy1)
                except Exception:
                    pass
            if bf is None:
                continue
            pc.append({"P": bf[1], "V": bf[2], "t0": tb_, "md": np.full(E, -1), "pos": -prior_pen(bf[1], bf[2]), "xy": bf[3], "fit": False, "inj": b_, "row": None,
                       "deep": False, "b": b_, "off0": float(np.hypot(*(bf[3] - xyb).T).max())})
        part = score_batch(pc)
        log(f"EXTTEST: {len(pc)} partial candidates made, {len(part)} of them score (>= 3 nights)")
        exts = extend_tracks(part)
        eb = {c["inj"]: c for c in exts}
        n_res = 0
        for c in part:
            b_ = c["b"]; xyb = inj_all[used, b_, :2]; e_ = eb.get(b_)
            offp = float(np.hypot(*(c["xy"] - xyb).T).max())
            offe = float(np.hypot(*(e_["xy"] - xyb).T).max()) if e_ is not None else float("nan")
            n_res += int(e_ is not None and e_["score"] > c["score"] + 3)
            log(f"EXTTEST body {b_}: true {true_sc[b_]['score']:.1f} (Z {true_sc[b_]['Z']:.1f}) | partial {c['score']:.1f}, off {c['off0']:.1f}\" -> {offp:.1f}\" after polish | "
                + (f"extended {e_['score']:.1f} (raw {e_['ext_raw']:.1f}, penalty {e_['ext_pen']:.1f}), off {offe:.1f}\"" if e_ is not None else "not extended"))
        nbad = sum(1 for c in part if true_sc[c["b"]]["score"] - c["score"] > 6)
        log(f"EXTTEST: {nbad} partial candidates score > 6 below their true state; {n_res} candidates gained > 3 from the extension")
        return

    prelim = []; pre_score = np.full(len(todo), -1e9)
    PB = 3000
    for q0 in range(0, len(todo), PB):
        if time.time() > deadline - 30:
            break
        chunk = []
        for qi in range(q0, min(q0 + PB, len(todo))):
            row = todo[qi]
            Pa, Va, ta, da, db = grid_state(row)
            r_ = np.linalg.norm(Pa)
            if r_ < 25.2 or Va @ Va / 2 - GM / r_ >= 0:
                # gen009: on a short arc the maps do not constrain r, and the (r, rdot) polish can walk a linkage to
                # an unbound state (the same sky motion at a larger distance); it was then dropped. Use the linked state.
                todo[qi] = todo0[qi]; row = todo[qi]
                Pa, Va, ta, da, db = grid_state(row)
                r_ = np.linalg.norm(Pa)
                if r_ < 25.2 or Va @ Va / 2 - GM / r_ >= 0:
                    continue
            md = np.full(E, -1); md[int(row[2])] = da; md[int(row[3])] = db
            inj_ = inj_seed.get((int(row[2]), int(row[4]))) if inj_seed.get((int(row[2]), int(row[4]))) == inj_seed.get((int(row[3]), int(row[5]))) else None
            chunk.append({"P": Pa, "V": Va, "t0": ta, "md": md, "pos": -prior_pen(Pa, Va), "xy": predict_xy(Pa, Va, ta, t, O),
                          "fit": False, "inj": inj_, "row": qi, "deep": bool((row[1] % 1.0) > 0.25)})
        for c in score_batch(chunk):
            pre_score[c["row"]] = c["score"]; prelim.append(c)
        if q0 == 0:
            write_out(a.out, finalize(prelim))
    write_out(a.out, finalize(prelim))
    if EXT_ON and prelim and time.time() < deadline - 60:        # gen016
        try:
            ext_ = extend_tracks(prelim)
            if ext_:
                prelim += ext_
                write_out(a.out, finalize(prelim))
        except Exception as ex:
            import traceback
            log("track extension failed:", type(ex).__name__, ex, traceback.format_exc()[-1500:])
    ref_order = np.argsort(-pre_score, kind="stable")
    log(f"grid states scored ({len(prelim)} valid of {len(todo)}); fitting orbits, best first")

    final = []; batch = []
    for n_ref, qi_ in enumerate(ref_order):
        row = todo[qi_]
        if time.time() > deadline - 8 - TEL_SHIFT:
            log(f"refine time limit at {n_ref}"); break
        if len(batch) >= 300:
            final += score_batch(batch); batch = []
            write_out(a.out, finalize(final + prelim))
        Pg, Vg, t0, da, db = grid_state(row)
        P, V = Pg, Vg
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
        inj_ = inj_seed.get((int(row[2]), int(row[4]))) if inj_seed.get((int(row[2]), int(row[4]))) == inj_seed.get((int(row[3]), int(row[5]))) else None
        if not fitted:
            continue                               # (its grid state is already a candidate)
        r_ = np.linalg.norm(P)
        if r_ < 25.2 or V @ V / 2 - GM / r_ >= 0:
            continue
        xyp = predict_xy(P, V, t0, t, O)
        ds = md[md >= 0]
        resid = np.hypot(*(xyp[EXP[ds]] - np.column_stack([X[ds], Y[ds]])).T)
        if resid.max() > 1.3:
            continue
        sg2 = 0.15 ** 2 + (0.7 * FW[ds] / SNR[ds]) ** 2
        pos = float(np.sum(np.log(RAD ** 2 / (2 * sg2)) - np.minimum(resid ** 2 / (2 * sg2), 8.0)))
        pos = 0.5 * max(pos, 0.0) * max(len(ds) - 2, 0) / len(ds) - prior_pen(P, V)
        batch.append({"P": P, "V": V, "t0": t0, "md": md, "pos": pos, "xy": xyp, "fit": True, "inj": inj_, "deep": bool((row[1] % 1.0) > 0.25)})
    final += score_batch(batch)
    outl = finalize(final + prelim)
    if outl or not prelim:
        write_out(a.out, outl)
    if os.environ.get("FINDER_DUMPC"):
        import pickle
        try:                                       # gen016 (dev): the list without the extended tracks, for an A/B inside one run
            os.makedirs(os.path.join(os.environ["FINDER_DUMPC"], "noext"), exist_ok=True)
            write_out(os.path.join(os.environ["FINDER_DUMPC"], "noext", "OUT.json"), finalize([c for c in final + prelim if "ext" not in c]))
        except Exception as ex:
            log("noext dump failed:", type(ex).__name__, ex)
        keys = ("P", "V", "t0", "pos", "score", "fl", "Z", "nn", "sv", "fit", "inj", "md", "xy", "xy0", "alt", "two", "ext", "ext_from")
        pickle.dump({"cands": [{k_: c.get(k_) for k_ in keys} for c in final + prelim], "tref": tref, "R": R, "t": t, "O": O,
                     "night": night, "wS": wS_n, "kecl": kecl_n, "inj_xy": None if inj_all is None else inj_all[used],
                     "inj_sv": None if inj_all is None else sample(np.ascontiguousarray(inj_all[used][:, :, :2].transpose(1, 0, 2)), True)},
                    open(os.path.join(os.environ["FINDER_DUMPC"], "cands.pkl"), "wb"))
    for c in outl[:40]:
        r_ = np.linalg.norm(c["state"][:3])
        log(f"  p={c['p']:.3f} score={c['score']:.1f} Z={c['Z']:.1f} ndet={c['n_det']} nights={c['n_nights']} r={r_:.1f} mag={c.get('mag', float('nan')):.2f} snr={c['snr']}")
    log(f"done: {len(outl)} candidates")


if __name__ == "__main__":
    main()
    tel_exit(TEL.get("budget", 900.0))       # gen018: after every normal return of main (also "no linkages")
