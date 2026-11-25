"""gen 132: forced-photometry vetting of final candidates on the RAW images (difference-of-epochs test).

For every candidate the orbit gives its sky position u(t_e) at each exposure.  We measure PSF-weighted flux S/N on the
calibrated image at that position ("on"), and ALSO at the positions the body occupies on the OTHER nights, measured in
this exposure ("off": the same patch of sky when the body is not there).  A real mover is bright on-track and blank
off-track; a static source / artifact / bright-star halo is bright at the same sky position at every epoch.

Features per candidate (all in S/N units, each exposure equal-weighted):
  zon    sum of on S/N / sqrt(n)
  zlbno  zon without the brightest night
  zoff   static S/N at the track positions measured in other-night exposures
  npos   nights with night S/N > 1.5, nnight nights measured
  chi    reduced chi^2 of per-night S/N around their weighted mean (variability)

python vet.py SCEN CANDS.json OUT_FEATS.json   (standalone use)
"""
import csv, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker"))
import orbit  # noqa: E402
_SD = os.path.join(HERE, "stackd")
if _SD not in sys.path:
    sys.path.append(_SD)

R_CUT = 10        # cutout half-size (px)


def _rowdec(bt):
    """Per-row decoder for a row-tiled RICE_1 / PLIO_1 HDU (raw bintable), or None."""
    from astropy.io.fits.hdu.compressed._compression import decompress_rice_1_c, decompress_plio_1_c
    h = bt.header
    try:
        ctype = h["ZCMPTYPE"].strip()
        nx, ny = int(h["ZNAXIS1"]), int(h["ZNAXIS2"])
        if int(h.get("ZTILE1", nx)) != nx or int(h.get("ZTILE2", 1)) != 1:
            return None
        import fastfits
        rdt = fastfits._row_dtype(bt.columns)
        if rdt is None or rdt.itemsize != int(h["NAXIS1"]) or int(h["NAXIS2"]) != ny:
            return None
        tab = np.frombuffer(bt._get_raw_data(ny * rdt.itemsize, np.uint8, bt._data_offset).tobytes(), rdt)
        heap = np.asarray(bt._get_raw_data(int(h["PCOUNT"]), np.uint8, bt._data_offset + bt._theap))
        desc = tab["COMPRESSED_DATA"]
        gdesc = tab["GZIP_COMPRESSED_DATA"] if "GZIP_COMPRESSED_DATA" in rdt.names else None
        if np.any(desc[:, 0] == 0) and gdesc is None:
            return None
        if ctype == "RICE_1":
            bs = int(h.get("ZVAL1", 32)) if str(h.get("ZNAME1", "")).strip().upper() == "BLOCKSIZE" else 32
            bp = int(h.get("ZVAL2", 4)) if str(h.get("ZNAME2", "")).strip().upper() == "BYTEPIX" else 4
            dt = f"i{bp}"
            hasz = "ZSCALE" in rdt.names
            zs = tab["ZSCALE"].astype(np.float32) if hasz else None
            zz = tab["ZZERO"].astype(np.float32) if hasz else None
            zb = h.get("ZBLANK", None)

            def row(r):
                n, off = desc[r]
                if n == 0:
                    import zlib
                    gn, goff = gdesc[r]
                    raw = zlib.decompress(heap[goff:goff + gn].tobytes(), 47)
                    return np.frombuffer(raw, ">f4" if int(h.get("ZBITPIX", -32)) == -32 else ">f8")[:nx].astype(np.float32)
                a = np.frombuffer(decompress_rice_1_c(heap[off:off + n].tobytes(), bs, bp, nx), dt)[:nx]
                if hasz:
                    v = a.astype(np.float32) * zs[r] + zz[r]
                    if zb is not None:
                        v[a == int(zb)] = np.nan
                    return v
                return a.astype(np.float32)
            return nx, ny, row
        if ctype == "PLIO_1":
            def row(r):
                n, off = desc[r]
                b = heap[off:off + 2 * n].view(">i2").astype("<i2").tobytes()
                return np.frombuffer(decompress_plio_1_c(b, nx), "i4")[:nx].astype(np.float32)
            return nx, ny, row
    except Exception:
        return None
    return None


class _HDU:
    def __init__(self, hc, hb, k, section=False):
        self.hc, self.hb, self.k = hc, hb, k
        self.dec = None
        self.full = None
        self.cache = {}
        self.section = section
        try:
            if not section and hb[k].header.get("ZIMAGE", False):
                self.dec = _rowdec(hb[k])
        except Exception:
            self.dec = None

    def rows(self, r0, r1):
        if self.section:
            v = self.cache.get((r0, r1))
            if v is None:
                v = np.asarray(self.hc[self.k].section[r0:r1, :], np.float32)
                self.cache[(r0, r1)] = v
            return v
        if self.dec is None:
            if self.full is None:
                self.full = np.asarray(self.hc[self.k].data, np.float32)
            return self.full[r0:r1]
        out = []
        for r in range(r0, r1):
            v = self.cache.get(r)
            if v is None:
                v = self.dec[2](r)
                self.cache[r] = v
            out.append(v)
        return np.stack(out)


def _one_exposure(job):
    from astropy.io import fits
    from astropy.wcs import WCS
    import warnings
    warnings.filterwarnings("ignore")
    scen, r, ra, de, deadline, own = job
    out = []
    if deadline is not None and time.time() > deadline:
        return out, False
    try:
        return _one_exposure_(scen, r, ra, de, deadline, own)
    except Exception:
        return out, False


def _one_exposure_(scen, r, ra, de, deadline, own):
    from astropy.io import fits
    from astropy.wcs import WCS
    out = []
    complete = True
    if True:
        try:
            hi = fits.open(os.path.join(scen, r["image"]), memmap=False)
            hw = fits.open(os.path.join(scen, r["weight"]), memmap=False)
            hm = fits.open(os.path.join(scen, r["mask"]), memmap=False)
            bi = fits.open(os.path.join(scen, r["image"]), memmap=False, disable_image_compression=True)
            bw = fits.open(os.path.join(scen, r["weight"]), memmap=False, disable_image_compression=True)
            bm = fits.open(os.path.join(scen, r["mask"]), memmap=False, disable_image_compression=True)
        except Exception as ex:
            return out, False
        for k in range(1, len(hi)):
            if deadline is not None and time.time() > deadline:
                complete = False
                break
            try:
                h = hi[k].header
                nx, ny = int(h["NAXIS1"]), int(h["NAXIS2"])
                w = WCS(h)
                # quick footprint test with the CCD centre
                cr, cd_ = w.wcs_pix2world([[nx / 2, ny / 2]], 0)[0]
                dra = ((ra - cr + 180) % 360 - 180) * math.cos(math.radians(cd_))
                near = (np.abs(dra) < 0.2) & (np.abs(de - cd_) < 0.2)
                if not near.any():
                    continue
                near[:, own + 1] = False          # own night's track position = the on position
                if not near.any():
                    continue
                ci, pj = np.nonzero(near)
                xy = w.wcs_world2pix(np.c_[ra[ci, pj], de[ci, pj]], 0)
                x, y = xy[:, 0], xy[:, 1]
                ins = (x >= R_CUT + 1) & (x < nx - R_CUT - 2) & (y >= R_CUT + 1) & (y < ny - R_CUT - 2)
                if not ins.any():
                    continue
                ci, pj, x, y = ci[ins], pj[ins], x[ins], y[ins]
                fw = float(h.get("FWHM", 4.0) or 4.0)
                if not np.isfinite(fw) or fw <= 0.5 or fw > 20:
                    fw = 4.0
                sg = fw / 2.3548
                RC = max(R_CUT, int(math.ceil(2.6 * fw)))
                HI, HW, HM = _HDU(hi, bi, k), _HDU(hw, bw, k, section=True), _HDU(hm, bm, k)
                order = np.argsort(y)
                for q in order:
                    x0, y0 = int(round(x[q])), int(round(y[q]))
                    if x0 - RC < 0 or y0 - RC < 0 or x0 + RC + 1 > nx or y0 + RC + 1 > ny:
                        continue
                    im = HI.rows(y0 - RC, y0 + RC + 1)[:, x0 - RC:x0 + RC + 1]
                    wt = HW.rows(y0 - RC, y0 + RC + 1)[:, x0 - RC:x0 + RC + 1]
                    mk = HM.rows(y0 - RC, y0 + RC + 1)[:, x0 - RC:x0 + RC + 1]
                    yy, xx = np.mgrid[-RC:RC + 1, -RC:RC + 1]
                    dx = xx - (x[q] - x0); dy = yy - (y[q] - y0)
                    rr2 = dx * dx + dy * dy
                    P = np.exp(-0.5 * rr2 / (sg * sg)); P /= 2 * math.pi * sg * sg
                    good = (mk == 0) & (wt > 0) & np.isfinite(im)
                    ann = good & (rr2 > (2.0 * fw) ** 2)
                    if ann.sum() < 10:
                        ann = good & (rr2 > (1.5 * fw) ** 2)
                    if ann.sum() < 6:
                        continue
                    b = float(np.median(im[ann]))
                    wp = np.where(good & (rr2 <= (2.0 * fw) ** 2), wt, 0.0)
                    den = float(np.sum(wp * P * P))
                    full = float(np.sum(np.where(rr2 <= (2.0 * fw) ** 2, np.median(wt[good]) if good.any() else 0.0, 0.0) * P * P))
                    if den <= 0 or full <= 0 or den < 0.5 * full:
                        continue
                    f = float(np.sum(wp * P * (im - b))) / den
                    s = f * math.sqrt(den)
                    if not np.isfinite(s):
                        continue
                    s = max(min(s, 50.0), -50.0)
                    out.append((int(ci[q]), int(pj[q]), s, math.sqrt(den)))
            except Exception:
                continue
        for hh in (hi, hw, hm, bi, bw, bm):
            try:
                hh.close()
            except Exception:
                pass
    return out, complete


TNO_COL = {"g": 0.55, "r": 1.0, "i": 1.25, "z": 1.4, "Y": 1.4, "VR": 0.85}   # = stackd gen124 colour factors


def _gain(rs):
    """ADU per unit physical flux of a red TNO, per exposure: 10^(0.4 (zp - 30)) x colour; no zp -> exptime x the
    median ADU-per-second of the exposures that have one (x 0.7: empty-zp exposures are often bad)."""
    def fnum(x):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None
    g = np.full(len(rs), np.nan)
    et = np.array([max(fnum(r.get("exptime")) or 1.0, 1.0) for r in rs])
    for i, r in enumerate(rs):
        zp = fnum(r.get("magzero"))
        if zp is not None and 15 < zp < 40:
            g[i] = 10 ** (0.4 * (zp - 30.0))
    ok = np.isfinite(g)
    rate = float(np.median(g[ok] / et[ok])) if ok.any() else 1.0
    g[~ok] = 0.7 * rate * et[~ok]
    col = np.array([TNO_COL.get((r.get("band") or "").strip(), TNO_COL.get((r.get("band") or "").strip()[:1], 1.0))
                    for r in rs])
    return (g * col).astype(np.float32)


def _wfeat(o, w, ne, NN):
    """Matched-filter (expected-S/N weighted) versions of zon / zlbno / zn / chi.  Weights are normalised to the
    candidate's median and clipped to [0.2, 3] (robust to a bad zero point).  Under the null each is N(0, 1)."""
    ok = np.isfinite(o) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return {}
    w = np.where(ok, w, 0.0)
    w = np.clip(w / np.median(w[ok]), 0.2, 3.0) * ok
    oo = np.where(ok, o, 0.0)
    wzon = float(np.sum(w * oo) / math.sqrt(np.sum(w * w)))
    zn = np.full(NN, np.nan); wn = np.zeros(NN)
    for j in range(NN):
        m = ok & (ne == j)
        if m.any():
            zn[j] = np.sum(w[m] * oo[m]) / math.sqrt(np.sum(w[m] ** 2)); wn[j] = math.sqrt(np.sum(w[m] ** 2))
    fin = np.isfinite(zn)
    nn = int(fin.sum())
    wzlbno = 0.0
    wchi = 0.0
    if nn >= 2:
        jb = int(np.nanargmax(zn))
        m = ok & (ne != jb)
        wzlbno = float(np.sum(w[m] * oo[m]) / math.sqrt(np.sum(w[m] ** 2))) if m.any() else 0.0
        # flux consistency: per-night flux estimates zn/wn vs the weighted mean, chi^2 / dof
        fl = zn[fin] / wn[fin]; iv = wn[fin] ** 2
        mu = float(np.sum(fl * iv) / np.sum(iv))
        wchi = float(np.sum((fl - mu) ** 2 * iv) / (nn - 1))
    return {"wzon": wzon, "wzlbno": wzlbno, "wchi": wchi, "wnpos": int(np.sum(zn[fin] > 1.5)),
            "wzn": [None if not np.isfinite(x) else round(float(x), 3) for x in zn]}


def night_of(t):
    return np.floor(np.asarray(t) - 0.5).astype(int)


def measure(scen, cands, deadline=None, max_exp=60, log=print, workers=4):
    """Return list of feature dicts (one per candidate, None if not measured)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    import warnings
    warnings.filterwarnings("ignore")
    t_start = time.time()
    rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
    t = np.array([float(r["mjd_mid"]) for r in rows])
    obs = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows])
    nt = night_of(t)
    C = len(cands)
    if C == 0:
        return []
    # exposure subset: at most max_exp, spread over nights (round robin)
    un = np.unique(nt)
    if deadline is not None:
        est = 0.3 + 0.005 * C * (1 + min(len(un), max_exp))       # s per exposure per worker (Mac measurement)
        nfit = int(0.7 * max(0.0, deadline - time.time()) * max(1, workers) / est)
        if nfit < max_exp:
            log(f"vet: time for ~{nfit} exposures (cap {max_exp})")
            max_exp = max(2, nfit)
    per = {n: list(np.flatnonzero(nt == n)) for n in un}
    sel = []
    while len(sel) < min(max_exp, len(rows)):
        added = False
        for n in un:
            if per[n]:
                sel.append(per[n].pop(len(per[n]) // 2)); added = True
                if len(sel) >= max_exp:
                    break
        if not added:
            break
    E = len(sel)   # round-robin order over nights: a deadline cut loses exposures evenly
    st = np.array([c["state"] for c in cands], float)
    ep = np.array([float(c["epoch_mjd"]) for c in cands])
    # predicted directions (C, N_all, 3)
    U = np.zeros((C, len(rows), 3))
    for k in range(C):
        U[k] = orbit.predict_dirs(st[k], ep[k], t, obs)
    # representative exposure per night (for the off-track positions): the first selected one of each night
    rep = {}
    for e in sel:
        rep.setdefault(nt[e], e)
    nights = sorted(rep)
    NN = len(nights)
    nidx = {n: i for i, n in enumerate(nights)}
    # on[c, e]: S/N ; off[c, e, j]: S/N at the track position of night j measured in exposure e
    on = np.full((C, E), np.nan, np.float32)
    off = np.full((C, E, NN), np.nan, np.float32)
    jobs = []
    for ie, e in enumerate(sel):
        pos = np.concatenate([U[:, e][:, None, :], U[:, [rep[n] for n in nights]]], 1)   # (C, 1+NN, 3)
        ra = np.degrees(np.arctan2(pos[..., 1], pos[..., 0])) % 360
        de = np.degrees(np.arcsin(np.clip(pos[..., 2], -1, 1)))
        jobs.append((scen, rows[e], ra, de, deadline, nidx[nt[e]]))
    res = None
    if workers > 1 and E > 1:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
            with ProcessPoolExecutor(max_workers=min(workers, E), mp_context=ctx) as ex:
                res = list(ex.map(_one_exposure, jobs))
        except Exception as ex_:
            log("vet: pool failed, serial", repr(ex_)[:200])
            res = None
    if res is None:
        res = [_one_exposure(j) for j in jobs]
    ncomp = sum(1 for rr in res if rr[1])
    sqd = np.full((C, E), np.nan, np.float32)    # gen 134: sqrt(den) of the on measurement (weight-map depth)
    for ie, (rr, _ok) in enumerate(res):
        for c_, p_, s, sq in rr:
            if p_ == 0:
                on[c_, ie] = s
                sqd[c_, ie] = sq
            else:
                off[c_, ie, p_ - 1] = s
    # off positions closer than 3" to the on position (slow mover / close nights) still hold the body's own flux
    for ie, e in enumerate(sel):
        for j, n in enumerate(nights):
            d = np.einsum("ci,ci->c", U[:, e], U[:, rep[n]])
            off[d > math.cos(3.0 * orbit.ARC), ie, j] = np.nan
    # per-exposure null calibration from the off-track measurements (the raw weight-map S/N is ~2.6x too wide)
    allv = off[np.isfinite(off)]
    gmed = float(np.median(allv)) if allv.size >= 30 else 0.0
    gsd = float(1.4826 * np.median(np.abs(allv - gmed))) if allv.size >= 30 else 1.0
    if not np.isfinite(gsd) or gsd <= 0:
        gsd = 1.0
    for ie in range(E):
        v = off[:, ie, :]
        v = v[np.isfinite(v)]
        if v.size >= 40:
            md = float(np.median(v)); sd = float(1.4826 * np.median(np.abs(v - md)))
            if not (np.isfinite(sd) and sd > 0):
                md, sd = gmed, gsd
        else:
            md, sd = gmed, gsd
        on[:, ie] = (on[:, ie] - md) / sd
        off[:, ie, :] = (off[:, ie, :] - md) / sd
        sqd[:, ie] = sqd[:, ie] / sd
    # gen 134: expected S/N per unit physical flux of a red TNO in each measurement (matched-filter weights):
    # flux in ADU = F * 10^(0.4 (zp - 30)) * colour(band); its normalised S/N = ADU flux * sqrt(den) / sd.
    gz = _gain([rows[e] for e in sel])
    wgt = sqd * gz[None, :]
    log(f"vet: null median {gmed:.2f} sd {gsd:.2f} over {allv.size} off-track samples")
    # features
    ne = np.array([nidx[nt[e]] for e in sel])
    feats = []
    if ncomp < E:
        log(f"vet: only {ncomp} of {E} exposures complete; features dropped")
        return [None] * C
    for c in range(C):
        o = on[c]
        ok = np.isfinite(o)
        if ok.sum() < 2:
            feats.append(None)
            continue
        zon = float(np.nansum(o) / math.sqrt(ok.sum()))
        zn = np.full(NN, np.nan)
        for j in range(NN):
            m = ok & (ne == j)
            if m.any():
                zn[j] = o[m].sum() / math.sqrt(m.sum())
        fin = np.isfinite(zn)
        nn = int(fin.sum())
        if nn >= 2:
            jb = int(np.nanargmax(zn))
            m = ok & (ne != jb)
            zlbno = float(o[m].sum() / math.sqrt(m.sum())) if m.any() else 0.0
        else:
            zlbno = 0.0
        npos = int(np.sum(zn[fin] > 1.5))
        # static: for night j's track position, exposures from other nights
        zs = []
        for j in range(NN):
            v = off[c, :, j]
            m = np.isfinite(v) & (ne != j)
            # require the off position to be >= 3 arcsec from the on position (moving body has left)
            if m.sum() >= 1:
                zs.append(v[m].sum() / math.sqrt(m.sum()))
        zoff = float(np.sum(zs) / math.sqrt(len(zs))) if zs else 0.0
        zoffmax = float(np.max(zs)) if zs else 0.0
        if nn >= 2:
            zz = zn[fin]
            mu = zz.mean()
            chi = float(np.sum((zz - mu) ** 2) / (nn - 1))
        else:
            chi = 0.0
        fd = {"zon": zon, "zlbno": zlbno, "zoff": zoff, "zoffmax": zoffmax, "npos": npos, "nnight": nn,
              "chi": chi, "nexp": int(ok.sum()),
              "zn": [None if not np.isfinite(x) else round(float(x), 3) for x in zn]}
        try:
            fd.update(_wfeat(o, wgt[c], ne, NN))
        except Exception:
            pass
        feats.append(fd)
    log(f"vet: {C} candidates x {E} exposures in {time.time() - t_start:.1f}s; measured {sum(f is not None for f in feats)}")
    return feats


if __name__ == "__main__":
    scen, cj, oj = sys.argv[1:4]
    cs = json.load(open(cj))["candidates"]
    fs = measure(scen, cs)
    json.dump({"feats": fs}, open(oj, "w"))
