"""Orbit-grid synthetic tracking (KBMOD-style shift-and-stack) for TNOs.  See NOTES.md.

1. detection workers (detect.py, gen 1-7 code) produce per-CCD matched-filter S/N maps (2x decimated);
2. all maps are resampled onto one 1"/px gnomonic mosaic per exposure (GPU), static sources masked, values clipped;
3. coarse search: max-pooled (c"/px) maps; every (distance, sky velocity) hypothesis on a grid and every coarse
   start pixel is scored by summing the pooled S/N along the exact parallax+linear-motion track;
4. the best coarse tracks are refined on the 1" mosaics (local grid search), fitted with a two-body orbit, output.
"""
import argparse, csv, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orbit import T0, log, left, GM, ARC, radec_to_unit, tangent_frame, predict_dirs, fit_orbit

PS = 1.0 * ARC


def n_workers():
    if os.environ.get("TNO_WORKERS"):
        return max(1, int(os.environ["TNO_WORKERS"]))
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = os.cpu_count() or 4
    try:
        import torch
        if torch.cuda.is_available():
            return int(max(2, min(7, ncpu - 1)))
    except Exception:
        pass
    return int(max(1, min(4, ncpu)))


def pick_exposures(rows, cap):
    t = np.array([float(r["mjd_mid"]) for r in rows])
    if len(rows) <= cap:
        return list(np.argsort(t))
    night = np.floor(t - 0.5).astype(int)
    un = np.unique(night)
    per = [sorted(np.flatnonzero(night == u), key=lambda i: t[i]) for u in un]
    # spread within each night: take evenly spaced members, round robin over nights
    sel = []
    k = 0
    order = [np.array(p)[np.argsort(np.abs(np.arange(len(p)) - (len(p) - 1) / 2))[::-1]] for p in per]
    # alternate ends first, then middle
    while len(sel) < cap:
        added = False
        for o in order:
            if k < len(o):
                sel.append(int(o[k])); added = True
                if len(sel) >= cap:
                    break
        if not added:
            break
        k += 1
    return sorted(sel, key=lambda i: t[i])


def run_detection(tasks, deadline):
    import detect
    from multiprocessing import get_context
    nw = n_workers()
    log("detection workers", nw, "exposures", len(tasks))
    pool = get_context("spawn").Pool(nw)
    res = {}
    try:
        ars = [(tk[0], pool.apply_async(detect.detect_exposure, (tk,))) for tk in tasks]
        for ei, ar in ars:
            rem = deadline - (time.time() - T0)
            if rem <= 0:
                break
            try:
                r = ar.get(timeout=rem)
                if r["maps"]:
                    res[ei] = r["maps"]
            except Exception as ex:
                log("exp", ei, "failed", ex)
    finally:
        pool.terminate()
    return res


def dev_of(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_mosaic(torch, dev, maps, F, lim):
    x0, y0, W, H = lim
    Ft = torch.tensor(F.T, dtype=torch.float32, device=dev)
    M = torch.full((H, W), float("nan"), dtype=torch.float16, device=dev)
    for g, m in maps:
        cu = radec_to_unit(g["rc"], g["dc"])
        c = F @ cu
        cx, cy = c[0] / c[2] / PS - x0, c[1] / c[2] / PS - y0
        r = g["rad"] * 1.05 / PS
        ia, ib = int(max(0, math.floor(cx - r))), int(min(W, math.ceil(cx + r)))
        ja, jb = int(max(0, math.floor(cy - r))), int(min(H, math.ceil(cy + r)))
        if ib <= ia or jb <= ja:
            continue
        xs = (torch.arange(ia, ib, device=dev, dtype=torch.float32) + x0) * PS
        ys = (torch.arange(ja, jb, device=dev, dtype=torch.float32) + y0) * PS
        TY, TX = torch.meshgrid(ys, xs, indexing="ij")
        v = TX[..., None] * Ft[:, 0] + TY[..., None] * Ft[:, 1] + Ft[:, 2]
        rc, dc = g["rc"], g["dc"]
        ex = torch.tensor([-math.sin(rc), math.cos(rc), 0.0], device=dev)
        ey = torch.tensor([-math.sin(dc) * math.cos(rc), -math.sin(dc) * math.sin(rc), math.cos(dc)], device=dev)
        cz = torch.tensor(cu, dtype=torch.float32, device=dev)
        den = v @ cz
        xi = (v @ ex) / den * 1e3
        eta = (v @ ey) / den * 1e3
        cxp, cyp = [torch.tensor(q, dtype=torch.float32, device=dev) for q in (g["cx"], g["cy"])]
        basis = (torch.ones_like(xi), xi, eta, xi * xi, xi * eta, eta * eta)
        px = sum(b * cxp[k] for k, b in enumerate(basis)) / 2.0
        py = sum(b * cyp[k] for k, b in enumerate(basis)) / 2.0
        mh, mw = m.shape
        inside = (px >= 0) & (px <= mw - 1) & (py >= 0) & (py <= mh - 1)
        if not bool(inside.any()):
            continue
        mt = torch.from_numpy(np.nan_to_num(m.astype(np.float32), nan=0.0)).to(dev)[None, None]
        vt = torch.from_numpy(np.isfinite(m).astype(np.float32)).to(dev)[None, None]
        grid = torch.stack([(2 * px + 1) / mw - 1, (2 * py + 1) / mh - 1], -1)[None]
        s = torch.nn.functional.grid_sample(mt, grid, mode="bilinear", align_corners=False)[0, 0]
        val = torch.nn.functional.grid_sample(vt, grid, mode="bilinear", align_corners=False)[0, 0]
        ok = inside & (val > 0.99)
        sub = M[ja:jb, ia:ib]
        M[ja:jb, ia:ib] = torch.where(ok, s.to(torch.float16), sub)
    return M


def write_out(path, cands):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"candidates": cands}, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=300)
    a = ap.parse_args()
    budget = a.budget_s
    write_out(a.out, [])
    sdir = a.scenario
    with open(os.path.join(sdir, "exposures.csv")) as f:
        rows = list(csv.DictReader(f))
    import torch
    dev = dev_of(torch)
    cuda = dev == "cuda"
    cap = 36 if cuda else 24
    sel = pick_exposures(rows, cap)
    tasks = [(int(i), os.path.join(sdir, rows[i]["image"]), os.path.join(sdir, rows[i]["mask"]), 1e9, True)
             for i in sel]
    maps = run_detection(tasks, budget * 0.4)
    exps = sorted(maps.keys(), key=lambda i: float(rows[i]["mjd_mid"]))
    log("maps for", len(exps), "exposures")
    if len(exps) < 3:
        return
    t = np.array([float(rows[i]["mjd_mid"]) for i in exps])
    obs = np.array([[float(rows[i][k]) for k in ("obs_x", "obs_y", "obs_z")] for i in exps])
    night = np.floor(t - 0.5).astype(int)
    if len(np.unique(night)) < 3:
        return
    # field frame
    cus = np.array([radec_to_unit(g["rc"], g["dc"]) for e in exps for g, m in maps[e]])
    cm = cus.mean(0); cm /= np.linalg.norm(cm)
    ra0, de0 = math.atan2(cm[1], cm[0]), math.asin(cm[2])
    F = tangent_frame(ra0, de0)
    rads = np.array([g["rad"] for e in exps for g, m in maps[e]])
    c3 = cus @ F.T
    th = c3[:, :2] / c3[:, 2:3]
    lo = (th - rads[:, None]).min(0); hi = (th + rads[:, None]).max(0)
    x0, y0 = int(math.floor(lo[0] / PS)), int(math.floor(lo[1] / PS))
    W, H = int(math.ceil(hi[0] / PS)) - x0, int(math.ceil(hi[1] / PS)) - y0
    lim = (x0, y0, W, H)
    log("mosaic", W, H, "dev", dev)
    Ne = len(exps)
    mos = []
    cnt = torch.zeros((H, W), dtype=torch.int16, device=dev)
    for e in exps:
        M = build_mosaic(torch, dev, maps[e], F, lim)
        cnt += (M > 2.5).to(torch.int16)
        mos.append(M)
        maps[e] = None
    # static sources: bright in >= 2 exposures (and >= 25% of them) at the same place
    nst = max(2, int(math.ceil(0.25 * Ne)))
    static = (cnt >= nst)[None, None].float()
    static = torch.nn.functional.max_pool2d(static, 5, 1, 2)[0, 0] > 0
    del cnt
    for k in range(Ne):
        M = mos[k]
        M = torch.where(static, torch.full_like(M, float("nan")), M)
        mos[k] = torch.clamp(M, -4, 5)
    log("mosaics done, static frac %.3f" % float(static.float().mean()))
    del static

    t_ref = float(0.5 * (t.min() + t.max()))
    dt = t - t_ref
    tmax = float(np.abs(dt).max())
    obs_ref = obs.mean(0)
    # nonlinear observer motion (perpendicular to line of sight) -> distance grid spacing
    A = np.c_[np.ones_like(dt), dt]
    coef, *_ = np.linalg.lstsq(A, obs, rcond=None)
    res = obs - A @ coef
    zc = F[2]
    resp = res - np.outer(res @ zc, zc)
    s_nl = max(float(np.linalg.norm(resp, axis=1).max()), 1e-4)

    # time-adaptive coarse pixel size
    remaining = budget - (time.time() - T0)
    search_s = remaining * 0.45
    Npix_full = W * H
    rate = 2e10 if cuda else 1e9   # gathers per second (rough)
    def cost(c):
        dg = c * PS / s_nl
        ng = max(1, (1 / 28 - 1 / 90) / dg)
        nv_tot = 0
        for gam in np.arange(1 / 90, 1 / 28, dg):
            D = 1 / gam
            wmax = 1.25 * math.sqrt(GM / D) / D
            dw = c * PS / tmax
            nv_tot += math.pi * (wmax / dw) ** 2
        return nv_tot * Npix_full / c ** 2 * Ne
    c = 4
    while cost(c) / rate > search_s and c < 40:
        c += 2
    log("coarse pixel", c, "est gathers %.2e" % cost(c), "s_nl %.4f au" % s_nl, "tmax", tmax)

    # coarse maps: max-pool, standardise
    Wc, Hc = W // c, H // c
    coarse = torch.zeros((Ne, Hc * Wc), dtype=torch.float16, device=dev)
    for k in range(Ne):
        M = mos[k][: Hc * c, : Wc * c].float()
        valid = torch.isfinite(M)
        Mp = torch.nn.functional.max_pool2d(torch.where(valid, M, torch.full_like(M, -1e4))[None, None], c, c)[0, 0]
        vf = torch.nn.functional.avg_pool2d(valid.float()[None, None], c, c)[0, 0]
        ok = vf > 0.5
        samp = Mp[ok]
        if samp.numel() < 100:
            coarse[k] = float("nan"); continue
        samp = samp[torch.randint(0, samp.numel(), (min(samp.numel(), 200000),), device=dev)]
        med = samp.median(); mad = (samp - med).abs().median() * 1.4826
        Z = torch.clamp((Mp - med) / torch.clamp(mad, min=1e-3), -3, 4)
        Z = torch.where(ok, Z, torch.full_like(Z, float("nan")))
        coarse[k] = Z.flatten().to(torch.float16)
    coarse_valid = torch.isfinite(coarse)
    coarse = torch.nan_to_num(coarse, nan=0.0)
    cv = torch.cat([coarse, torch.zeros((Ne, 1), dtype=coarse.dtype, device=dev)], 1)   # sentinel at end
    cflat = cv.flatten()
    vflat = torch.cat([coarse_valid, torch.zeros((Ne, 1), dtype=torch.bool, device=dev)], 1).flatten().to(torch.float16)

    # pixel unit vectors (coarse centres)
    gx = (torch.arange(Wc, device=dev, dtype=torch.float32) * c + c / 2 + x0) * PS
    gy = (torch.arange(Hc, device=dev, dtype=torch.float32) * c + c / 2 + y0) * PS
    GY, GX = torch.meshgrid(gy, gx, indexing="ij")
    GX, GY = GX.flatten(), GY.flatten()
    nrm = torch.sqrt(1 + GX * GX + GY * GY)
    # unit vector in field frame coordinates: (GX, GY, 1)/nrm
    ux, uy, uz = GX / nrm, GY / nrm, 1 / nrm
    Np = GX.numel()
    # observer offsets in field frame
    Ff = F
    dob = (obs_ref[None] - obs) @ Ff.T           # (Ne,3)
    hyps = []
    dg = c * PS / s_nl
    dw = c * PS / tmax
    for gam in np.arange(1 / 90 + dg / 2, 1 / 28, dg):
        D = 1 / gam
        wmax = 1.25 * math.sqrt(GM / D) / D
        n = int(math.ceil(wmax / dw))
        w = np.arange(-n, n + 1) * dw
        WX, WY = np.meshgrid(w, w)
        k = WX ** 2 + WY ** 2 <= wmax ** 2
        for wx, wy in zip(WX[k], WY[k]):
            hyps.append((D, wx, wy))
    hyps = np.array(hyps)
    Nh = len(hyps)
    log("hypotheses", Nh, "coarse pix", Np)

    def track_consts(hy):
        """hy (n,3) -> C (n, Ne, 3) field-frame offsets so that P_i = D*u + C_i."""
        D = hy[:, 0:1]
        acc = -GM / (D * D)                       # radial accel magnitude, direction ~ -(obs_ref + D z) ~ along body
        # approximate acceleration direction: away from Sun position; heliocentric body ~ obs_ref + D*z
        bz = (obs_ref @ Ff.T)[None] + np.c_[np.zeros_like(D), np.zeros_like(D), D]
        bh = bz / np.linalg.norm(bz, axis=1, keepdims=True)
        V = np.c_[hy[:, 1] * D[:, 0], hy[:, 2] * D[:, 0], np.zeros(len(hy))]
        C = dob[None] + V[:, None, :] * dt[None, :, None] + 0.5 * (acc * bh)[:, None, :] * (dt ** 2)[None, :, None]
        return C

    nmin = max(3, int(math.ceil(0.6 * Ne)))
    best = torch.full((Np,), -1e9, device=dev)
    besth = torch.zeros((Np,), dtype=torch.long, device=dev)
    B = max(1, int(2.5e7 // Np))
    tdead = time.time() + search_s
    hord = np.random.default_rng(0).permutation(Nh)    # random order: an interrupted search still covers all D
    done_h = 0
    for b0 in range(0, Nh, B):
        if time.time() > tdead:
            log("coarse search interrupted at", done_h, "of", Nh)
            break
        idx = hord[b0:b0 + B]
        hy = hyps[idx]
        C = torch.tensor(track_consts(hy), dtype=torch.float32, device=dev)
        Dt = torch.tensor(hy[:, 0], dtype=torch.float32, device=dev)[:, None]
        S = torch.zeros((len(idx), Np), device=dev)
        Nv = torch.zeros((len(idx), Np), device=dev)
        for i in range(Ne):
            px = Dt * ux[None] + C[:, i, 0:1]
            py = Dt * uy[None] + C[:, i, 1:2]
            pz = Dt * uz[None] + C[:, i, 2:3]
            ix = torch.floor((px / pz / PS - x0) / c).long()
            iy = torch.floor((py / pz / PS - y0) / c).long()
            off = (ix < 0) | (ix >= Wc) | (iy < 0) | (iy >= Hc)
            k = torch.where(off, torch.full_like(ix, Hc * Wc), iy * Wc + ix) + i * (Hc * Wc + 1)
            S += cflat[k].float()
            Nv += vflat[k].float()
        sc = torch.where(Nv >= nmin, S / torch.sqrt(torch.clamp(Nv, min=1)), torch.full_like(S, -1e9))
        m, am = sc.max(0)
        upd = m > best
        best = torch.where(upd, m, best)
        besth = torch.where(upd, torch.as_tensor(idx, device=dev)[am], besth)
        done_h += len(idx)
    log("coarse search done", done_h, "hyps")

    # candidate seeds: local maxima of best map
    bm = best.view(Hc, Wc)
    loc = (bm == torch.nn.functional.max_pool2d(bm[None, None], 5, 1, 2)[0, 0]) & (bm > -1e8)
    vals = bm[loc]
    pix = torch.nonzero(loc.flatten())[:, 0]
    K = 3000 if cuda else 1500
    o = torch.argsort(vals, descending=True)[:K]
    pix = pix[o].cpu().numpy(); seedsc = vals[o].cpu().numpy()
    seedh = besth[torch.as_tensor(pix, device=dev)].cpu().numpy()
    log("seeds", len(pix), "coarse score range %.2f..%.2f" % (seedsc.min(), seedsc.max()))

    # ------------------------------------------------------------------ fine refinement on 1" mosaics
    fine = []
    for k in range(Ne):
        M = mos[k].float()
        samp = M[torch.isfinite(M)]
        samp = samp[torch.randint(0, samp.numel(), (min(samp.numel(), 300000),), device=dev)]
        med = samp.median(); mad = (samp - med).abs().median() * 1.4826
        fine.append(((M - med) / torch.clamp(mad, min=1e-3)).to(torch.float16))
        mos[k] = None
    FM = torch.stack(fine)          # (Ne, H, W)
    del fine
    FMf = FM.flatten()

    def score_tracks(th0x, th0y, D, wx, wy, need_each=False):
        """Exact-geometry score of many tracks (torch float32 1-D tensors). Returns score, nvalid, per-exp values."""
        nrm = torch.sqrt(1 + th0x ** 2 + th0y ** 2)
        hy = np.stack([D.cpu().numpy(), wx.cpu().numpy(), wy.cpu().numpy()], 1).astype(np.float64)
        C = torch.tensor(track_consts(hy), dtype=torch.float32, device=dev)
        S = torch.zeros_like(th0x); Nv = torch.zeros_like(th0x)
        vals = []
        for i in range(Ne):
            px = D * th0x / nrm + C[:, i, 0]
            py = D * th0y / nrm + C[:, i, 1]
            pz = D / nrm + C[:, i, 2]
            fx = px / pz / PS - x0; fy = py / pz / PS - y0
            off = (fx < 0) | (fx > W - 2) | (fy < 0) | (fy > H - 2)
            fx = torch.where(off, torch.zeros_like(fx), fx); fy = torch.where(off, torch.zeros_like(fy), fy)
            ixf = torch.floor(fx); iyf = torch.floor(fy)
            ax = fx - ixf; ay = fy - iyf
            kk = iyf.long() * W + ixf.long() + i * (H * W)
            v = ((1 - ax) * (1 - ay) * FMf[kk].float() + ax * (1 - ay) * FMf[kk + 1].float()
                 + (1 - ax) * ay * FMf[kk + W].float() + ax * ay * FMf[kk + W + 1].float())
            good = ~off & torch.isfinite(v)
            v = torch.where(good, torch.clamp(v, -3, 4.5), torch.zeros_like(v))
            S += v; Nv += good.float()
            if need_each:
                vals.append(torch.where(good, v, torch.full_like(v, float("nan"))))
        sc = torch.where(Nv >= nmin, S / torch.sqrt(torch.clamp(Nv, min=1)), torch.full_like(S, -1e9))
        return sc, Nv, (torch.stack(vals, 1) if need_each else None)

    hs = hyps[seedh]
    cx_ = ((pix % Wc) * c + c / 2 + x0) * PS
    cy_ = ((pix // Wc) * c + c / 2 + y0) * PS
    cur = np.stack([cx_, cy_, 1 / hs[:, 0], hs[:, 1], hs[:, 2]], 1)     # (K,5): th0x, th0y, gamma, wx, wy
    steps = np.array([c * PS / 2, c * PS / 2, dg / 2, dw / 2, dw / 2])
    offs1 = np.array(np.meshgrid(*[np.array([-1, 0, 1])] * 5, indexing="ij")).reshape(5, -1).T   # 243
    curscore = np.full(len(cur), -1e9)
    fine_deadline = budget * 0.85
    for it in range(9):
        if time.time() - T0 > fine_deadline:
            log("fine refinement stopped at iteration", it); break
        newc = []; news = []
        CH = max(1, 2000000 // len(offs1))
        for s0 in range(0, len(cur), CH):
            blk = cur[s0:s0 + CH]
            P = (blk[:, None, :] + offs1[None] * steps[None, None]).reshape(-1, 5)
            P[:, 2] = np.clip(P[:, 2], 1 / 120, 1 / 22)
            T = torch.tensor(P, dtype=torch.float32, device=dev)
            sc, _, _ = score_tracks(T[:, 0], T[:, 1], 1 / T[:, 2], T[:, 3], T[:, 4])
            sc = sc.view(len(blk), -1)
            m, am = sc.max(1)
            am = am.cpu().numpy()
            newc.append(P.reshape(len(blk), -1, 5)[np.arange(len(blk)), am]); news.append(m.cpu().numpy())
        cur = np.concatenate(newc); curscore = np.concatenate(news)
        steps = steps * 0.6
        log("refine it", it, "top scores", np.round(np.sort(curscore)[::-1][:5], 2))
    # final per-exposure values
    T = torch.tensor(cur, dtype=torch.float32, device=dev)
    sc, Nv, vals = score_tracks(T[:, 0], T[:, 1], 1 / T[:, 2], T[:, 3], T[:, 4], need_each=True)
    vals = vals.cpu().numpy(); sc = sc.cpu().numpy()
    # per-night evidence: count nights with mean value > 1
    un = np.unique(night)
    nightv = np.stack([np.nanmean(np.where(np.isfinite(vals[:, night == u]), vals[:, night == u], np.nan), 1)
                       if True else None for u in un], 1)
    nn_pos = np.sum(np.nan_to_num(nightv, nan=-9) > 1.0, 1)
    # robust: score without the single largest exposure (kills one-exposure artefacts)
    v0 = np.nan_to_num(vals, nan=0.0)
    nv = np.sum(np.isfinite(vals), 1)
    sc_rob = (v0.sum(1) - v0.max(1)) / np.sqrt(np.maximum(nv - 1, 1))
    ok = (nn_pos >= 3) & (sc > 0)
    order = np.argsort(-sc_rob)
    out = []
    taken = []
    for j in order:
        if not ok[j] or len(out) >= 120:
            continue
        th0 = cur[j, :2]
        if any(np.hypot(*(th0 - q)) < 6 * ARC for q in taken):
            continue
        taken.append(th0)
        D = 1 / cur[j, 2]
        u = np.array([th0[0], th0[1], 1.0]); u /= np.linalg.norm(u)
        Ce = track_consts(cur[j:j + 1, [2, 3, 4]] * [0, 1, 1] + np.array([[D, 0, 0]]))[0]
        P = D * u[None] + Ce                                  # field frame, (Ne,3)
        Pe = P @ F                                            # equatorial
        uobs = Pe / np.linalg.norm(Pe, axis=1, keepdims=True)
        X0 = obs_ref + D * (u @ F)
        V = np.array([cur[j, 3] * D, cur[j, 4] * D, 0.0]) @ F
        try:
            st, r = fit_orbit(np.r_[X0, V], t_ref, t, obs, uobs)
        except Exception:
            continue
        if not np.all(np.isfinite(st)) or np.max(r) > 1.0:
            continue
        R = np.linalg.norm(st[:3]); v2 = st[3:] @ st[3:]
        if R < 25.5 or v2 > 2 * GM / R * 0.98:
            continue
        s_eff = float(sc_rob[j])
        p = 1 / (1 + math.exp(-(s_eff - 8.0) / 1.0))
        out.append({"epoch_mjd": t_ref, "state": [float(x) for x in st], "p": round(p, 5),
                    "score": round(float(sc[j]), 3), "nights": int(nn_pos[j])})
    log("output", len(out), "candidates; top", [o_["score"] for o_ in out[:8]])
    write_out(a.out, out)


if __name__ == "__main__":
    main()
