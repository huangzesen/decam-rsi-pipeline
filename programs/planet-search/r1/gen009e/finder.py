"""Gen 9e: seedless GPU shift-and-stack (synthetic tracking) over a BK-frame orbit grid.  See NOTES.md.

1. detect.py (from gen 7a) only to get per-CCD matched-filter S/N maps (2x decimated) + CCD geometry.
2. Per epoch (exposures binned within ~40 min of each other) a 1"/px tangent-plane S/N mosaic (fine), static
   sources masked, then a max-pooled CxC coarse mosaic, normalised to zero-mean/unit-sd noise.
3. For each inverse distance gamma (coarse-to-fine order) every coarse epoch map is warped into the BK alpha frame;
   for every rate alpha-dot on a grid (|alpha-dot| <= bound speed) the epoch maps are shifted by alpha-dot*t and
   summed.  A running max over all hypotheses (and its hypothesis id) is kept per alpha cell.
4. Local maxima of the max map -> local refinement of (alpha, alpha-dot, gamma) on the fine mosaics -> two-body
   state fit -> score (summed S/N over epochs, nights positive) -> p.
"""
import argparse, csv, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import orbit
from orbit import T0, log, GM, C_AU, ARC, predict_dirs, fit_orbit, tangent_frame

PS = 1.0 * ARC           # fine mosaic pixel


def device():
    import torch
    if torch.cuda.is_available():
        return torch, "cuda"
    if torch.backends.mps.is_available():
        return torch, "mps"
    return torch, "cpu"


def n_workers():
    if os.environ.get("TNO_WORKERS"):
        return int(os.environ["TNO_WORKERS"])
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = os.cpu_count() or 4
    import torch
    return int(max(2, min(7, ncpu - 1))) if torch.cuda.is_available() else 4


def run_detection(rows, sdir, deadline, nmax):
    """Map-producing detection of up to nmax exposures, nights round-robin.  Returns {ei: maps}."""
    import detect
    from multiprocessing import get_context
    t = np.array([float(r["mjd_mid"]) for r in rows])
    night = np.floor(t - 0.5).astype(int)
    per = [sorted(np.flatnonzero(night == u), key=lambda i: t[i]) for u in np.unique(night)]
    # spread picks over each night: alternate ends -> middle so few picks still span the night
    order = []
    k = 0
    while len(order) < len(rows):
        for p in per:
            if k < len(p):
                order.append(p[k])
        k += 1
    order = order[:nmax]
    tasks = [(int(i), os.path.join(sdir, rows[i]["image"]), os.path.join(sdir, rows[i]["mask"]), 50.0, True)
             for i in order]
    nw = n_workers()
    log("workers", nw, "exposures", len(tasks), "of", len(rows))
    pool = get_context("spawn").Pool(nw)
    res = {}
    inflight = []
    it = iter(tasks)
    durs = []
    sub_done = False
    try:
        while True:
            while not sub_done and len(inflight) < nw:
                est = np.median(durs) if durs else 25.0
                if time.time() - T0 + est > deadline and res:
                    sub_done = True; break
                try:
                    tk = next(it)
                except StopIteration:
                    sub_done = True; break
                inflight.append((pool.apply_async(detect.detect_exposure, (tk,)), time.time()))
            if not inflight:
                break
            still = []
            for ar, ts in inflight:
                if ar.ready():
                    try:
                        o = ar.get()
                        if o["maps"]:
                            res[o["ei"]] = o["maps"]
                        durs.append(time.time() - ts)
                        log(f"exp {o['ei']}: {len(o['maps'])} ccds ({time.time() - ts:.1f}s) err={o['err']}")
                    except Exception as ex:
                        log("worker error", ex)
                else:
                    still.append((ar, ts))
            inflight = still
            if time.time() - T0 > deadline + 30:
                log("detection hard stop"); break
            time.sleep(0.05)
    finally:
        pool.terminate()
    return res


def mosaic(torch, dev, maps, F, lim):
    """Resample the CCD S/N maps of one exposure onto the fine tangent grid; returns (H,W) float32 (nan off data)."""
    x0, y0, W, H = lim
    Ft = torch.tensor(F.T, dtype=torch.float32, device=dev)
    M = torch.full((H, W), float("nan"), dtype=torch.float32, device=dev)
    for g, m in maps:
        rc, dc = g["rc"], g["dc"]
        cu = np.array([math.cos(dc) * math.cos(rc), math.cos(dc) * math.sin(rc), math.sin(dc)])
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
        M[ja:jb, ia:ib] = torch.where(ok, s, M[ja:jb, ia:ib])
    return M


def sample(torch, flat, lim, tx, ty):
    """Bilinear sample of a flattened fine mosaic at tangent coords (rad); nan off data."""
    x0, y0, W, H = lim
    px = tx / PS - x0
    py = ty / PS - y0
    off = (px < 0) | (px > W - 2) | (py < 0) | (py > H - 2) | ~torch.isfinite(px) | ~torch.isfinite(py)
    px = torch.where(off, torch.zeros_like(px), px)
    py = torch.where(off, torch.zeros_like(py), py)
    ix = torch.floor(px); iy = torch.floor(py)
    fx = px - ix; fy = py - iy
    k = iy.long() * W + ix.long()
    s = (1 - fx) * (1 - fy) * flat[k] + fx * (1 - fy) * flat[k + 1] + (1 - fx) * fy * flat[k + W] + fx * fy * flat[k + W + 1]
    return torch.where(off, torch.full_like(s, float("nan")), s)


def valid_state(st):
    r = float(np.linalg.norm(st[:3]))
    return np.all(np.isfinite(st)) and r >= 25.5 and np.dot(st[3:], st[3:]) / 2 - GM / r < 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=300)
    a = ap.parse_args()
    budget = a.budget_s
    sdir = a.scenario
    json.dump({"candidates": []}, open(a.out, "w"))
    rows = list(csv.DictReader(open(os.path.join(sdir, "exposures.csv"))))
    torch, dev = device()
    cuda = dev == "cuda"
    nmax = int(os.environ.get("TNO_NMAX", 48 if cuda else 24))
    maps = run_detection(rows, sdir, budget * 0.35, nmax)
    exps = sorted(maps)
    if len(exps) < 3:
        log("too few exposures"); return
    et = np.array([float(rows[e]["mjd_mid"]) for e in exps])
    eobs = np.array([[float(rows[e][k]) for k in ("obs_x", "obs_y", "obs_z")] for e in exps])
    night = np.floor(et - 0.5).astype(int)
    if len(np.unique(night)) < 3:
        log("fewer than 3 nights"); return
    # common frame
    ra0 = np.median([math.radians(float(rows[e]["ra_center"])) for e in exps])
    de0 = np.median([math.radians(float(rows[e]["dec_center"])) for e in exps])
    F = tangent_frame(ra0, de0)
    # extent from CCD centres
    th = []
    for e in exps:
        for g, m in maps[e]:
            cu = np.array([math.cos(g["dc"]) * math.cos(g["rc"]), math.cos(g["dc"]) * math.sin(g["rc"]), math.sin(g["dc"])])
            c = F @ cu
            th.append((c[0] / c[2], c[1] / c[2], g["rad"]))
    th = np.array(th)
    lo = np.percentile(th[:, :2] - th[:, 2:3], 0.5, axis=0)
    hi = np.percentile(th[:, :2] + th[:, 2:3], 99.5, axis=0)
    x0, y0 = int(math.floor(lo[0] / PS)), int(math.floor(lo[1] / PS))
    W, H = int(math.ceil(hi[0] / PS)) - x0, int(math.ceil(hi[1] / PS)) - y0
    W, H = min(W, 9000), min(H, 9000)
    lim = (x0, y0, W, H)
    # epochs: exposures within 0.03 d of each other in the same night are averaged (body moves < ~2")
    order = np.argsort(et)
    groups = []
    for i in order:
        if groups and night[groups[-1][-1]] == night[i] and et[i] - et[groups[-1][0]] < 0.03:
            groups[-1].append(i)
        else:
            groups.append([i])
    t1 = time.time()
    fine = []
    for gr in groups:
        S = None; N = None
        for i in gr:
            M = mosaic(torch, dev, maps[exps[i]], F, lim)
            v = torch.isfinite(M)
            S = torch.where(v, M, torch.zeros_like(M)) if S is None else S + torch.where(v, M, torch.zeros_like(M))
            N = v.float() if N is None else N + v.float()
            maps[exps[i]] = None
        fine.append(torch.where(N > 0, S / torch.sqrt(torch.clamp(N, min=1)), torch.full_like(S, float("nan"))))
    del maps
    gt = np.array([et[gr].mean() for gr in groups])
    gobs = np.array([eobs[gr].mean(0) for gr in groups])
    gnight = np.array([night[gr[0]] for gr in groups])
    NE = len(groups)
    log(f"{NE} epochs, {len(np.unique(gnight))} nights, mosaics {W}x{H} on {dev} in {time.time() - t1:.1f}s")
    # static mask: per-night max > 2 on >= half the nights with data (>= 2), or > 8 on >= 2 nights
    un = np.unique(gnight)
    CH = 1024
    for r0 in range(0, H, CH):
        cnt = nh = nb = 0
        for nn in un:
            mx = None
            for k in np.flatnonzero(gnight == nn):
                v = torch.nan_to_num(fine[k][r0:r0 + CH], nan=-99.0)
                mx = v if mx is None else torch.maximum(mx, v)
            cnt = cnt + (mx > 2.0).float(); nh = nh + (mx > -98).float(); nb = nb + (mx > 8.0).float()
        bad = ((cnt >= 2) & (cnt >= 0.5 * nh)) | (nb >= 2)
        bad = torch.nn.functional.max_pool2d(bad.float()[None, None], 5, 1, 2)[0, 0] > 0
        for k in range(NE):
            fine[k][r0:r0 + CH][bad] = float("nan")
    # coarse maps: max-pool C x C, normalised by robust noise stats, clipped
    C = int(os.environ.get("TNO_C", 5))
    coarse = []
    for k in range(NE):
        f = torch.clamp(torch.nan_to_num(fine[k], nan=-5.0), -5, 7)
        cm = torch.nn.functional.max_pool2d(f[None, None], C, C)[0, 0]
        okm = cm > -4.9
        s = cm[okm][::7]
        if s.numel() < 100:
            coarse.append(torch.zeros_like(cm)); continue
        med = torch.median(s); sd = 1.4826 * torch.median(torch.abs(s - med))
        z = torch.where(okm, (cm - med) / torch.clamp(sd, min=0.05), torch.zeros_like(cm))
        coarse.append(torch.clamp(z, -3, 5))
    Hc, Wc = coarse[0].shape
    CS = C * PS
    fine_flat = [f.reshape(-1) for f in fine]
    log(f"coarse {Wc}x{Hc}, C={C}, t={time.time() - T0:.1f}s")
    # ---------------------------------------------------------------- gamma / rate grid
    t_ref = 0.5 * (gt.min() + gt.max())
    of = gobs @ F.T
    tt = gt - t_ref
    A = np.c_[np.ones_like(tt), tt]
    resid = of[:, :2] - A @ np.linalg.lstsq(A, of[:, :2], rcond=None)[0]
    curv = float(np.max(np.hypot(resid[:, 0], resid[:, 1])))
    GT = float(os.environ.get("TNO_GTOL", 4.0))      # gamma step in units of a 1-cell curvature displacement
    dg = (GT * CS) / max(curv, 1e-6)
    # each epoch map is max-dilated by its own curvature displacement over half a gamma step, then renormalised
    rk = np.hypot(resid[:, 0], resid[:, 1])
    for k in range(NE):
        h = int(math.ceil(rk[k] * dg / 2 / CS - 0.5))
        if h <= 0:
            continue
        d = torch.nn.functional.max_pool2d(coarse[k][None, None], 2 * h + 1, 1, h)[0, 0]
        s = d[::3, ::3].flatten()
        med = torch.median(s); sd = 1.4826 * torch.median(torch.abs(s - med))
        coarse[k] = torch.clamp((d - med) / torch.clamp(sd, min=0.05), -3, 5)
        log(f"epoch {k}: dilated by {h} cells")
    gmin, gmax = 1 / 150.0, 1 / 25.5
    grid = np.arange(gmin, gmax + dg, dg) if dg < (gmax - gmin) else np.array([1 / 45.0])
    # coarse-to-fine order (bit-reversal-like): typical distances (35-50 au) first
    pri = np.abs(1 / grid - 42.0)
    gorder = list(np.argsort(pri))
    log(f"curvature {curv:.4f} au, dgamma {dg:.2e}, {len(grid)} gammas")
    # alpha grid of coarse cells (centres) in tangent coords
    ax = (torch.arange(Wc, device=dev, dtype=torch.float32) + 0.5) * CS + x0 * PS
    ay = (torch.arange(Hc, device=dev, dtype=torch.float32) + 0.5) * CS + y0 * PS
    AY, AX = torch.meshgrid(ay, ax, indexing="ij")
    tmax = float(np.max(np.abs(tt))) + 0.1
    best = torch.full((Hc, Wc), -1e9, device=dev)
    bid = torch.zeros((Hc, Wc), dtype=torch.int64, device=dev)
    hyps = []
    P = int(os.environ.get("TNO_PAD", 0))
    search_deadline = budget * 0.80
    nrm = 1.0 / math.sqrt(NE)
    ng = 0
    for gi in gorder:
        if time.time() - T0 > search_deadline:
            break
        g = float(grid[gi])
        k_ = 0.5 * GM * g ** 3
        tp = tt - 1.0 / (C_AU * g)
        uu = 1 - k_ * tp * tp
        fe = 1 - k_ * tp * tp - g * of[:, 2]
        npara = math.sqrt(2 * GM * g ** 3)
        step = CS / tmax
        nr = int(math.ceil(npara / step))
        # shifts in coarse cells per epoch for each rate
        r1 = np.arange(-nr, nr + 1) * step
        RX, RY = np.meshgrid(r1, r1)
        keep = np.hypot(RX, RY) <= 0.9 * npara
        rates = np.c_[RX[keep], RY[keep]]
        sh = np.rint(rates[:, None, :] * (tp / fe)[None, :, None] / CS).astype(int)     # (R, NE, 2)
        sh, uix = np.unique(sh, axis=0, return_index=True)
        rates = rates[uix]
        maxs = int(np.abs(sh).max()) + 1
        # warp each epoch's coarse map into the alpha frame at this gamma (rate 0), padded by maxs
        Wp, Hp = Wc + 2 * maxs, Hc + 2 * maxs
        warped = []
        pax = (torch.arange(Wp, device=dev, dtype=torch.float32) - maxs + 0.5) * CS + x0 * PS
        pay = (torch.arange(Hp, device=dev, dtype=torch.float32) - maxs + 0.5) * CS + y0 * PS
        PY, PX = torch.meshgrid(pay, pax, indexing="ij")
        for k in range(NE):
            thx = (PX * uu[k] - g * of[k, 0]) / fe[k]
            thy = (PY * uu[k] - g * of[k, 1]) / fe[k]
            gx = ((thx - x0 * PS) / CS) / Wc * 2 - 1
            gy = ((thy - y0 * PS) / CS) / Hc * 2 - 1
            w = torch.nn.functional.grid_sample(coarse[k][None, None], torch.stack([gx, gy], -1)[None],
                                                mode="nearest", padding_mode="zeros", align_corners=False)[0, 0]
            warped.append(w.to(torch.float16))
        hbase = len(hyps)
        B = 32
        buf = torch.empty((B, Hc, Wc), dtype=torch.float16, device=dev)
        for r0 in range(0, len(rates), B):
            nb = min(B, len(rates) - r0)
            for b in range(nb):
                ri = r0 + b
                for k in range(NE):
                    dx, dy = int(sh[ri, k, 0]), int(sh[ri, k, 1])
                    sl = warped[k][maxs + dy:maxs + dy + Hc, maxs + dx:maxs + dx + Wc]
                    if k == 0:
                        buf[b].copy_(sl)
                    else:
                        buf[b].add_(sl)
            bv, bi = buf[:nb].max(0)
            bv = bv.float()
            upd = bv > best
            best = torch.maximum(bv, best)
            bid = torch.where(upd, bi.long() + (hbase + r0), bid)
            if time.time() - T0 > search_deadline:
                break
        for ri in range(len(rates)):
            hyps.append((g, rates[ri, 0], rates[ri, 1]))
        ng += 1
        log(f"gamma {g:.4f} (r {1 / g:.1f} au): {len(rates)} rates, t={time.time() - T0:.1f}s")
    log(f"searched {ng}/{len(grid)} gammas, {len(hyps)} hypotheses")
    # ---------------------------------------------------------------- peaks of the max map
    bz = best * nrm
    pk = (bz == torch.nn.functional.max_pool2d(bz[None, None], 5, 1, 2)[0, 0]) & (bz > 3.0)
    ys, xs = torch.nonzero(pk, as_tuple=True)
    v = bz[ys, xs]
    o = torch.argsort(-v)[:int(os.environ.get("TNO_NPEAK", 600))]
    ys, xs, v = ys[o].cpu().numpy(), xs[o].cpu().numpy(), v[o].cpu().numpy()
    ids = bid[ys, xs].cpu().numpy() if len(ys) else np.zeros(0, int)
    log(f"{len(ys)} peaks, top coarse z {np.round(v[:10], 2)}")
    # ---------------------------------------------------------------- refinement on fine mosaics
    T_tt = torch.tensor(tt, dtype=torch.float32, device=dev)
    T_of = torch.tensor(of, dtype=torch.float32, device=dev)

    def score(g, al, ad):
        """g, al (K,2), ad (K,2) float tensors (K,) -> per-epoch S/N (K, NE) (nan off data)."""
        tp = T_tt[None] - 1.0 / (C_AU * g[:, None])
        k_ = 0.5 * GM * g[:, None] ** 3
        uu = 1 - k_ * tp * tp
        fe = uu - g[:, None] * T_of[None, :, 2]
        out = []
        for k in range(NE):
            thx = (al[:, 0] * uu[:, k] + ad[:, 0] * tp[:, k] - g * T_of[k, 0]) / fe[:, k]
            thy = (al[:, 1] * uu[:, k] + ad[:, 1] * tp[:, k] - g * T_of[k, 1]) / fe[:, k]
            out.append(sample(torch, fine_flat[k], lim, thx, thy))
        return torch.stack(out, 1)

    def obj(S):
        v = torch.isfinite(S)
        s = torch.clamp(torch.nan_to_num(S, nan=0.0), -3, 6)
        return s.sum(1) / torch.sqrt(torch.clamp(v.float().sum(1), min=1))

    cands = []
    lin = torch.linspace(-1, 1, 5, device=dev)
    tdead = budget * 0.93
    for pi in range(len(ys)):
        if time.time() - T0 > tdead:
            break
        g0, rx, ry = hyps[ids[pi]]
        al = np.array([float(AX[ys[pi], xs[pi]]), float(AY[ys[pi], xs[pi]])])
        ad = np.array([rx, ry])
        gg = g0
        sa, sr, sgm = CS * 0.6, step * 0.6, dg * 0.6
        for it in range(4):
            # 5^4 grid on (alpha, rate), 3 gammas
            Q = torch.stack(torch.meshgrid(lin, lin, lin, lin, indexing="ij"), -1).reshape(-1, 4)
            Gs = torch.tensor([gg - sgm, gg, gg + sgm], device=dev, dtype=torch.float32).clamp(min=gmin * 0.9, max=gmax)
            K = Q.shape[0]
            Gv = Gs.repeat_interleave(K)
            Q3 = Q.repeat(3, 1)
            alv = torch.tensor(al, dtype=torch.float32, device=dev)[None] + Q3[:, :2] * sa
            adv = torch.tensor(ad, dtype=torch.float32, device=dev)[None] + Q3[:, 2:] * sr
            z = obj(score(Gv, alv, adv))
            j = int(torch.argmax(z))
            al, ad, gg = alv[j].cpu().numpy().astype(float), adv[j].cpu().numpy().astype(float), float(Gv[j])
            sa, sr, sgm = sa * 0.45, sr * 0.45, sgm * 0.5
        S = score(torch.tensor([gg], device=dev), torch.tensor(al[None], dtype=torch.float32, device=dev),
                  torch.tensor(ad[None], dtype=torch.float32, device=dev))[0].cpu().numpy()
        valid = np.isfinite(S)
        if valid.sum() < 3:
            continue
        sc = np.clip(S[valid], -3, 6)
        zf = float(sc.sum() / math.sqrt(valid.sum()))
        nights_pos = len(np.unique(gnight[valid & (np.nan_to_num(S) > 1.5)]))
        nights_dat = len(np.unique(gnight[valid]))
        if nights_pos < 3:
            continue
        cands.append(dict(z=zf, g=gg, al=al, ad=ad, npos=nights_pos, ndat=nights_dat, S=S))
        if len(cands) % 50 == 0:
            log(f"refined {pi + 1} peaks, {len(cands)} cands")
    log(f"{len(cands)} candidates refined, t={time.time() - T0:.1f}s")
    cands.sort(key=lambda c: -c["z"])
    # ---------------------------------------------------------------- states, dedup, write
    out = []
    kept = []
    for c in cands:
        if time.time() - T0 > budget * 0.98 or len(out) >= 200:
            break
        g, al, ad = c["g"], c["al"], c["ad"]
        if kept and min(np.hypot(*(al - k_)) for k_ in kept) < 4 * ARC:
            continue
        k_ = 0.5 * GM * g ** 3
        tp = tt - 1.0 / (C_AU * g)
        uu = 1 - k_ * tp * tp
        fe = uu - g * of[:, 2]
        thx = (al[0] * uu + ad[0] * tp - g * of[:, 0]) / fe
        thy = (al[1] * uu + ad[1] * tp - g * of[:, 1]) / fe
        vv = np.c_[thx, thy, np.ones(NE)] @ F
        vv /= np.linalg.norm(vv, axis=1)[:, None]
        state0 = np.r_[F.T @ (np.array([al[0], al[1], 1.0]) / g), F.T @ (np.array([ad[0], ad[1], 0.0]) / g)]
        try:
            st, res = fit_orbit(state0, t_ref, gt, gobs, vv)
        except Exception:
            continue
        if not valid_state(st) or np.max(res) > 1.0:
            continue
        kept.append(al)
        z = c["z"] + 0.3 * (c["npos"] - 3)
        p = 1 / (1 + math.exp(-1.5 * (z - 9.0)))
        out.append(dict(epoch_mjd=float(t_ref), state=[float(x) for x in st], p=round(float(p), 5)))
        if len(out) % 25 == 0:
            json.dump({"candidates": out}, open(a.out, "w"))
    out.sort(key=lambda d: -d["p"])
    json.dump({"candidates": out[:200]}, open(a.out, "w"))
    log(f"wrote {len(out)} candidates; top p {[d['p'] for d in out[:8]]}; z {[round(c['z'], 1) for c in cands[:8]]}")


if __name__ == "__main__":
    main()
