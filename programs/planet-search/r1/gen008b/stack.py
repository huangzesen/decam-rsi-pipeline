"""Gen 5c: pair-seeded multi-night shift-and-stack on the GPU.

For each assumed inverse distance gamma (BK frame, same linearised model as the linker), every pair of seed
detections on two different nights fixes (alpha, alpha-dot); the body's predicted position on every other exposure
is sampled on per-exposure S/N mosaics (the stored matched-filter S/N maps resampled onto one tangent-plane grid,
1"/px, on the GPU).  The clipped S/N summed over the exposures of the *other* nights is the evidence: a body seen
at S/N ~3-4 on two nights and below threshold elsewhere can still stack up.  Survivors are converted to a two-body
state (fit to the model's predicted positions), re-scored with exact predictions, de-duplicated against each other
and against the linker's candidates.
"""
import math, os, time
import numpy as np
from scipy.spatial import cKDTree
from orbit import GM, C_AU, ARC, T0, log, predict_dirs, fit_orbit

PS = 1.0 * ARC          # mosaic pixel (rad)


def _dev(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_mosaics(torch, dev, FP, exps, F, lim):
    """lim = (x0, y0, W, H) in mosaic pixels. Returns dict e -> (1,1,H,W) float16 tensor (nan off data)."""
    x0, y0, W, H = lim
    Ft = torch.tensor(F.T, dtype=torch.float32, device=dev)
    out = {}
    for e in exps:
        M = torch.full((H, W), float("nan"), dtype=torch.float16, device=dev)
        for g, m, cu, cmin in FP.maps[e]:
            # CCD centre in mosaic pixels
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
        out[e] = M[None, None]
    return out


def _nanmed_mps(torch, blk):
    """nanmedian over dim 0 (MPS lacks nanmedian): nan -> -inf/+inf alternately is biased; use sort."""
    n = torch.isfinite(blk).sum(0)
    s, _ = torch.sort(torch.nan_to_num(blk, nan=float("inf")), dim=0)
    k = torch.clamp((n - 1) // 2, min=0)
    med = torch.gather(s, 0, k[None].long())[0]
    return torch.where(n > 0, med, torch.full_like(med, float("nan")))


def sample(torch, mos, lim, tx, ty):
    """tx, ty (N,) float32 tensors in rad -> (N,) S/N, nan off data.  Manual bilinear gather with float32 coordinates
    (grid_sample would need a float16 grid for a float16 mosaic: ~6 px error across a 12k-px mosaic)."""
    x0, y0, W, H = lim
    flat = mos.view(-1)
    px = tx / PS - x0
    py = ty / PS - y0
    off = (px < 0) | (px > W - 2) | (py < 0) | (py > H - 2) | ~torch.isfinite(px) | ~torch.isfinite(py)
    px = torch.where(off, torch.zeros_like(px), px)
    py = torch.where(off, torch.zeros_like(py), py)
    ix = torch.floor(px); iy = torch.floor(py)
    fx = px - ix; fy = py - iy
    k = iy.long() * W + ix.long()
    v00 = flat[k].float(); v01 = flat[k + 1].float(); v10 = flat[k + W].float(); v11 = flat[k + W + 1].float()
    s = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v01 + (1 - fx) * fy * v10 + fx * fy * v11
    return torch.where(off, torch.full_like(s, float("nan")), s)


def run_stack(L, deadline, existing_pd=None):
    """L: dict of linker locals. Returns list of candidate dicts (state, p, mag, z...)."""
    import torch; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False  # gen 8b: exact fp32 on A100
    dev = _dev(torch)
    FP, F, grid, order = L["FP"], L["F"], L["grid"], L["order"]
    et, eobs, enight, E, D = L["et"], L["eobs"], L["enight"], L["E"], L["D"]
    u, gi, sn, t_ref, done = L["u"], L["gi"], L["sn"], L["t_ref"], L["done"]
    core = L["core"]
    mexps = [e for e in done if FP.has(e)]
    if len(np.unique(enight[mexps])) < 3:
        return []
    t1 = time.time()
    ug = u[gi] @ F.T
    th = ug[:, :2] / ug[:, 2:3]
    lo = np.percentile(th, 0.1, axis=0) - 0.01
    hi = np.percentile(th, 99.9, axis=0) + 0.01
    x0, y0 = int(math.floor(lo[0] / PS)), int(math.floor(lo[1] / PS))
    W, H = int(math.ceil(hi[0] / PS)) - x0, int(math.ceil(hi[1] / PS)) - y0
    if W * H * len(mexps) * 2 > 12e9:
        mexps = mexps[: max(3, int(12e9 / (W * H * 2)))]
    lim = (x0, y0, W, H)
    mos = build_mosaics(torch, dev, FP, mexps, F, lim)
    # static sources: per-pixel median over exposures (nan-aware) > 1.5 -> masked in every mosaic (stars and their
    # wings would otherwise dominate any stack that crosses them)
    if len(np.unique(enight[mexps])) >= 3:
        # static sources: positive (S/N > 2) on >= half of the nights with data (and >= 2 nights) at the same pixel,
        # or > 8 on >= 2 nights -> masked in every mosaic.  Per-night max first, so a slow body seen on several
        # exposures of one night is not masked.  (Stars and their wings would otherwise dominate the stacks.)
        CH = 1024
        nlist = np.unique(enight[mexps])
        for r0 in range(0, H, CH):
            cnt = None
            for nn in nlist:
                ee = [e for e in mexps if enight[e] == nn]
                mx = None
                for e in ee:
                    v = mos[e][0, 0, r0:r0 + CH].float()
                    v = torch.nan_to_num(v, nan=-99.0)
                    mx = v if mx is None else torch.maximum(mx, v)
                has = (mx > -98).float(); pos = (mx > 2.0).float(); big = (mx > 8.0).float()
                if cnt is None:
                    cnt, nh, nb = pos, has, big
                else:
                    cnt, nh, nb = cnt + pos, nh + has, nb + big
            bad = ((cnt >= 2) & (cnt >= 0.5 * nh)) | (nb >= 2)
            # grow by 1 px (wings)
            bad = torch.nn.functional.max_pool2d(bad.float()[None, None], 3, 1, 1)[0, 0] > 0
            for e in mexps:
                v = mos[e][0, 0, r0:r0 + CH]
                v[bad] = float("nan")
    log(f"stack: {len(mexps)} mosaics {W}x{H} on {dev} in {time.time() - t1:.1f}s")
    mexps = np.array(mexps)
    mn = enight[mexps]
    obs_f = eobs @ F.T
    te = et[mexps] - t_ref
    # seeds: usable detections S/N >= 3.5, per night the brightest 5000 (one exposure per night: the best seeing
    # is not known here, so all exposures; duplicates of one body on the same night are harmless)
    nights = np.unique(mn)
    seeds = []
    for nn in nights:
        m = gi[(enight[E[gi]] == nn) & (sn[gi] >= 3.5) & (sn[gi] < 40)]
        m = m[np.argsort(-sn[m])[:5000]]
        seeds.append(m)
    seeds = np.concatenate(seeds)
    ns = enight[E[seeds]]
    us = u[seeds] @ F.T
    theta = us[:, :2] / us[:, 2:3]
    ts = et[E[seeds]] - t_ref
    os_ = obs_f[E[seeds]]
    un = np.unique(ns)
    hits = []
    ZT = float(os.environ.get("TNO_STACK_ZT", 6.8))   # z over the other nights' exposures; null max ~6.1 (scen02, 30 gammas)
    ZA = 6.0
    NGMAX = int(os.environ.get("TNO_STACK_NG", 100000))
    REJ = {}
    ng = 0
    for kk, gidx in enumerate(order):
        if time.time() - T0 > deadline or ng >= NGMAX:
            break
        g = grid[gidx]
        ng += 1
        tp = ts - 1.0 / (C_AU * g)
        k_ = 0.5 * GM * g ** 3
        f = 1.0 - k_ * tp * tp - g * os_[:, 2]
        phi = theta * f[:, None] + g * os_[:, :2]
        uu = 1.0 - k_ * tp * tp
        npara = math.sqrt(2 * GM * g ** 3)
        tpe = te - 1.0 / (C_AU * g)
        uue = 1.0 - k_ * tpe * tpe
        fe = 1.0 - k_ * tpe * tpe - g * obs_f[mexps, 2]
        T_tpe = torch.tensor(tpe, dtype=torch.float32, device=dev)
        T_uue = torch.tensor(uue, dtype=torch.float32, device=dev)
        T_fe = torch.tensor(fe, dtype=torch.float32, device=dev)
        T_oe = torch.tensor(g * obs_f[mexps, :2], dtype=torch.float32, device=dev)
        idx_n = [np.flatnonzero(ns == n_) for n_ in un]
        trees = [cKDTree(phi[ix]) for ix in idx_n]
        for p in range(len(un)):
            for q in range(p + 1, len(un)):
                dt = un[q] - un[p]
                sd = trees[p].sparse_distance_matrix(trees[q], npara * (dt + 0.5) + 2 * ARC, output_type="ndarray")
                if len(sd) == 0:
                    continue
                i = idx_n[p][sd["i"]]; j = idx_n[q][sd["j"]]
                Dd = uu[i] * tp[j] - uu[j] * tp[i]
                al = (phi[i] * tp[j, None] - phi[j] * tp[i, None]) / Dd[:, None]
                ad = (uu[i, None] * phi[j] - uu[j, None] * phi[i]) / Dd[:, None]
                ok = (np.hypot(al[:, 0], al[:, 1]) <= core) & (np.hypot(ad[:, 0], ad[:, 1]) <= npara)
                if not ok.any():
                    continue
                i, j, al, ad = i[ok], j[ok], al[ok], ad[ok]
                # relative coordinates in float32 (al ~ 1e-2 rad, fine)
                A = torch.tensor(al, dtype=torch.float32, device=dev)
                Ad = torch.tensor(ad, dtype=torch.float32, device=dev)
                other = (mn != un[p]) & (mn != un[q])
                ks = np.flatnonzero(other)
                if len(ks) == 0:
                    continue
                S = torch.zeros(len(i), device=dev); N = torch.zeros(len(i), device=dev)
                NS = {}
                for k in ks:
                    ph = A * T_uue[k] + Ad * T_tpe[k]
                    tt = (ph - T_oe[k]) / T_fe[k]
                    s = sample(torch, mos[mexps[k]], lim, tt[:, 0], tt[:, 1])
                    v = torch.isfinite(s)
                    s = torch.clamp(torch.nan_to_num(s, nan=0.0), -3.0, 4.0)
                    S += s; N += v.float()
                    key = int(mn[k])
                    if key not in NS:
                        NS[key] = [torch.zeros_like(S), torch.zeros_like(S)]
                    NS[key][0] += s; NS[key][1] += v.float()
                z = S / torch.sqrt(torch.clamp(N, min=1.0))
                T_sab = torch.tensor(np.minimum(sn[seeds[i]], 6.0) + np.minimum(sn[seeds[j]], 6.0), dtype=torch.float32, device=dev)
                zall = (S + T_sab) / torch.sqrt(N + 2.0)
                # nights with positive evidence (night stack > 1.5)
                npos = torch.zeros_like(S)
                for a_, b_ in NS.values():
                    npos += ((a_ / torch.sqrt(torch.clamp(b_, min=1.0))) > 1.5).float()
                sel = (z >= ZT) & (npos >= min(2, len(NS))) & (npos >= 1) & (zall >= ZA)
                if bool(sel.any()):
                    w = torch.nonzero(sel)[:, 0].cpu().numpy()
                    zz = zall[sel].cpu().numpy(); zo = z[sel].cpu().numpy()
                    for a, b, c, d, zv, npv, nv, zov in zip(i[w], j[w], al[w], ad[w], zz, npos[sel].cpu().numpy(), N[sel].cpu().numpy(), zo):
                        hits.append((float(zv), int(seeds[a]), int(seeds[b]), g, c.copy(), d.copy(), int(npv), int(nv), float(zov)))
        if len(hits) > 200000:
            hits.sort(key=lambda h: -h[0]); hits = hits[:50000]
    log(f"stack: {ng}/{len(grid)} gammas, {len(hits)} hits (z>={ZT}, zall>={ZA}) in {time.time() - t1:.1f}s")
    if not hits:
        return []
    zs = sorted(hits, key=lambda h: -h[8])[:15]
    log('stack top z_other', [(round(h[8], 1), h[7], h[6]) for h in zs])
    hits.sort(key=lambda h: -(h[8] + 0.3 * h[6]))
    # de-duplicate on the predicted position at t_ref (alpha) + rate
    out = []
    kept_al = []
    used_seed = set()
    n_h = 0
    for h in hits:
        n_h += 1
        if len(out) >= 300 or time.time() - T0 > deadline + 20:
            break
        zv, a, b, g, al, ad = h[:6]
        if a in used_seed or b in used_seed:
            continue
        if kept_al and np.min(np.hypot(*(np.array(kept_al) - al).T)) < 4 * ARC:
            continue
        # two-body state: fit to the model's predictions on all map exposures + the two seeds
        rf = np.array([al[0], al[1], 1.0]) / g
        vf = np.array([ad[0], ad[1], 0.0]) / g
        state0 = np.r_[F.T @ rf, F.T @ vf]
        tpe = te - 1.0 / (C_AU * g); k_ = 0.5 * GM * g ** 3
        ph = al[None] * (1 - k_ * tpe * tpe)[:, None] + ad[None] * tpe[:, None]
        tt = (ph - g * obs_f[mexps, :2]) / (1.0 - k_ * tpe * tpe - g * obs_f[mexps, 2])[:, None]
        vv = np.c_[tt, np.ones(len(tt))] @ F
        vv /= np.linalg.norm(vv, axis=1)[:, None]
        tobs = np.r_[et[mexps], et[E[a]], et[E[b]]]
        oobs = np.r_[eobs[mexps], eobs[E[a]][None], eobs[E[b]][None]]
        uobs = np.r_[vv, u[a][None], u[b][None]]
        try:
            st, res = fit_orbit(state0, t_ref, tobs, oobs, uobs)
        except Exception:
            REJ['fitx'] = REJ.get('fitx', 0) + 1; continue
        if not np.all(np.isfinite(st)) or res[-2:].max() > 1.0:
            REJ['fit'] = REJ.get('fit', 0) + 1
            if REJ['fit'] <= 3:
                log('DEBUG fit res', np.round(res, 2), 1 / g, zv)
                p0 = predict_dirs(state0, t_ref, tobs, oobs)
                log('DEBUG init res', np.round(np.arccos(np.clip(np.sum(p0 * uobs, 1), -1, 1)) / ARC, 2), 'seed sep', np.arccos(np.dot(u[a], u[b])) / ARC, et[E[a]], et[E[b]])
            continue
        r = float(np.linalg.norm(st[:3]))
        if not (25.5 < r < 300) or np.dot(st[3:], st[3:]) / 2 - GM / r >= 0:
            REJ['r'] = REJ.get('r', 0) + 1; continue
        pdm = predict_dirs(st, t_ref, et[done], eobs[done])
        if existing_pd is not None and len(existing_pd):
            # same body as a linker candidate (within 3" on most exposures)?
            dd = np.linalg.norm(existing_pd - pdm[None], axis=2) / ARC
            if np.any(np.median(dd, axis=1) < 3.0):
                REJ['dup'] = REJ.get('dup', 0) + 1; continue
        kept_al.append(al)
        used_seed.add(a); used_seed.add(b)
        mz = L["emz"]
        mags = [mz[E[x]] - 2.5 * math.log10(D[x, 4]) for x in (a, b) if np.isfinite(mz[E[x]]) and D[x, 4] > 0]
        out.append(dict(state=st, z=h[8], zall=zv, npos=h[6], nother=h[7], sn2=(float(sn[a]), float(sn[b])), r=r,
                        mag=float(np.median(mags)) if mags else None))
    log("stack rejects", REJ, "hits used", n_h)
    log(f"stack: {len(out)} candidates; top z {[round(c['z'], 1) for c in out[:10]]}")
    return out


def p_stack(c):
    """Modest confidence: below the linker's confirmed bodies, above its 3-point junk only for strong stacks."""
    z = c["z"] + 0.2 * (c["npos"] - 2)
    return float(0.6 / (1 + math.exp(-2.0 * (z - 7.6))))
