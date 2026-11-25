"""Gen 6a: pixel-level within-night shift-and-stack -> synthetic seed detections for the linker.

For each night with >= NMIN exposures that have stored S/N maps, the maps are resampled onto one tangent-plane
mosaic (stack.build_mosaics, 1"/px), stationary catalogue sources are masked, per-exposure S/N is clipped to
[-3, 6], and for every rate of the hough rate grid (bodies at 25-150 au) the exposures are shifted (integer pixels)
back to the night's mid-time and summed: z = sum / sqrt(n).  Peaks of max-over-rates z >= THR become synthetic
detections on the exposure nearest mid-night (position propagated along the peak's rate), which the linker uses as
extra seeds for that night.  This reaches bodies a factor ~sqrt(n) fainter than single-exposure seeds (weight-2
bodies in nights with many exposures), which the catalogue-level hough (needs >= 3 detections >= 3 sigma) misses.
"""
import math, os, time
import numpy as np
from orbit import ARC, log

PS = 1.0 * ARC


def run(L, deadline):
    """Returns list of (e_ref, unit_vector(3,), z) synthetic detections."""
    import torch
    import stack, hough
    dev = stack._dev(torch)
    FP, F, et, enight, E, u, sn, st, gi, vobs, done = (L[k] for k in
        ("FP", "F", "et", "enight", "E", "u", "sn", "st", "gi", "vobs", "done"))
    NMIN = int(os.environ.get("TNO_NS_MIN", 4))
    THR = float(os.environ.get("TNO_NS_THR", 5.6))
    CAPN = 1500
    CLIP = float(os.environ.get("TNO_NS_CLIP", 10.0))
    mexp = np.array([e for e in done if FP.has(e)], int)
    if len(mexp) == 0:
        return []
    nights = np.unique(enight[mexp])
    cnt = {n: int((enight[mexp] == n).sum()) for n in nights}
    nights = [n for n in sorted(nights, key=lambda q: -cnt[q]) if cnt[n] >= NMIN]
    out = []
    # static catalogue sources (all nights) in the frame, for masking
    sidx = np.flatnonzero(st & (sn >= 4))
    us_ = u[sidx] @ F.T
    sth = us_[:, :2] / us_[:, 2:3]
    ssn = sn[sidx]
    for ni, nn in enumerate(nights):
        ndl = time.time() + (deadline - time.time()) / (len(nights) - ni)   # equal share of the remaining time
        if time.time() > deadline - 3:
            log("nightstack: deadline")
            break
        t1 = time.time()
        ee = mexp[enight[mexp] == nn]
        ee = ee[np.argsort(et[ee])]
        if len(ee) > 32:
            ee = ee[np.linspace(0, len(ee) - 1, 32).astype(int)]
        tm = 0.5 * (et[ee].min() + et[ee].max())
        dth = (et[ee] - tm) * 24.0
        T_h = float(dth.max() - dth.min())
        if T_h <= 0:
            continue
        # area: usable detections of this night
        m = gi[np.isin(E[gi], ee)]
        if len(m) < 100:
            continue
        uf = u[m] @ F.T
        th = uf[:, :2] / uf[:, 2:3]
        lo = np.percentile(th, 0.2, axis=0) - 0.003
        hi = np.percentile(th, 99.8, axis=0) + 0.003
        x0, y0 = int(math.floor(lo[0] / PS)), int(math.floor(lo[1] / PS))
        W, H = int(math.ceil(hi[0] / PS)) - x0, int(math.ceil(hi[1] / PS)) - y0
        while W * H * len(ee) * 2 > 8e9 and len(ee) > NMIN:
            ee = ee[::2] if len(ee) >= 2 * NMIN else ee[:-1]
        if len(ee) < NMIN:
            continue
        tm = 0.5 * (et[ee].min() + et[ee].max())
        dth = (et[ee] - tm) * 24.0
        T_h = float(dth.max() - dth.min())
        lim = (x0, y0, W, H)
        mos = stack.build_mosaics(torch, dev, FP, list(ee), F, lim)
        # static mask: catalogue stationary sources, radius ~2" (bright: larger)
        mask = torch.zeros((1, 1, H, W), dtype=torch.float16, device=dev)
        big = torch.zeros((1, 1, H, W), dtype=torch.float16, device=dev)
        px = np.round(sth[:, 0] / PS - x0).astype(np.int64)
        py = np.round(sth[:, 1] / PS - y0).astype(np.int64)
        ins = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if ins.any():
            k = torch.from_numpy(py[ins] * W + px[ins]).to(dev)
            mask.view(-1)[k] = 1.0
            kb = ins & (ssn > 60)
            if kb.any():
                big.view(-1)[torch.from_numpy(py[kb] * W + px[kb]).to(dev)] = 1.0
        mask = torch.nn.functional.max_pool2d(mask.float(), 5, 1, 2)
        if bool((big > 0).any()):
            mask = torch.maximum(mask, torch.nn.functional.max_pool2d(big.float(), 15, 1, 7))
        mask = mask[0, 0] > 0
        stackin = []
        for e in ee:
            v = mos[e][0, 0]
            v = torch.where(mask, torch.full_like(v, float("nan")), v)
            stackin.append(torch.clamp(v, -3.0, CLIP))
        del mos
        valid = [torch.isfinite(v) for v in stackin]
        vals = [torch.nan_to_num(v, nan=0.0) for v in stackin]
        del stackin
        vof = F @ vobs[ee].mean(0)
        V, dv = hough.rate_grid(vof, T_h, dv_min=0.3, nmax=500)
        if os.environ.get("TNO_NS_FLIP"):      # dev: sign test
            V = -V
        best = torch.full((H, W), -1e9, dtype=torch.float32, device=dev)
        bidx = torch.zeros((H, W), dtype=torch.int16, device=dev)
        nmin_s = max(NMIN - 1, int(math.ceil(0.6 * len(ee))))
        S = torch.zeros((H, W), dtype=torch.float32, device=dev)
        N = torch.zeros((H, W), dtype=torch.float32, device=dev)
        nr = 0
        for ri, (vx, vy) in enumerate(V):
            if time.time() > ndl:
                break
            S.zero_(); N.zero_()
            for k in range(len(ee)):
                # stacked(x) = sum_k M_k(x + v dt_k)
                sx = int(round(vx * dth[k])); sy = int(round(vy * dth[k]))
                xa, xb = max(0, -sx), min(W, W - sx)
                ya, yb = max(0, -sy), min(H, H - sy)
                if xb <= xa or yb <= ya:
                    continue
                S[ya:yb, xa:xb] += vals[k][ya + sy:yb + sy, xa + sx:xb + sx]
                N[ya:yb, xa:xb] += valid[k][ya + sy:yb + sy, xa + sx:xb + sx]
            z = torch.where(N >= nmin_s, S / torch.sqrt(torch.clamp(N, min=1.0)), torch.full_like(S, -1e9))
            upd = z > best
            best = torch.where(upd, z, best)
            bidx = torch.where(upd, torch.full_like(bidx, ri), bidx)
            nr += 1
        del vals, valid, S, N
        bp = torch.nn.functional.max_pool2d(best[None, None], 7, 1, 3)[0, 0]
        pk = (best >= THR) & (best >= bp)
        idx = torch.nonzero(pk.view(-1)).view(-1)
        if len(idx) > 4 * CAPN:
            vv = best.view(-1)[idx]
            idx = idx[torch.topk(vv, 4 * CAPN).indices]
        idx_c = idx.cpu().numpy()
        if len(idx_c) == 0:
            log(f"nightstack night {nn}: {len(ee)} exp, {nr}/{len(V)} rates, 0 peaks ({time.time() - t1:.1f}s)")
            continue
        iy, ix = idx_c // W, idx_c % W
        bnp = best.cpu().numpy()
        rv = bidx.cpu().numpy()[iy, ix].astype(int)
        zz = bnp[iy, ix]
        # sub-pixel quadratic peak
        def q(a, b, c):
            den = a - 2 * b + c
            return np.where(np.abs(den) > 1e-6, np.clip(0.5 * (a - c) / den, -0.5, 0.5), 0.0)
        ixc, iyc = np.clip(ix, 1, W - 2), np.clip(iy, 1, H - 2)
        fx = q(bnp[iyc, ixc - 1], bnp[iyc, ixc], bnp[iyc, ixc + 1])
        fy = q(bnp[iyc - 1, ixc], bnp[iyc, ixc], bnp[iyc + 1, ixc])
        fx = np.nan_to_num(fx); fy = np.nan_to_num(fy)
        kref = int(np.argmin(np.abs(dth)))
        eref = int(ee[kref])
        X = (ix + fx + x0) * PS + V[rv, 0] * dth[kref] * ARC
        Y = (iy + fy + y0) * PS + V[rv, 1] * dth[kref] * ARC
        vf = np.stack([X, Y, np.ones_like(X)], 1)
        vf /= np.linalg.norm(vf, axis=1, keepdims=True)
        uu = vf @ F            # frame -> ICRS (uf = F u)
        # static-source veto: the same sky position on exposures of *other* nights must look like blank sky
        oth = mexp[enight[mexp] != nn]
        if len(oth) > 12:
            oth = oth[np.linspace(0, len(oth) - 1, 12).astype(int)]
        keep = np.ones(len(uu), bool)
        if len(oth) >= 2:
            fs = np.stack([FP.sample(int(e), uu) for e in oth], 1)
            nfin = np.isfinite(fs).sum(1)
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(nfin[:, None] > 0, fs, 0.0), 1)
                hi_ = np.nansum(fs > 2.5, 1) / np.maximum(nfin, 1)
            keep = (nfin >= 2) & (med < 1.0) & (hi_ < 0.3)
        nk0 = len(uu)
        nk1 = int(keep.sum())
        # sources bright enough for the single-exposure catalogue (S/N >= 5 within 2.5" on the reference exposure,
        # static or not) are already handled by the normal seeds; their clipped plateaus also give offset peaks
        from scipy.spatial import cKDTree
        cm = np.flatnonzero((E == eref) & (sn >= 5))
        if len(cm) and keep.any() and not os.environ.get("TNO_NS_NOCAT"):
            ki = np.flatnonzero(keep)
            dd_, _ = cKDTree(u[cm]).query(uu[ki], distance_upper_bound=2.5 * ARC)
            keep[ki[np.isfinite(dd_)]] = False
        sel_ = np.flatnonzero(keep)
        sel_ = sel_[np.argsort(-zz[sel_])]
        if len(sel_) > 1:      # de-duplicate: drop peaks within 2" of a higher one
            tr_ = cKDTree(uu[sel_])
            drop = np.zeros(len(sel_), bool)
            for a_, b_ in tr_.query_pairs(2.0 * ARC, output_type="ndarray"):
                drop[max(a_, b_)] = True
            sel_ = sel_[~drop]
        sel_ = sel_[:CAPN]
        for a_ in sel_:
            out.append((eref, uu[a_], float(zz[a_])))
        zz = zz[sel_]
        log(f"nightstack veto: {nk0} peaks -> static {nk1} -> catalogue/dup {len(sel_)}")
        log(f"nightstack night {nn}: {len(ee)} exp, {nr}/{len(V)} rates, {W}x{H} px, {len(zz)} peaks "
            f"(top z {np.sort(zz)[::-1][:3].round(1).tolist()}) ({time.time() - t1:.1f}s)")
        if dev == "mps":
            torch.mps.empty_cache()
        elif dev == "cuda":
            torch.cuda.empty_cache()
    return out
