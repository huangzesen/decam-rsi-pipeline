"""Catalogue-level shift-and-stack within one night (gen 2).

For a night with >= 3 processed exposures, every usable >= 3 sigma detection is shifted back to the night's mid-time
along each rate (vx, vy) of a grid restricted to what a body beyond 25 au can do (reflex motion of the observer for
1/gamma in [25, 150] au, +- the bound-orbit speed), and hashed into ~1.1" cells (two offset grids).  A cell holding
detections from >= 3 distinct exposures with summed S/N (root-sum-square) >= thr is a faint tracklet that the
pair-based seeding misses (e.g. 3-4 detections of S/N 3-3.5 out of 8 exposures).  Each such cell contributes its
highest-S/N member as a linker seed.  Runs in torch on the CPU.
"""
import math, time
import numpy as np

GM = 2.959122082855911e-4
RAD2AS = 206264.806


def _dev():
    import torch
    if torch.backends.mps.is_available():
        return torch, "mps"
    if torch.cuda.is_available():
        return torch, "cuda"
    return torch, "cpu"


def rate_grid(vof, T_h, dv_min=0.25, nmax=700):
    """Rates (arcsec/h) in the tangent frame. vof: observer velocity in frame (au/day)."""
    dv = max(1.4 / max(T_h, 1e-3), dv_min)
    gam = np.linspace(1 / 150.0, 1 / 25.0, 40)
    gp = gam / (1 - gam)                                    # ~1/(observer distance)
    cen = -gp[:, None] * vof[None, :2] * RAD2AS / 24.0      # arcsec/h
    rad = np.sqrt(2 * GM * gam) * gp * RAD2AS / 24.0 + 0.3
    while True:
        lo = (cen - rad[:, None]).min(0) - dv
        hi = (cen + rad[:, None]).max(0) + dv
        gx = np.arange(lo[0], hi[0] + dv, dv)
        gy = np.arange(lo[1], hi[1] + dv, dv)
        V = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        d = np.linalg.norm(V[:, None, :] - cen[None], axis=2) - rad[None] - 0.5 * dv
        V = V[(d <= 0).any(1)]
        if len(V) <= nmax:
            return V, dv
        dv *= 1.25


def night_tracklets(x, y, dt_h, eidx, sn, vof, thr=6.0, cell=1.1, cap=1500, min_exp=3, tlimit=20.0):
    """x, y arcsec (tangent frame), dt_h hours from night mid, eidx local exposure index (< 64), sn S/N.
    Returns indices (into the inputs) of seed representatives, best first, and their tracklet S/N."""
    t0 = time.time()
    n = len(x)
    if n == 0 or len(np.unique(eidx)) < min_exp:
        return np.zeros(0, int), np.zeros(0)
    T_h = float(dt_h.max() - dt_h.min())
    V, dv = rate_grid(vof, T_h)
    import torch
    # CPU: torch.unique/searchsorted on mps gave wrong (empty) results and was slower -> never mps.
    # CUDA (grader A100): run there, fall back to the CPU on any error.
    if torch.cuda.is_available():
        try:
            return _run(torch, "cuda", x, y, dt_h, eidx, sn, V, thr, cell, cap, min_exp, t0, tlimit)
        except Exception as ex:
            print("hough cuda failed, cpu fallback:", ex, flush=True)
    torch.set_num_threads(_nthreads())
    return _run(torch, "cpu", x, y, dt_h, eidx, sn, V, thr, cell, cap, min_exp, t0, tlimit)


def _nthreads():
    import os
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 4
    try:
        omp = int(os.environ.get("OMP_NUM_THREADS", "0"))
    except Exception:
        omp = 0
    return max(1, min(8, n, omp if omp > 0 else 8))


def _run(torch, dev, x, y, dt_h, eidx, sn, V, thr, cell, cap, min_exp, t0, tlimit):
    n = len(x)
    f32 = torch.float32
    X = torch.tensor(x - x.mean(), dtype=f32, device=dev)
    Y = torch.tensor(y - y.mean(), dtype=f32, device=dev)
    Tt = torch.tensor(dt_h, dtype=f32, device=dev)
    Ei = torch.tensor(eidx.astype(np.int64), device=dev)
    S2 = torch.tensor((sn ** 2).astype(np.float32), device=dev)
    SN = torch.tensor(sn.astype(np.float32), device=dev)
    K = 1 << 16
    best = {}   # point idx -> best tracklet S/N
    bs = max(1, int(8e6 // max(n, 1)))
    Vt = torch.tensor(V, dtype=f32, device=dev)
    for off in (0.0, 0.5):
        for r0 in range(0, len(V), bs):
            if time.time() - t0 > tlimit:
                break
            v = Vt[r0:r0 + bs]
            nb = len(v)
            xs = X[None] - v[:, 0:1] * Tt[None]
            ys = Y[None] - v[:, 1:2] * Tt[None]
            cx = torch.floor(xs / cell + off).long() + K // 2
            cy = torch.floor(ys / cell + off).long() + K // 2
            rid = torch.arange(nb, device=dev)[:, None]
            key = (rid * K + cx) * K + cy                                  # (nb, n)
            kf = key.reshape(-1)
            ke = kf * 64 + Ei.repeat(nb)
            # distinct exposures per cell
            uke = torch.unique(ke)
            uk, cnt = torch.unique_consecutive(uke // 64, return_counts=True)
            good = uk[cnt >= min_exp]
            if good.numel() == 0:
                continue
            # points in good cells
            pos = torch.searchsorted(good, kf)
            pos = torch.clamp(pos, max=good.numel() - 1)
            inb = good[pos] == kf
            pidx = torch.nonzero(inb).squeeze(1)
            if pidx.numel() == 0:
                continue
            cid = pos[pidx]
            pt = pidx % n
            ssum = torch.zeros(good.numel(), dtype=f32, device=dev).index_add_(0, cid, S2[pt])
            stot = torch.sqrt(ssum)
            okc = stot >= thr
            if not okc.any():
                continue
            # representative: highest-S/N member of each passing cell
            sel = okc[cid]
            cid, pt = cid[sel], pt[sel]
            mx = torch.full((good.numel(),), -1.0, dtype=f32, device=dev).scatter_reduce_(0, cid, SN[pt], "amax")
            isrep = SN[pt] >= mx[cid]
            rp = pt[isrep].cpu().numpy()
            rs = stot[cid[isrep]].cpu().numpy()
            for a, b in zip(rp.tolist(), rs.tolist()):
                if b > best.get(a, 0.0):
                    best[a] = b
    if not best:
        return np.zeros(0, int), np.zeros(0)
    idx = np.fromiter(best.keys(), int)
    s = np.fromiter(best.values(), float)
    o = np.argsort(-s)[:cap]
    return idx[o], s[o]
