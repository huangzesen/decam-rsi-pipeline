"""Straight GPU port (PyTorch) of link_cpu.py: same transform, pairs, cells and verification, same output.

Usage: python link_gpu.py DATA.npz [link_cpu options] [--device cuda] [--batch 96] [--fp32]
Pairs: brute force over exposure pairs. For each hypothesis the exposures are padded into one (n_exp, n_max, 2)
tensor; exposure pairs that are on different nights, within the window, and whose footprints (in the transformed
frame) can be joined by a bound orbit are compared in batches of `batch` (all n_max x n_max distances at once).
Pair states and cell keys are computed on the GPU and the keys sorted on the GPU. Verification (--verify gpu, the
default) also runs on the GPU: all candidates of one hypothesis are fitted at once as segments of one flat array
(2x2 normal equations per candidate), clipped as link_cpu.verify does, and extended by comparing each survivor's
predicted position with every detection of every exposure. --verify host sends the candidates to link_cpu.verify.
Two stages (BENCHMARK.md, 10-28): --gdot-tol 0 --refine --polish runs the gamma-dot = 0 grid, fits a gamma-dot
correction per candidate in verification (--refine), and fits (gamma, gamma-dot) per merged linkage before extending
it again (--polish). On a one-season window it gives the default grid's >= 5-night linkages for ~1/12 of the cost.
"""
import json, time
import numpy as np
import torch
import link_cpu as L
from orbits import GM, C_AU, ARCSEC

OFF = 1 << 15


def setup(W, dev):
    """Exposure-major layout: detections of each exposure contiguous; padding index -1."""
    exp = W["exp"]
    o = np.argsort(exp, kind="stable")
    ue, start, cnt = np.unique(exp[o], return_index=True, return_counts=True)
    nmax = int(cnt.max())
    pad = -np.ones((len(ue), nmax), np.int64)
    for k, (s, c) in enumerate(zip(start, cnt)):
        pad[k, :c] = o[s:s + c]
    et = W["exp_t"][ue]
    night = W["exp_night"][ue]
    G = dict(pad=torch.as_tensor(pad, device=dev), valid=torch.as_tensor(pad >= 0, device=dev), et=et, night=night,
             theta=torch.as_tensor(W["theta"], device=dev), obs=torch.as_tensor(W["obs"], device=dev),
             t=torch.as_tensor(W["t"], device=dev), n_exp=len(ue), nmax=nmax,
             det_night=torch.as_tensor(W["night"] - W["night"].min(), device=dev), et_d=torch.as_tensor(et, device=dev))
    th = torch.as_tensor(W["theta"], device=dev)[G["pad"].clamp(min=0)]
    G["Theta"] = torch.where(G["valid"][..., None], th, torch.full_like(th, float("nan")))   # (n_exp, nmax, 2)
    G["eobs"] = torch.as_tensor(W["exp_obs"][ue], device=dev)                                  # (n_exp, 3)
    return G



def seed_layout(G, keep, dev):
    """--seed-mask: a second per-exposure layout holding only the detections that may seed a candidate (keep, per
    window detection), aligned with G's exposures. The pair search and clustering run on it; verification still
    extends against every detection, so a cut that drops one detection of a real object costs it a night only when
    too few of its detections are left to seed it."""
    pad = G["pad"].cpu().numpy()
    k = np.where(pad >= 0, keep[np.maximum(pad, 0)], False)
    cnt = k.sum(1)
    ps = -np.ones((pad.shape[0], max(1, int(cnt.max()))), np.int64)
    for e in range(pad.shape[0]):
        ps[e, :cnt[e]] = pad[e][k[e]]
    G["pad_seed"] = torch.as_tensor(ps, device=dev)
    G["valid_seed"] = G["pad_seed"] >= 0
    G["nmax_seed"] = ps.shape[1]

def seg_fit(cid, nc, phi, u, t, w):
    """Least-squares line phi = alpha u + alpha-dot t per candidate (segments `cid`, weights w); residual per entry."""
    w = w.to(phi.dtype)
    z = lambda *s: torch.zeros(nc, *s, dtype=phi.dtype, device=phi.device)
    Suu, Sut, Stt = z().index_add_(0, cid, w * u * u), z().index_add_(0, cid, w * u * t), z().index_add_(0, cid, w * t * t)
    bu = z(2).index_add_(0, cid, (w * u)[:, None] * phi)
    bt = z(2).index_add_(0, cid, (w * t)[:, None] * phi)
    D = (Suu * Stt - Sut * Sut)[:, None]
    al = (Stt[:, None] * bu - Sut[:, None] * bt) / D
    ad = (Suu[:, None] * bt - Sut[:, None] * bu) / D
    res = phi - u[:, None] * al[cid] - t[:, None] * ad[cid]
    return al, ad, torch.sqrt((res * res).sum(1))


def seg_fit5(cid, nc, phi, u, t, th, w, ridge=0.0, gd=0.0):
    """--refine: the line plus a gamma-dot correction dg. phi is exactly linear in gamma-dot (d phi / d gamma-dot =
    theta t'), so phi_observed = alpha u + alpha-dot t - dg theta t is a 5-parameter linear least-squares fit per
    candidate. ridge * (gd + dg)^2 is added to the squared residuals: a Gaussian prior on the total gamma-dot, so a
    gamma-dot the data do not pin down stays near 0. Returns alpha, alpha-dot (nc, 2), dg (nc,), residual per entry."""
    w = w.to(phi.dtype)
    z0 = torch.zeros_like(u)
    rx = torch.stack([u, t, z0, z0, -th[:, 0] * t], 1)
    ry = torch.stack([z0, z0, u, t, -th[:, 1] * t], 1)
    M = torch.zeros(nc, 25, dtype=phi.dtype, device=phi.device).index_add_(
        0, cid, w[:, None] * (rx[:, :, None] * rx[:, None, :] + ry[:, :, None] * ry[:, None, :]).reshape(-1, 25)).reshape(nc, 5, 5)
    b = torch.zeros(nc, 5, dtype=phi.dtype, device=phi.device).index_add_(0, cid, w[:, None] * (rx * phi[:, :1] + ry * phi[:, 1:]))
    M[:, 4, 4] += ridge
    b[:, 4] -= ridge * gd
    d = torch.diagonal(M, dim1=1, dim2=2)
    M = M + torch.diag_embed(d * 1e-10 + 1e-300)               # keeps degenerate (e.g. 3-point) systems solvable
    p = torch.linalg.solve(M, b[..., None])[..., 0]
    res = torch.stack([phi[:, 0] - (rx * p[cid]).sum(1), phi[:, 1] - (ry * p[cid]).sum(1)], 1)
    return torch.stack([p[:, 0], p[:, 2]], 1), torch.stack([p[:, 1], p[:, 3]], 1), p[:, 4], torch.sqrt((res * res).sum(1))


def seg_count(cid, nc, m=None):
    c = cid if m is None else cid[m]
    return torch.bincount(c, minlength=nc)


def seg_nights(cid, nc, night, m):
    if not bool(m.any()):
        return torch.zeros(nc, dtype=torch.int64, device=cid.device)
    nr = int(night.max()) + 1
    k = torch.unique(cid[m] * nr + night[m])
    return torch.bincount(k // nr, minlength=nc)


def index_extender(G, P, te, ue, fe, gxe, tol, a, dev):
    """--extend index: the survivors' extension through a cell index instead of a comparison with every detection of
    every exposure. The detections of this hypothesis (P, transformed frame) are binned per exposure into square cells
    of side 1.1 tol + margin and sorted by cell key; each (survivor, exposure) query looks only at the 3 x 3 cells around
    its search point. Without --refine that is the prediction. With --refine a detection P moves to
    P + DG theta te, theta = (P - gamma x_E) / f_e, which is linear in P: (1 + c) P - c gamma x_E with c = DG te / f_e.
    It lies within tol of the prediction exactly when P lies within tol / (1 + c) of
    P* = (pred + c gamma x_E) / (1 + c), so P* is the search point and that radius must fit inside a cell; the rare
    query whose radius does not (c < -0.13) is compared with every detection of its exposure instead. Every hit is
    checked with the brute-force expression. Same contract as the brute-force extend: per query the nearest detection
    within tol (ties: the first found)."""
    ne, nmax = P.shape[0], P.shape[1]
    h = 1.1 * tol + 0.05 * ARCSEC
    Pf = P.reshape(-1, 2)
    fi = torch.nonzero(torch.isfinite(Pf).all(1), as_tuple=True)[0]      # flat (exposure, slot) of real detections
    lo = Pf[fi].min(0).values - 2 * h
    span = Pf[fi].max(0).values + 2 * h - lo
    nx, ny = int(span[0] / h) + 3, int(span[1] / h) + 3
    cx = ((Pf[fi, 0] - lo[0]) / h).long(); cy = ((Pf[fi, 1] - lo[1]) / h).long()
    Ks, o = torch.sort(((fi // nmax) * ny + cy) * nx + cx)
    fo = fi[o]
    Th = G["Theta"].reshape(-1, 2)
    flat_pad = G["pad"].reshape(-1)
    offs = torch.tensor([[dx, dy] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], device=dev)
    qchunk = max(1, int(a.extend_chunk // ne))
    empty = torch.zeros(0, dtype=torch.int64, device=dev)

    def extend(A0, A1, DG):
        add_c, add_d = [], []
        for s0 in range(0, len(A0), qchunk):
            pred = (ue[None, :, None] * A0[s0:s0 + qchunk, None, :] + te[None, :, None] * A1[s0:s0 + qchunk, None, :]).reshape(-1, 2)
            c = len(pred) // ne
            qe = torch.arange(ne, device=dev).repeat(c)                     # exposure of each query
            qs = torch.arange(c, device=dev).repeat_interleave(ne) + s0     # survivor of each query
            q, near = pred, torch.ones(len(pred), dtype=torch.bool, device=dev)
            if a.refine:
                c = DG[qs] * te[qe] / fe[qe]
                q = (pred + c[:, None] * gxe[qe]) / (1 + c)[:, None]
                near = (1 + c) * h >= tol                                    # search radius tol / (1 + c) within a cell
                far = torch.nonzero(~near, as_tuple=True)[0]
                if len(far):                                                 # brute force for those queries
                    Pr = P[qe[far]] + DG[qs[far], None, None] * G["Theta"][qe[far]] * te[qe[far], None, None]
                    v, k = torch.nan_to_num(((Pr - pred[far, None, :]) ** 2).sum(-1), nan=float("inf")).min(-1)
                    w = v <= tol * tol
                    add_c.append(qs[far[w]]); add_d.append(G["pad"][qe[far[w]], k[w]])
            ix = torch.floor((q[:, 0] - lo[0]) / h).long(); iy = torch.floor((q[:, 1] - lo[1]) / h).long()
            qi = torch.nonzero(near & (ix >= 1) & (ix < nx - 1) & (iy >= 1) & (iy < ny - 1), as_tuple=True)[0]
            if len(qi) == 0:
                continue
            k9 = ((qe[qi, None] * ny + iy[qi, None] + offs[None, :, 1]) * nx + ix[qi, None] + offs[None, :, 0]).reshape(-1)
            l9 = torch.searchsorted(Ks, k9); cnt = torch.searchsorted(Ks, k9, right=True) - l9
            tot = int(cnt.sum())
            if tot == 0:
                continue
            rep = torch.repeat_interleave(torch.arange(len(k9), device=dev), cnt)
            pos = l9[rep] + torch.arange(tot, device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
            hq, hf = qi[rep // 9], fo[pos]
            pr = Pf[hf]
            if a.refine:
                pr = pr + DG[qs[hq], None] * Th[hf] * te[qe[hq], None]
            d2 = ((pr - pred[hq]) ** 2).sum(1)
            m = d2 <= tol * tol
            hq, hf, d2 = hq[m], hf[m], d2[m]
            if len(hq) == 0:
                continue
            best = torch.full((len(pred),), float("inf"), dtype=d2.dtype, device=dev).scatter_reduce_(0, hq, d2, "amin")
            m = d2 <= best[hq]
            hq, hf = hq[m], hf[m]
            uq, inv = torch.unique(hq, return_inverse=True)                 # one detection per query (ties: the first)
            first = torch.full((len(uq),), len(hq), dtype=torch.int64, device=dev).scatter_reduce_(
                0, inv, torch.arange(len(hq), device=dev), "amin")
            add_c.append(qs[hq[first]]); add_d.append(flat_pad[hf[first]])
        return (add_c or [empty]), (add_d or [empty])
    return extend


def verify_gpu(W, G, P, phi, tp, u, g, gd, cid, det, a):
    """GPU version of link_cpu.verify. cid, det: flat (candidate, detection) entries, unique, sorted by candidate."""
    dev = phi.device
    tol = a.fit_tol * ARCSEC
    if len(cid) == 0:
        return []
    # drop duplicate candidates (the two shifted grids find the same sets): signature (size, sum, sum of squares, hash)
    nc = int(cid.max()) + 1
    sig = torch.zeros(nc, 4, dtype=torch.int64, device=dev)
    sig[:, 0].index_add_(0, cid, torch.ones_like(det))
    sig[:, 1].index_add_(0, cid, det)
    sig[:, 2].index_add_(0, cid, det * det)
    sig[:, 3].index_add_(0, cid, (det * 2654435761) & 0x7FFFFFFF)
    present = sig[:, 0] > 0
    _, inv = torch.unique(sig, dim=0, return_inverse=True)
    first = torch.full((int(inv.max()) + 1,), nc, dtype=torch.int64, device=dev).scatter_reduce_(0, inv, torch.arange(nc, device=dev), "amin")
    keepc = present & (first[inv] == torch.arange(nc, device=dev))
    m = keepc[cid]
    cid, det = cid[m], det[m]
    newid = torch.cumsum(keepc.long(), 0) - 1
    cid = newid[cid]
    nc = int(keepc.sum())
    rows = lambda k: torch.stack([phi[:, 0][k], phi[:, 1][k]], 1)    # 1-D gathers (see states() in link_gamma_gpu)
    trows = lambda k: torch.stack([G["theta"][:, 0][k], G["theta"][:, 1][k]], 1)
    ph, t_, u_, ni = rows(det), tp[det], u[det], G["det_night"][det]
    th_ = trows(det) if a.refine else None
    # prior on the total gamma-dot: sd = vr_frac * n_para (the grid's range), astrometric sigma per coordinate
    ridge = (a.sigma * ARCSEC / (a.vr_frac * np.sqrt(2 * GM * g ** 3))) ** 2

    def fit(cid, nc, ph, u_, t_, th_, ni, w):
        al, ad, r = seg_fit(cid, nc, ph, u_, t_, w)
        dg = torch.zeros(nc, dtype=ph.dtype, device=ph.device)
        if a.refine:              # gamma-dot only for candidates on >= refine_nights nights (3 nights fit it exactly)
            a5, d5, g5, r5 = seg_fit5(cid, nc, ph, u_, t_, th_, w, ridge, gd)
            f5 = seg_nights(cid, nc, ni, w) >= a.refine_nights
            al, ad, dg = torch.where(f5[:, None], a5, al), torch.where(f5[:, None], d5, ad), torch.where(f5, g5, dg)
            r = torch.where(f5[cid], r5, r)
        return al, ad, dg, r
    # clip up to three times, as link_cpu.verify
    act = torch.ones(len(cid), dtype=torch.bool, device=dev)
    done = torch.zeros(nc, dtype=torch.bool, device=dev)
    cnt_all = seg_count(cid, nc)
    start = torch.cumsum(cnt_all, 0) - cnt_all
    for _ in range(3):
        n_act = seg_count(cid, nc, act)
        done |= n_act < 3
        _, _, _, r = fit(cid, nc, ph, u_, t_, th_, ni, act)
        rmax = torch.zeros(nc, dtype=r.dtype, device=dev).scatter_reduce_(0, cid, torch.where(act, r, 0.0), "amax")
        done |= rmax <= tol
        if bool(done.all()):
            break
        rk = torch.where(act, r, torch.full_like(r, float("inf")))
        o = torch.sort(rk, stable=True).indices
        o = o[torch.sort(cid[o], stable=True).indices]
        med = rk[o[(start + n_act // 2).clamp(max=len(cid) - 1)]]
        thr = torch.clamp(3 * med, min=tol)
        clip = ~done[cid] & (r > thr[cid])
        act &= ~clip
    n_act = seg_count(cid, nc, act)
    ok = (n_act >= 3) & (seg_nights(cid, nc, ni, act) >= 3)
    al, ad, dg, r = fit(cid, nc, ph, u_, t_, th_, ni, act)
    rmax = torch.zeros(nc, dtype=r.dtype, device=dev).scatter_reduce_(0, cid, torch.where(act, r, 0.0), "amax")
    ok &= rmax <= tol
    surv = torch.nonzero(ok, as_tuple=True)[0]
    if len(surv) == 0:
        return []
    # extend every survivor to every exposure: nearest detection within tol
    te = G["et_d"] - 1.0 / (C_AU * g)
    ue = 1.0 - 0.5 * GM * g ** 3 * te * te
    ne, nmax = P.shape[0], P.shape[1]
    ch = max(1, int(2e8 // (ne * nmax * 2 * 8)))

    def extend(A0, A1, DG):
        add_c, add_d = [], []
        for s0 in range(0, len(A0), ch):
            pred = ue[None, :, None] * A0[s0:s0 + ch, None, :] + te[None, :, None] * A1[s0:s0 + ch, None, :]
            if a.refine:                                      # detections moved to each survivor's gamma-dot
                Pr = P[None] + DG[s0:s0 + ch, None, None, None] * G["Theta"][None] * te[None, :, None, None]
                d2 = ((Pr - pred[:, :, None, :]) ** 2).sum(-1)
            else:
                d2 = ((P[None] - pred[:, :, None, :]) ** 2).sum(-1)
            d2 = torch.nan_to_num(d2, nan=float("inf"))
            v, k = d2.min(-1)
            cc, ee = torch.nonzero(v <= tol * tol, as_tuple=True)
            add_c.append(cc + s0); add_d.append(G["pad"][ee, k[cc, ee]])
        return add_c, add_d
    if a.extend == "index":
        fe = 1.0 + gd * te - 0.5 * GM * g ** 3 * te * te - g * G["eobs"][:, 2]
        extend = index_extender(G, P, te, ue, fe, g * G["eobs"][:, :2], tol, a, dev)
    remap = torch.full((nc,), -1, dtype=torch.int64, device=dev)
    remap[surv] = torch.arange(len(surv), device=dev)
    m = act & (remap[cid] >= 0)
    c2, d2_ = remap[cid[m]], det[m]
    A0, A1, DG = al[surv], ad[surv], dg[surv]
    N = len(phi)
    ns = len(surv)
    alive = torch.ones(ns, dtype=torch.bool, device=dev)
    # --refine: extend, refit (more nights pin gamma-dot down), extend again with the new fit; refine_iter rounds
    for it in range(a.refine_iter if a.refine else 1):
        add_c, add_d = extend(A0, A1, DG)
        key = torch.unique(torch.cat([c2] + add_c) * N + torch.cat([d2_] + add_d))
        c2, d2_ = key // N, key % N
        ph, t_, u_, ni = rows(d2_), tp[d2_], u[d2_], G["det_night"][d2_]
        th_ = trows(d2_) if a.refine else None
        _, _, _, r = fit(c2, ns, ph, u_, t_, th_, ni, torch.ones_like(c2, dtype=torch.bool))
        keep = r <= tol
        nn = seg_nights(c2, ns, ni, keep)
        good = nn >= a.minnights
        keep &= good[c2]
        al, ad, dg, r = fit(c2, ns, ph, u_, t_, th_, ni, keep)
        if it + 1 < (a.refine_iter if a.refine else 1):      # next round starts from the kept detections only
            alive = good
            c2, d2_ = c2[keep], d2_[keep]
            A0, A1, DG = al, ad, dg
    if a.refine:
        good &= alive
        keep &= good[c2]
    sq =torch.zeros(ns, dtype=r.dtype, device=dev).index_add_(0, c2, torch.where(keep, r * r, 0.0))
    nk = seg_count(c2, ns, keep)
    out = []
    gi = torch.nonzero(good, as_tuple=True)[0].cpu().numpy()
    if len(gi) == 0:
        return out
    c2h, dh, kh = c2.cpu().numpy(), d2_.cpu().numpy(), keep.cpu().numpy()
    alh, adh, sqh, nkh, nnh, dgh = (x.cpu().numpy() for x in (al, ad, sq, nk, nn, dg))
    dk, ck = dh[kh], c2h[kh]
    brk = np.searchsorted(ck, gi), np.searchsorted(ck, gi, side="right")
    for q, s, e in zip(gi, *brk):
        out.append(dict(det=dk[s:e], nights=int(nnh[q]), rms=float(np.sqrt(sqh[q] / nkh[q]) / ARCSEC), gamma=g,
                        gdot=float(gd + dgh[q]),
                        state=[float(alh[q, 0]), float(adh[q, 0]), float(alh[q, 1]), float(adh[q, 1])]))
    return out


def polish(W, G, links, a, dev):
    """Stage 2 (--polish): fit (gamma, gamma-dot) per merged linkage, then extend it again at its own orbit.
    Gauss-Newton on the two nonlinear parameters, all linkages at once; alpha and alpha-dot are solved inside every
    residual evaluation (variable projection), the Jacobian by finite differences. The transform of every detection
    of an exposure at a linkage's orbit is theta f_e + gamma x_E,e, with f_e one number per exposure, so the extension
    is a broadcast over (linkage, exposure, detection) as in verify_gpu. A linkage is replaced by its polished version
    only if that has at least as many nights."""
    if not links:
        return links
    tol = a.fit_tol * ARCSEC
    f64 = torch.float64
    nc = len(links)
    th0, th1, ob = G["theta"][:, 0], G["theta"][:, 1], G["obs"]

    def resid(cid, det, g, gd, w):
        gg, gdd = g[cid], gd[cid]
        tp = G["t"][det] - 1.0 / (C_AU * gg)
        k = 0.5 * GM * gg ** 3
        f = 1.0 + gdd * tp - k * tp * tp - gg * ob[:, 2][det]
        ph = torch.stack([th0[det] * f + gg * ob[:, 0][det], th1[det] * f + gg * ob[:, 1][det]], 1)
        u = 1.0 - k * tp * tp
        al, ad, r = seg_fit(cid, nc, ph, u, tp, w)
        return ph - u[:, None] * al[cid] - tp[:, None] * ad[cid], al, ad, r

    def ss(cid, res, w):
        return torch.zeros(nc, dtype=f64, device=dev).index_add_(0, cid, torch.where(w, (res * res).sum(1), 0.0))

    def gauss_newton(cid, det, g, gd, w, iters):
        npara = torch.sqrt(2 * GM * g ** 3)
        for _ in range(iters):
            r0 = resid(cid, det, g, gd, w)[0]
            s0 = ss(cid, r0, w)
            hg, hd = 1e-6 * g, 1e-3 * npara
            jg = (resid(cid, det, g + hg, gd, w)[0] - r0) / hg[cid, None]
            jd = (resid(cid, det, g, gd + hd, w)[0] - r0) / hd[cid, None]
            wf = w.to(f64)[:, None]
            S = lambda x, y: torch.zeros(nc, dtype=f64, device=dev).index_add_(0, cid, (wf * x * y).sum(1))
            A11, A12, A22 = S(jg, jg) * (1 + 1e-6), S(jg, jd), S(jd, jd) * (1 + 1e-6)
            b1, b2 = -S(jg, r0), -S(jd, r0)
            D = A11 * A22 - A12 * A12
            ok = D > 0
            Dg = torch.where(ok, (A22 * b1 - A12 * b2) / D, 0.0).clamp(-a.dgamma, a.dgamma)
            Dd = torch.where(ok, (A11 * b2 - A12 * b1) / D, 0.0)
            Dd = torch.maximum(torch.minimum(Dd, 0.5 * npara), -0.5 * npara)
            s1 = ss(cid, resid(cid, det, g + Dg, gd + Dd, w)[0], w)
            better = s1 < s0
            g, gd = torch.where(better, g + Dg, g), torch.where(better, gd + Dd, gd)
        return g, gd

    ne, nmax = G["n_exp"], G["nmax"]
    ch = max(1, int(2e8 // (ne * nmax * 2 * 8)))

    def extend(g, gd, al, ad):
        add_c, add_d = [], []
        for s0 in range(0, nc, ch):
            gg, gdd = g[s0:s0 + ch, None], gd[s0:s0 + ch, None]
            te = G["et_d"][None, :] - 1.0 / (C_AU * gg)                       # (c, ne)
            k = 0.5 * GM * gg ** 3
            fe = 1.0 + gdd * te - k * te * te - gg * G["eobs"][None, :, 2]
            pred = (1.0 - k * te * te)[..., None] * al[s0:s0 + ch, None, :] + te[..., None] * ad[s0:s0 + ch, None, :]
            Pl = G["Theta"][None] * fe[:, :, None, None] + gg[:, :, None, None] * G["eobs"][None, :, None, :2]
            d2 = torch.nan_to_num(((Pl - pred[:, :, None, :]) ** 2).sum(-1), nan=float("inf"))
            v, kk = d2.min(-1)
            cc, ee = torch.nonzero(v <= tol * tol, as_tuple=True)
            add_c.append(cc + s0); add_d.append(G["pad"][ee, kk[cc, ee]])
        return add_c, add_d

    n = [len(x["det"]) for x in links]
    cid = torch.as_tensor(np.repeat(np.arange(nc), n), device=dev)
    det = torch.as_tensor(np.concatenate([x["det"] for x in links]).astype(np.int64), device=dev)
    g = torch.as_tensor([x["gamma"] for x in links], dtype=f64, device=dev)
    gd = torch.as_tensor([x.get("gdot", 0.0) for x in links], dtype=f64, device=dev)
    N = len(G["t"])
    for iters in (6, 3):
        w = torch.ones(len(cid), dtype=torch.bool, device=dev)
        g, gd = gauss_newton(cid, det, g, gd, w, iters)
        _, al, ad, _ = resid(cid, det, g, gd, w)
        add_c, add_d = extend(g, gd, al, ad)
        key = torch.unique(torch.cat([cid] + add_c) * N + torch.cat([det] + add_d))
        cid, det = key // N, key % N
        r = resid(cid, det, g, gd, torch.ones(len(cid), dtype=torch.bool, device=dev))[3]
        keep = r <= tol
        cid, det = cid[keep], det[keep]
    w = torch.ones(len(cid), dtype=torch.bool, device=dev)
    res, al, ad, r = resid(cid, det, g, gd, w)
    nn = seg_nights(cid, nc, G["det_night"][det], w).cpu().numpy()
    sq, nk = ss(cid, res, w).cpu().numpy(), seg_count(cid, nc).cpu().numpy()
    ch_, dh = cid.cpu().numpy(), det.cpu().numpy()
    brk = np.searchsorted(ch_, np.arange(nc)), np.searchsorted(ch_, np.arange(nc), side="right")
    gh, gdh, alh, adh = g.cpu().numpy(), gd.cpu().numpy(), al.cpu().numpy(), ad.cpu().numpy()
    out = []
    for q, (s, e) in enumerate(zip(*brk)):
        if nn[q] >= links[q]["nights"] and nn[q] >= a.minnights:
            out.append(dict(det=dh[s:e], nights=int(nn[q]), rms=float(np.sqrt(sq[q] / max(nk[q], 1)) / ARCSEC),
                            gamma=float(gh[q]), gdot=float(gdh[q]),
                            state=[float(alh[q, 0]), float(adh[q, 0]), float(alh[q, 1]), float(adh[q, 1])]))
        else:
            out.append(links[q])
    return out


def link_gamma_gpu(W, G, g, gd, a, timing, dev, sync):
    t0 = time.perf_counter()
    tp = G["t"] - 1.0 / (C_AU * g)
    k = 0.5 * GM * g ** 3
    f = 1.0 + gd * tp - k * tp * tp - g * G["obs"][:, 2]
    phi = G["theta"] * f[:, None] + g * G["obs"][:, :2]
    u = 1.0 - k * tp * tp
    npara = float(np.sqrt(2 * GM * g ** 3))
    pidx = G["pad"].clamp(min=0)
    P = torch.stack([phi[:, 0][pidx], phi[:, 1][pidx]], -1)   # (n_exp, nmax, 2) float64; 1-D gathers
    P = torch.where(G["valid"][..., None], P, torch.full_like(P, float("nan")))
    pad_s, nmax_s, pidx_s, Ps = G["pad"], G["nmax"], pidx, P             # pair search on these (--seed-mask: seeds only)
    if "pad_seed" in G:
        pad_s, nmax_s = G["pad_seed"], G["nmax_seed"]
        pidx_s = pad_s.clamp(min=0)
        Ps = torch.stack([phi[:, 0][pidx_s], phi[:, 1][pidx_s]], -1)
        Ps = torch.where(G["valid_seed"][..., None], Ps, torch.full_like(Ps, float("nan")))
    cen = torch.nanmean(P, 1)
    rad = torch.sqrt(torch.nan_to_num(((P - cen[:, None]) ** 2).sum(-1), nan=0.0).max(1).values)
    reach = W["core"] + npara * (W["half"] + 1) + 5 * a.cell * ARCSEC
    cen_h, rad_h = cen.cpu().numpy(), rad.cpu().numpy()
    tolpos = 2 * a.sigma * ARCSEC
    # candidate exposure pairs
    et, night = G["et"], G["night"]
    E1, E2 = np.triu_indices(G["n_exp"], 1)
    dt = et[E2] - et[E1]
    dn = night[E2] - night[E1]
    R = npara * (dn + 1) + tolpos
    ok = (dn >= a.dtmin) & (dn <= (a.window or np.inf))
    ok &= np.hypot(*(cen_h[E1] - cen_h[E2]).T) <= rad_h[E1] + rad_h[E2] + R
    near = np.hypot(*cen_h.T) - rad_h <= reach
    ok &= near[E1] & near[E2]
    E1, E2, R = E1[ok], E2[ok], R[ok]
    Pf = Ps.float() if a.fp32 else Ps
    sync()
    timing["transform"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    cell = a.cell * ARCSEC
    half = W["half"]
    K0, K1, II, JJ = [], [], [], []                         # keys on the two shifted grids; detection indices (int32)
    npairs = 0

    phx, phy = phi[:, 0].contiguous(), phi[:, 1].contiguous()

    def states(i, j):
        """Pair states at the window mid-time, core and bound-orbit cuts, cell keys on both shifted grids.
        Gathers 1-D arrays only: torch 2.11's vectorized row gather (phi[i] on (N, 2) float64) ran at ~65 GB/s on GH200
        and was 78% of its pair time."""
        ti, tj, ui, uj = tp[i], tp[j], u[i], u[j]
        xi, yi, xj, yj = phx[i], phy[i], phx[j], phy[j]
        D = ui * tj - uj * ti
        al = torch.stack([(xi * tj - xj * ti) / D, (yi * tj - yj * ti) / D], 1)
        ad = torch.stack([(ui * xj - uj * xi) / D, (ui * yj - uj * yi) / D], 1)
        keep = (torch.hypot(al[:, 0], al[:, 1]) <= W["core"]) & (ad.abs() <= npara).all(1)
        i, j, al, ad = i[keep], j[keep], al[keep], ad[keep]
        x = torch.cat([al, ad * half], 1) / cell
        for sh, KL in ((0.0, K0), (0.5, K1)):
            q = (torch.floor(x + sh).long() + OFF).clamp(0, 2 * OFF - 1)
            KL.append((q[:, 0] << 48) | (q[:, 1] << 32) | (q[:, 2] << 16) | q[:, 3])
        II.append(i.int()); JJ.append(j.int())

    if a.pairs == "fused":
        import pairs_triton
        oi, oj, k0, k1, npairs = pairs_triton.find_states(
            Ps.float().contiguous(), phi[pidx_s].reshape(-1, 2).contiguous(), tp[pidx_s].reshape(-1).contiguous(),
            u[pidx_s].reshape(-1).contiguous(), torch.as_tensor(E1, device=dev), torch.as_tensor(E2, device=dev),
            torch.as_tensor(R ** 2, device=dev, dtype=torch.float32), float(W["core"]), npara, float(half), cell)
        flat = pad_s.reshape(-1)
        if len(oi):
            K0.append(k0); K1.append(k1); II.append(flat[oi.long()].int()); JJ.append(flat[oj.long()].int())
        del oi, oj, k0, k1
    elif a.pairs == "triton":
        import pairs_triton
        P32 = Ps.float().contiguous()
        flat = pad_s.reshape(-1)
        eg = max(1, int(4e9 // nmax_s ** 2))               # exposure pairs per kernel call: bounded raw output
        for g0 in range(0, len(E1), eg):
            oi, oj = pairs_triton.find_pairs(P32, torch.as_tensor(E1[g0:g0 + eg], device=dev),
                                             torch.as_tensor(E2[g0:g0 + eg], device=dev),
                                             torch.as_tensor(R[g0:g0 + eg] ** 2, device=dev, dtype=torch.float32))
            npairs += len(oi)
            for c0 in range(0, len(oi), a.chunk):
                states(flat[oi[c0:c0 + a.chunk].long()], flat[oj[c0:c0 + a.chunk].long()])
            del oi, oj
    else:
        for b0 in range(0, len(E1), a.batch):
            e1 = torch.as_tensor(E1[b0:b0 + a.batch], device=dev)
            e2 = torch.as_tensor(E2[b0:b0 + a.batch], device=dev)
            r2 = torch.as_tensor(R[b0:b0 + a.batch] ** 2, device=dev, dtype=Pf.dtype)
            A, B = Pf[e1], Pf[e2]                             # (b, nmax, 2)
            if a.dist == "mm":                                # |a|^2 + |b|^2 - 2 a.b: no (b, nmax, nmax, 2) temporary
                d2 = torch.baddbmm((A * A).sum(-1)[:, :, None] + (B * B).sum(-1)[:, None, :], A, B.transpose(1, 2), alpha=-2.0)
            else:
                d2 = ((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1)
            bb, ii, jj = torch.nonzero(d2 <= r2[:, None, None], as_tuple=True)
            if len(bb) == 0:
                continue
            npairs += len(bb)
            states(pad_s[e1[bb], ii], pad_s[e2[bb], jj])
    sync()
    timing["pairs"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    if not II:
        return [], npairs, 0
    I = torch.cat(II); J = torch.cat(JJ)
    del II, JJ
    KS = [torch.cat(K0), torch.cat(K1)]
    del K0, K1
    members = []
    base = 0
    for s in range(2):
        ks, o = torch.sort(KS[s])
        KS[s] = None
        uk, cnt = torch.unique_consecutive(ks, return_counts=True)
        st = torch.cumsum(cnt, 0) - cnt
        big = torch.nonzero(cnt >= a.minpairs, as_tuple=True)[0]
        if len(big) == 0:
            continue
        c = cnt[big]
        run = torch.repeat_interleave(torch.arange(len(big), device=dev), c)
        pos = torch.repeat_interleave(st[big], c) + torch.arange(int(c.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(c, 0) - c, c)
        m = o[pos]
        del ks, o, uk, st
        members.append((run + base, I[m].long(), J[m].long()))
        base += len(big)
    nkept = len(I)
    sync()
    timing["cluster"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    if not members:
        return [], npairs, nkept
    if a.verify == "gpu":
        N = len(phi)
        key = torch.unique(torch.cat([torch.cat([r, r]) for r, _, _ in members]) * N
                           + torch.cat([torch.cat([i, j]) for _, i, j in members]))
        links = verify_gpu(W, G, P, phi, tp, u, g, gd, key // N, key % N, a)
        sync()
    else:
        cands = []
        for run, i, j in members:
            run, i, j = run.cpu().numpy(), i.cpu().numpy(), j.cpu().numpy()
            o = np.argsort(run, kind="stable")
            run, i, j = run[o], i[o], j[o]
            brk = np.flatnonzero(np.diff(run)) + 1
            for ci, cj in zip(np.split(i, brk), np.split(j, brk)):
                cands.append(np.unique(np.concatenate([ci, cj])))
        phi_h, tp_h, u_h = phi.cpu().numpy(), tp.cpu().numpy(), u.cpu().numpy()
        links = L.verify(W, phi_h, tp_h, u_h, cands, a)
    timing["verify"] += time.perf_counter() - t0
    return links, npairs, nkept


def main():
    ap = L.parser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--fp32", action="store_true", help="distances in float32 (states stay float64)")
    ap.add_argument("--verify", choices=("gpu", "host"), default="gpu", help="where candidates are fitted and extended")
    ap.add_argument("--dist", choices=("diff", "mm"), default="mm",
                    help="pair distances from differences, or from |a|^2 + |b|^2 - 2 a.b (batched matrix product)")
    ap.add_argument("--save-links", default="", help="write the merged linkages (window detection indices) to this .npz")
    ap.add_argument("--pairs", choices=("auto", "torch", "triton", "fused"), default="auto",
                    help="pair finding: batched torch distances; the Triton kernel (pairs_triton.py, float32 distances); "
                         "fused = Triton also computes states, cuts and cell keys (float64); "
                         "auto = Triton on CUDA, torch if it fails to compile or run")
    ap.add_argument("--chunk", type=int, default=1 << 25, help="pairs per state/key chunk with --pairs triton")
    ap.add_argument("--refine", action="store_true",
                    help="fit a gamma-dot correction per candidate in verification (5-parameter linear fit), so a coarse "
                         "gamma-dot grid (e.g. --gdot-tol 0) still gives complete linkages")
    ap.add_argument("--refine-iter", type=int, default=2, help="with --refine: extension rounds")
    ap.add_argument("--refine-nights", type=int, default=4, help="with --refine: fit gamma-dot only on >= this many nights")
    ap.add_argument("--polish", action="store_true",
                    help="stage 2: fit (gamma, gamma-dot) per merged linkage and extend it again at that orbit")
    ap.add_argument("--extend", choices=("brute", "index"), default="brute",
                    help="verification's extension: compare with every detection of every exposure, or through a cell index")
    ap.add_argument("--extend-chunk", type=float, default=4e6, help="--extend index: (survivor, exposure) queries per chunk")
    ap.add_argument("--seed-mask", default="", help=".npz with a boolean 'seed' per data detection: only those seed "
                    "candidates (pair search, clustering); verification extends against all detections")
    ap.add_argument("--shuffle", action="store_true",
                    help="run the hypotheses in random order, so a run cut by --budget is an unbiased sample for timing")
    a = ap.parse_args()
    dev = torch.device(a.device)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    d, meta = L.load(a.data)
    W = L.window(d, meta, a.season, a.core, a.span)
    G = setup(W, dev)
    if a.seed_mask:
        seed_layout(G, np.load(a.seed_mask)["seed"][W["sel"]], dev)
    grid = L.hypotheses(a, W["half"])
    if a.shuffle:
        grid = [grid[k] for k in np.random.default_rng(0).permutation(len(grid))]
    timing = dict(transform=0.0, pairs=0.0, cluster=0.0, verify=0.0)
    links, per = [], []
    # warm-up (kernel compilation, allocator)
    a._g = grid[0][0]
    if a.pairs == "auto":
        a.pairs = "triton" if dev.type == "cuda" else "torch"
        if a.pairs == "triton":
            try:
                link_gamma_gpu(W, G, grid[0][0], grid[0][1], a, dict(timing), dev, sync)
            except Exception as e:
                print(f"Triton pair kernel failed ({type(e).__name__}: {e}); using torch pairs", flush=True)
                a.pairs = "torch"
    link_gamma_gpu(W, G, grid[0][0], grid[0][1], a, dict(timing), dev, sync)
    T0 = time.perf_counter()
    for g, gd in grid:
        a._g = g
        t1 = time.perf_counter()
        Lk, npairs, nkept = link_gamma_gpu(W, G, g, gd, a, timing, dev, sync)
        per.append(dict(r=float(1 / g), gdot=gd, pairs=int(npairs), kept=int(nkept), links=len(Lk), sec=time.perf_counter() - t1))
        links += Lk
        if a.budget and time.perf_counter() - T0 > a.budget:
            break
    links = L.merge(links)
    if a.polish:
        t1 = time.perf_counter()
        links = L.merge(polish(W, G, links, a, dev))
        sync()
        timing["polish"] = time.perf_counter() - t1
    wall = time.perf_counter() - T0
    ev = L.evaluate(d, W, links, a)
    import copy
    # stricter linkage requirements, applied to the same linkages: >= k nights (findable also needs k nights)
    ev_k = {}
    for k in range(a.minnights + 1, a.minnights + 5):
        ak = copy.copy(a); ak.minnights = k
        e = L.evaluate(d, W, [x for x in links if x["nights"] >= k], ak)
        ev_k[k] = {q: e[q] for q in ("findable", "found", "false")}
        ev_k[k]["missed"] = [m["id"] for m in e["missed"]]
    if a.save_links:
        n = np.array([len(x["det"]) for x in links], np.int64)
        np.savez_compressed(a.save_links, det=np.concatenate([x["det"] for x in links]) if links else np.zeros(0, np.int64),
                            off=np.concatenate([[0], np.cumsum(n)]), nights=np.array([x["nights"] for x in links]),
                            rms=np.array([x["rms"] for x in links]), gamma=np.array([x["gamma"] for x in links]),
                            state=np.array([x["state"] for x in links]).reshape(-1, 4), sel=W["sel"])
    gpu_s = timing["transform"] + timing["pairs"] + timing["cluster"] + (timing["verify"] if a.verify == "gpu" else 0.0) \
        + timing.get("polish", 0.0)
    a_full = copy.copy(a); a_full.sample = 0
    n_full = len(L.hypotheses(a_full, W["half"]))
    res = dict(args={k: v for k, v in vars(a).items() if not k.startswith("_")}, device=str(dev),
               device_name=torch.cuda.get_device_name(0) if dev.type == "cuda" else a.device,
               n_det=int(len(W["sel"])), n_seed=int(G["valid_seed"].sum()) if "pad_seed" in G else int(len(W["sel"])),
               n_exp=G["n_exp"], nmax=G["nmax"], n_hyp=len(grid), n_hyp_full=n_full, half=W["half"], n_done=len(per),
               n_gamma=len(set(g for g, _ in grid)), wall_s=wall, gpu_s=gpu_s, timing=timing,
               pairs=int(sum(p["pairs"] for p in per)), pairs_per_gpu_s=sum(p["pairs"] for p in per) / max(1e-9, gpu_s),
               eval=ev, eval_min_nights=ev_k, per_gamma=per)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_gamma", "args")}, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
