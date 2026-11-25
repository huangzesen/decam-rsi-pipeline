"""Fused pair finding for link_gpu.py (--pairs triton): no (b, n, n) distance tensor.

The detections of each exposure sit in a padded (n_exp, nmax, 2) float32 array (NaN padding). For every candidate
exposure pair (e1, e2) with radius R, the detections are compared in BI x BJ blocks inside one kernel. Two passes: the
first counts the pairs within R per block, a prefix sum gives each block its output offset, the second writes the
(e1 slot, e2 slot) indices of those pairs (slot = exposure * nmax + position). Same comparison as the torch path in
float32: (ax - bx)^2 + (ay - by)^2 <= R^2.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _block(P, E1, E2, R2, nmax, BI: tl.constexpr, BJ: tl.constexpr):
    p = tl.program_id(0)
    oi = tl.program_id(1) * BI + tl.arange(0, BI)
    oj = tl.program_id(2) * BJ + tl.arange(0, BJ)
    e1 = tl.load(E1 + p).to(tl.int64)
    e2 = tl.load(E2 + p).to(tl.int64)
    r2 = tl.load(R2 + p)
    mi = oi < nmax
    mj = oj < nmax
    ax = tl.load(P + (e1 * nmax + oi) * 2, mask=mi, other=float("nan"))
    ay = tl.load(P + (e1 * nmax + oi) * 2 + 1, mask=mi, other=float("nan"))
    bx = tl.load(P + (e2 * nmax + oj) * 2, mask=mj, other=float("nan"))
    by = tl.load(P + (e2 * nmax + oj) * 2 + 1, mask=mj, other=float("nan"))
    dx = ax[:, None] - bx[None, :]
    dy = ay[:, None] - by[None, :]
    m = (dx * dx + dy * dy) <= r2
    return m, e1, e2, oi, oj


@triton.jit
def _count(P, E1, E2, R2, CNT, nmax, nbi, nbj, BI: tl.constexpr, BJ: tl.constexpr):
    m, e1, e2, oi, oj = _block(P, E1, E2, R2, nmax, BI, BJ)
    k = (tl.program_id(0) * nbi + tl.program_id(1)) * nbj + tl.program_id(2)
    tl.store(CNT + k, tl.sum(tl.sum(m.to(tl.int32), axis=1), axis=0))


@triton.jit
def _write(P, E1, E2, R2, OFF, OI, OJ, nmax, nbi, nbj, BI: tl.constexpr, BJ: tl.constexpr):
    m, e1, e2, oi, oj = _block(P, E1, E2, R2, nmax, BI, BJ)
    k = (tl.program_id(0) * nbi + tl.program_id(1)) * nbj + tl.program_id(2)
    base = tl.load(OFF + k)
    mf = tl.reshape(m.to(tl.int32), (BI * BJ,))
    pos = base + (tl.cumsum(mf, axis=0) - mf).to(tl.int64)
    zero = tl.zeros((BI, BJ), dtype=tl.int64)
    si = tl.reshape(e1 * nmax + oi[:, None] + zero, (BI * BJ,))
    sj = tl.reshape(e2 * nmax + oj[None, :] + zero, (BI * BJ,))
    keep = mf > 0
    tl.store(OI + pos, si.to(tl.int32), mask=keep)
    tl.store(OJ + pos, sj.to(tl.int32), mask=keep)


@triton.jit
def _states(P, E1, E2, R2, PH, TP, U, nmax, core2, npara, half, cell, BI: tl.constexpr, BJ: tl.constexpr):
    """Distance test, then (float64) the pair state at the window mid-time, the core and bound-orbit cuts and the cell
    coordinates, as link_gpu.link_gamma_gpu.states does."""
    m, e1, e2, oi, oj = _block(P, E1, E2, R2, nmax, BI, BJ)
    si = e1 * nmax + oi
    sj = e2 * nmax + oj
    mi = oi < nmax
    mj = oj < nmax
    xi = tl.load(PH + si * 2, mask=mi, other=0.0)
    yi = tl.load(PH + si * 2 + 1, mask=mi, other=0.0)
    ti = tl.load(TP + si, mask=mi, other=0.0)
    ui = tl.load(U + si, mask=mi, other=1.0)
    xj = tl.load(PH + sj * 2, mask=mj, other=0.0)
    yj = tl.load(PH + sj * 2 + 1, mask=mj, other=0.0)
    tj = tl.load(TP + sj, mask=mj, other=1.0)
    uj = tl.load(U + sj, mask=mj, other=1.0)
    D = ui[:, None] * tj[None, :] - uj[None, :] * ti[:, None]
    D = tl.where(m, D, 1.0)
    alx = (xi[:, None] * tj[None, :] - xj[None, :] * ti[:, None]) / D
    aly = (yi[:, None] * tj[None, :] - yj[None, :] * ti[:, None]) / D
    adx = (ui[:, None] * xj[None, :] - uj[None, :] * xi[:, None]) / D
    ady = (ui[:, None] * yj[None, :] - uj[None, :] * yi[:, None]) / D
    keep = m & (alx * alx + aly * aly <= core2) & (tl.abs(adx) <= npara) & (tl.abs(ady) <= npara)
    return keep, m, si, sj, alx / cell, aly / cell, adx * half / cell, ady * half / cell


@triton.jit
def _key(x0, x1, x2, x3, sh):
    q0 = tl.minimum(tl.maximum(tl.floor(x0 + sh).to(tl.int64) + 32768, 0), 65535)     # OFF = 1 << 15, as link_gpu
    q1 = tl.minimum(tl.maximum(tl.floor(x1 + sh).to(tl.int64) + 32768, 0), 65535)
    q2 = tl.minimum(tl.maximum(tl.floor(x2 + sh).to(tl.int64) + 32768, 0), 65535)
    q3 = tl.minimum(tl.maximum(tl.floor(x3 + sh).to(tl.int64) + 32768, 0), 65535)
    return (q0 << 48) | (q1 << 32) | (q2 << 16) | q3


@triton.jit
def _fcount(P, E1, E2, R2, PH, TP, U, CNT, CNTD, nmax, nbi, nbj, core2, npara, half, cell,
            BI: tl.constexpr, BJ: tl.constexpr):
    keep, m, si, sj, x0, x1, x2, x3 = _states(P, E1, E2, R2, PH, TP, U, nmax, core2, npara, half, cell, BI, BJ)
    k = (tl.program_id(0) * nbi + tl.program_id(1)) * nbj + tl.program_id(2)
    tl.store(CNT + k, tl.sum(tl.sum(keep.to(tl.int32), axis=1), axis=0))
    tl.store(CNTD + k, tl.sum(tl.sum(m.to(tl.int32), axis=1), axis=0))


@triton.jit
def _fwrite(P, E1, E2, R2, PH, TP, U, OFFS, OI, OJ, K0, K1, nmax, nbi, nbj, core2, npara, half, cell,
            BI: tl.constexpr, BJ: tl.constexpr):
    keep, m, si, sj, x0, x1, x2, x3 = _states(P, E1, E2, R2, PH, TP, U, nmax, core2, npara, half, cell, BI, BJ)
    k = (tl.program_id(0) * nbi + tl.program_id(1)) * nbj + tl.program_id(2)
    base = tl.load(OFFS + k)
    mf = tl.reshape(keep.to(tl.int32), (BI * BJ,))
    pos = base + (tl.cumsum(mf, axis=0) - mf).to(tl.int64)
    ok = mf > 0
    zero = tl.zeros((BI, BJ), dtype=tl.int64)
    tl.store(OI + pos, tl.reshape(si[:, None] + zero, (BI * BJ,)).to(tl.int32), mask=ok)
    tl.store(OJ + pos, tl.reshape(sj[None, :] + zero, (BI * BJ,)).to(tl.int32), mask=ok)
    tl.store(K0 + pos, tl.reshape(_key(x0, x1, x2, x3, 0.0), (BI * BJ,)), mask=ok)
    tl.store(K1 + pos, tl.reshape(_key(x0, x1, x2, x3, 0.5), (BI * BJ,)), mask=ok)


def find_states(P, PH, TP, U, E1, E2, R2, core, npara, half, cell, BI=32, BJ=32):
    """As find_pairs, but returns only the pairs that pass the core and bound-orbit cuts, with their cell keys:
    (oi, oj) int32 slots, (k0, k1) int64 keys on the two shifted grids, and the number of pairs within R before the
    cuts. PH (slots, 2), TP, U (slots,) float64."""
    nmax = P.shape[1]
    nbi, nbj = triton.cdiv(nmax, BI), triton.cdiv(nmax, BJ)
    grid = (len(E1), nbi, nbj)
    args = (nmax, nbi, nbj, core * core, npara, half, cell)
    cnt = torch.empty(len(E1) * nbi * nbj, dtype=torch.int32, device=P.device)
    cntd = torch.empty_like(cnt)
    _fcount[grid](P, E1, E2, R2, PH, TP, U, cnt, cntd, *args, BI=BI, BJ=BJ)
    c = cnt.to(torch.int64)
    off = torch.cumsum(c, 0) - c
    total = int(c.sum())
    oi = torch.empty(total, dtype=torch.int32, device=P.device)
    oj = torch.empty(total, dtype=torch.int32, device=P.device)
    k0 = torch.empty(total, dtype=torch.int64, device=P.device)
    k1 = torch.empty(total, dtype=torch.int64, device=P.device)
    if total:
        _fwrite[grid](P, E1, E2, R2, PH, TP, U, off, oi, oj, k0, k1, *args, BI=BI, BJ=BJ)
    return oi, oj, k0, k1, int(cntd.to(torch.int64).sum())


def find_pairs(P, E1, E2, R2, BI=64, BJ=64):
    """P: (n_exp, nmax, 2) float32 contiguous (NaN padding); E1, E2: int64 exposure indices; R2: float32 radii^2.
    Returns int32 slot indices (oi, oj) of every pair within R, ordered by exposure pair then block."""
    nmax = P.shape[1]
    nbi, nbj = triton.cdiv(nmax, BI), triton.cdiv(nmax, BJ)
    grid = (len(E1), nbi, nbj)
    cnt = torch.empty(len(E1) * nbi * nbj, dtype=torch.int32, device=P.device)
    _count[grid](P, E1, E2, R2, cnt, nmax, nbi, nbj, BI=BI, BJ=BJ)
    c = cnt.to(torch.int64)
    off = torch.cumsum(c, 0) - c
    total = int(c.sum())
    oi = torch.empty(total, dtype=torch.int32, device=P.device)
    oj = torch.empty(total, dtype=torch.int32, device=P.device)
    if total:
        _write[grid](P, E1, E2, R2, off, oi, oj, nmax, nbi, nbj, BI=BI, BJ=BJ)
    return oi, oj
