"""gen129: old (gen128) vs TNO_WIN_FAST window/deep shift-and-add loop; bit-identity + micro-benchmark.
Loop bodies copied verbatim from stackd/finder.py.  usage: python gen129_winfast_test.py [bench]"""
import sys, time, math, numpy as np, torch
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
def sync():
    if dev == "mps": torch.mps.synchronize()
    elif dev == "cuda": torch.cuda.synchronize()

def setup(K, Hc, Wc, P, lmo_, vcon_):
    Dm = (torch.randn((K, Hc + 2 * P, Wc + 2 * P), device=dev) * 2.0).half()
    Dm[:, :, :P // 2] = 0          # uncovered strip
    best = torch.full((Hc, Wc), -1e4, dtype=torch.float16, device=dev)
    bidx = torch.zeros((Hc, Wc), dtype=torch.int32, device=dev)
    s1_ = torch.zeros((Hc, Wc), dtype=torch.float32, device=dev) if vcon_ else None
    s2_ = torch.zeros((Hc, Wc), dtype=torch.float32, device=dev) if vcon_ else None
    return Dm, best, bidx, s1_, s2_

def old(Dm, best, bidx, s1_, s2_, VX, VY, dts, offs, P, Hc, Wc, lmo_, vcon_, deadline):
    nacc_ = 0; nhyp = 0; nfa = 0
    for h in range(len(VX)):
        if (h & 63) == 0 and time.time() > deadline:
            break
        acc = None
        mxs_ = None
        for k_ in range(Dm.shape[0]):
            sx = int(round(VX[h] * dts[k_] - offs[k_, 0] * 0)); sy = int(round(VY[h] * dts[k_]))
            sl = Dm[k_, P + sy:P + sy + Hc, P + sx:P + sx + Wc]
            acc = sl.clone() if acc is None else acc.add_(sl)
            if lmo_:
                mxs_ = sl.clone() if mxs_ is None else torch.maximum(mxs_, sl, out=mxs_)
        if lmo_:
            acc.sub_(mxs_)
        if vcon_:
            af_ = acc.float(); s1_ += af_; s2_.addcmul_(af_, af_); del af_; nacc_ += 1
        better = acc > best
        best = torch.where(better, acc, best)
        bidx = torch.where(better, torch.full_like(bidx, h), bidx)
        nhyp += 1
        nfa += Dm.shape[0]
    sync()
    return best, bidx, s1_, s2_, nhyp, h

def new(Dm, best, bidx, s1_, s2_, VX, VY, dts, offs, P, Hc, Wc, lmo_, vcon_, deadline):
    nacc_ = 0; nhyp = 0; nfa = 0
    h = -1
    K_ = Dm.shape[0]
    acc_b = torch.empty((Hc, Wc), dtype=Dm.dtype, device=dev)
    mxs_b = torch.empty((Hc, Wc), dtype=Dm.dtype, device=dev) if lmo_ else None
    bet_b = torch.empty((Hc, Wc), dtype=torch.bool, device=dev)
    af_b = torch.empty((Hc, Wc), dtype=torch.float32, device=dev) if vcon_ else None
    dts_ = np.asarray(dts, dtype=np.float64)[:K_]
    SXs = np.rint(np.asarray(VX, np.float64)[:, None] * dts_[None, :]).astype(np.int64).tolist()
    SYs = np.rint(np.asarray(VY, np.float64)[:, None] * dts_[None, :]).astype(np.int64).tolist()
    for h in range(len(VX)):
        if (h & 63) == 0 and time.time() > deadline:
            break
        sxh, syh = SXs[h], SYs[h]
        sls = [Dm[k_, P + syh[k_]:P + syh[k_] + Hc, P + sxh[k_]:P + sxh[k_] + Wc] for k_ in range(K_)]
        torch.add(sls[0], sls[1], out=acc_b)
        for k_ in range(2, K_):
            acc_b.add_(sls[k_])
        if lmo_:
            torch.maximum(sls[0], sls[1], out=mxs_b)
            for k_ in range(2, K_):
                torch.maximum(mxs_b, sls[k_], out=mxs_b)
            acc_b.sub_(mxs_b)
        if vcon_:
            af_b.copy_(acc_b); s1_ += af_b; s2_.addcmul_(af_b, af_b); nacc_ += 1
        torch.gt(acc_b, best, out=bet_b)
        torch.where(bet_b, acc_b, best, out=best)
        bidx.masked_fill_(bet_b, h)
        nhyp += 1
        nfa += K_
    sync()
    return best, bidx, s1_, s2_, nhyp, h

def case(K, Hc, Wc, nights, nh, lmo_, vcon_, seed, deep):
    rng = np.random.default_rng(seed)
    nt = np.sort(rng.choice(np.arange(60), nights, replace=False)).astype(float)
    dts = np.sort(np.concatenate([nt, nt[rng.integers(0, nights, K - nights)] + rng.uniform(0.01, 0.1, K - nights)]))
    dts = dts - dts.mean() if deep else dts - dts[0] + rng.uniform(-0.5, 0.5)
    # velocity grid: integer multiples of dv, sorted by closeness to a circular point (as finder.py)
    span = np.abs(dts).max()
    vmax = 0.9 * 60 / span
    dv = 1.0 / span
    nv = int(vmax / dv) + 1
    vv = np.arange(-nv, nv + 1) * dv
    VX, VY = np.meshgrid(vv, vv); VX = VX.ravel(); VY = VY.ravel()
    vcx, vcy = 0.6 * vmax, 0.1 * vmax
    o_ = np.argsort((VX - vcx) ** 2 + (VY - vcy) ** 2)[:nh]
    VX, VY = VX[o_], VY[o_]
    P = int(vmax * span + 8)
    offs = rng.normal(size=(K, 2))
    torch.manual_seed(seed)
    S = setup(K, Hc, Wc, P, lmo_, vcon_)
    res = {}
    for nm, fn in (("old", old), ("new", new)):
        args = [S[0], S[1].clone(), S[2].clone(), None if S[3] is None else S[3].clone(), None if S[4] is None else S[4].clone()]
        fn(*args, VX, VY, dts, offs, P, Hc, Wc, lmo_, vcon_, 1e18)   # warm-up
        args = [S[0], S[1].clone(), S[2].clone(), None if S[3] is None else S[3].clone(), None if S[4] is None else S[4].clone()]
        t0 = time.time(); r = fn(*args, VX, VY, dts, offs, P, Hc, Wc, lmo_, vcon_, 1e18); res[nm] = (r, time.time() - t0)
    (a, ta), (b, tb) = res["old"], res["new"]
    same = torch.equal(a[0].view(torch.int16), b[0].view(torch.int16)) and torch.equal(a[1], b[1]) and a[4] == b[4]
    if vcon_:
        same = same and torch.equal(a[2].view(torch.int32), b[2].view(torch.int32)) and torch.equal(a[3].view(torch.int32), b[3].view(torch.int32))
    print(f"K={K:2d} {Hc}x{Wc} nh={len(VX)} lmo={int(lmo_)} vcon={int(vcon_)}: identical={same} "
          f"old {ta / len(VX) * 1e3:.3f} ms/hyp new {tb / len(VX) * 1e3:.3f} ms/hyp speedup {ta / tb:.2f}x", flush=True)
    return same

if __name__ == "__main__":
    bench = len(sys.argv) > 1 and sys.argv[1] == "bench"
    ok = True
    if not bench:
        for (K, n, lmo, vc, deep, sd) in [(4, 3, 0, 0, 0, 1), (6, 4, 0, 0, 0, 2), (10, 7, 0, 0, 0, 3), (2, 2, 0, 0, 0, 4),
                                          (5, 5, 1, 1, 1, 5), (9, 8, 1, 1, 1, 6), (3, 3, 0, 1, 1, 7), (12, 8, 1, 1, 1, 8)]:
            ok &= case(K, 600, 700, n, 400, bool(lmo), bool(vc), sd, bool(deep))
        print("ALL IDENTICAL" if ok else "MISMATCH")
    else:
        for (K, n, lmo, vc, deep) in [(4, 3, 0, 0, 0), (8, 6, 0, 0, 0), (12, 8, 0, 0, 0), (8, 8, 1, 1, 1)]:
            for HW in (1000, 2600):
                ok &= case(K, HW, HW, n, 192 if HW > 2000 else 384, bool(lmo), bool(vc), 11, bool(deep))
        print("ALL IDENTICAL" if ok else "MISMATCH")
