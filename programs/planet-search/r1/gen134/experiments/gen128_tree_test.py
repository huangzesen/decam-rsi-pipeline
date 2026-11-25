import sys, time, numpy as np, torch
sys.path.insert(0, sys.argv[1]); import fstk
dev = "mps"; rng = np.random.default_rng(0)
for K, dts in [(7, np.array([-28.4, -27.4, -20.1, -2.0, 1.0, 15.3, 28.4])), (4, np.array([-14.5, -13.4, 10.2, 14.5])), (6, np.array([-27, -26, -1, 0.5, 25, 27.]))]:
    Hc, Wc, P = 600, 700, 60; WD = Wc + 2 * P; HD = Hc + 2 * P; L = Hc * WD
    Dq = torch.tensor(rng.integers(-18, 18, (K, HD * WD)), dtype=torch.int8, device=dev)
    dv = 1 / dts.max(); ii = np.arange(-40, 41); IX, IY = np.meshgrid(ii, ii)
    ok = (np.hypot(IX, IY) > 25) & (np.hypot(IX, IY) < 40) & (IX > 0)
    IXo, IYo = IX[ok], IY[ok]; VX, VY = IXo * dv, IYo * dv
    o_ = np.argsort((VX - 30*dv) ** 2 + VY ** 2); VX, VY, IXo, IYo = VX[o_], VY[o_], IXo[o_], IYo[o_]
    SX = np.round(np.outer(VX, dts)).astype(np.int64); SY = np.round(np.outer(VY, dts)).astype(np.int64)
    OFF = (P + SY) * WD + (P + SX)
    best = torch.full((L,), -127, dtype=torch.int8, device=dev); acc = torch.empty_like(best)
    t0 = time.time()
    for h in range(len(VX)):
        of_ = OFF[h]; torch.add(Dq[0, of_[0]:of_[0] + L], Dq[1, of_[1]:of_[1] + L], out=acc)
        for k_ in range(2, K): acc.add_(Dq[k_, of_[k_]:of_[k_] + L])
        torch.maximum(best, acc, out=best)
    torch.mps.synchronize(); t1 = time.time()
    best2 = torch.full((L,), -127, dtype=torch.int8, device=dev)
    kb = (IXo // 8) * 100003 + (IYo // 8); _, bid = np.unique(kb, return_inverse=True); bid = bid.ravel()
    SA, SB = fstk._split(list(np.argsort(dts)), dts); npass = 0
    for b_ in np.unique(bid):
        hs = np.nonzero(bid == b_)[0]; OFFb = OFF[hs]; cnt = [0]
        pa, ia = fstk._build(torch, Dq, OFFb, SA, dts, L, cnt); pb, ib = fstk._build(torch, Dq, OFFb, SB, dts, L, cnt)
        for j_, h in enumerate(hs):
            ta, ba = pa[ia[j_]]; tb, bb = pb[ib[j_]]
            oa = int(OFFb[j_, SA[0]]) - ba; ob = int(OFFb[j_, SB[0]]) - bb
            torch.add(ta[oa:oa + L], tb[ob:ob + L], out=acc); torch.maximum(best2, acc, out=best2)
        npass += cnt[0] + 2 * len(hs)
    torch.mps.synchronize(); t2 = time.time()
    print(K, len(VX), "identical", bool(torch.equal(best, best2)), f"passes/hyp {npass/len(VX):.2f}", f"flat {t1-t0:.2f}s tree {t2-t1:.2f}s")
