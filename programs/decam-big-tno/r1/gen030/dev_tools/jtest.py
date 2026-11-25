import torch, math
torch.manual_seed(1)
nA, nB = 20000, 20000
pA = torch.rand(nA, 2, dtype=torch.float64) * 0.2; pB = torch.rand(nB, 2, dtype=torch.float64) * 0.2
wA_ = torch.rand(nA, 2, dtype=torch.float64) * torch.tensor([35.0, 80.0]) ; wB_ = torch.rand(nB, 2, dtype=torch.float64) * torch.tensor([35.0, 80.0])
cpos_ = 0.004; cw_ = 3.2
x0 = min(float(pA[:, 0].min()), float(pB[:, 0].min())) - 3 * cpos_; y0 = min(float(pA[:, 1].min()), float(pB[:, 1].min())) - 3 * cpos_
u0 = min(float(wA_[:, 0].min()), float(wB_[:, 0].min())) - 3 * cw_
v0_ = min(float(wA_[:, 1].min()), float(wB_[:, 1].min())) - 2 * cw_
vR_ = max(float(wA_[:, 1].max()), float(wB_[:, 1].max())) + 2 * cw_ - v0_
fl_ = lambda v_, o_, c_: torch.floor((v_ - o_) / (2.0 * c_)).to(torch.int64)
kBx, kBy, kBu = fl_(pB[:, 0], x0, cpos_), fl_(pB[:, 1], y0, cpos_), fl_(wB_[:, 0], u0, cw_)
lox, hix = fl_(pA[:, 0] - cpos_, x0, cpos_), fl_(pA[:, 0] + cpos_, x0, cpos_)
loy, hiy = fl_(pA[:, 1] - cpos_, y0, cpos_), fl_(pA[:, 1] + cpos_, y0, cpos_)
lou, hiu = fl_(wA_[:, 0] - cw_, u0, cw_), fl_(wA_[:, 0] + cw_, u0, cw_)
My_ = int(max(int(kBy.max()), int(hiy.max()))) + 2; Mu_ = int(max(int(kBu.max()), int(hiu.max()))) + 2
KB_, oB = torch.sort(((kBx * My_ + kBy) * Mu_ + kBu).double() + ((wB_[:, 1] - v0_) / vR_).clamp(0.0, 0.999999))
fa0_ = ((wA_[:, 1] - cw_ - v0_) / vR_).clamp(0.0, 0.999999); fa1_ = ((wA_[:, 1] + cw_ - v0_) / vR_).clamp(0.0, 0.999999)
pairs = set(); tot = 0
for sx_ in (0, 1):
    for sy_ in (0, 1):
        for su_ in (0, 1):
            c3_ = (((hix if sx_ else lox) * My_ + (hiy if sy_ else loy)) * Mu_ + (hiu if su_ else lou)).double()
            lo_ = torch.searchsorted(KB_, c3_ + fa0_); hi_ = torch.searchsorted(KB_, c3_ + fa1_, right=True)
            cn_ = hi_ - lo_
            if sx_: cn_ = cn_ * (hix != lox)
            if sy_: cn_ = cn_ * (hiy != loy)
            if su_: cn_ = cn_ * (hiu != lou)
            tot += int(cn_.sum())
            ia = torch.repeat_interleave(torch.arange(nA), cn_); st = torch.cumsum(cn_, 0) - cn_
            off = torch.arange(int(cn_.sum())) - torch.repeat_interleave(st, cn_)
            ib = oB[lo_[ia] + off]
            pairs |= set(zip(ia.tolist(), ib.tolist()))
# brute force
dp = (pA[:, None] - pB[None]).abs(); dw = (wA_[:, None] - wB_[None]).abs()
ok = (dp[..., 0] <= cpos_) & (dp[..., 1] <= cpos_) & (dw[..., 0] <= cw_) & (dw[..., 1] <= cw_)
bi, bj = torch.nonzero(ok, as_tuple=True); bf = set(zip(bi.tolist(), bj.tolist()))
okp = (dp[..., 0] <= cpos_) & (dp[..., 1] <= cpos_)
print('join candidates', tot, 'distinct', len(pairs), 'exact box', len(bf), 'missed', len(bf - pairs), 'position-only box', int(okp.sum()))
