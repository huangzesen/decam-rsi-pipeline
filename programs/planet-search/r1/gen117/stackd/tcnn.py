"""gen 106 (arm B; ported gen117, TNO_CNN default on, =0 off): self-trained whole-track CNN classifier.

Trained inside stackd on THIS scenario's own maps: a hypothesis (state at t_ep) is turned into K per-night weighted
co-add cutouts (RxR on the 1" S/N maps, centred on the predicted track position) plus the all-night co-add cutout.
Positives = synthetic point sources (matched-filter profile with each frame's FWHM, per-night S/N 1.5-6, depth
weighted) injected along real seed tracks moved to random sky positions, or on top of low-ranked seed tracks;
negatives = the same tracks without injection (random positions, low-ranked seeds).  The network is shared over
nights (2-channel 9x9 input: night cutout, all-night cutout) and pools over nights (masked mean + max), so K can vary.
"""
import math
import time

import numpy as np

RC = 4            # cutout half-size (9x9)
import os
AMIN = float(os.environ.get("TNO_CNN_AMIN", "1.0")); AMAX = float(os.environ.get("TNO_CNN_AMAX", "12"))   # injected per-night S/N
AS_PER_NPX = 0.263
ALOG = os.environ.get("TNO_CNN_ALOG", "1") == "1"
JITC = float(os.environ.get("TNO_CNN_JITC", "0.6"))


def _pix(orbit, S, t_ep, t, obs, TF0, x0, y0, PIX):
    u = orbit.predict_dirs_b(S, t_ep, t, obs)
    q = u @ TF0.T
    X = (q[..., 0] / q[..., 2] - x0) / PIX
    Y = (q[..., 1] / q[..., 2] - y0) / PIX
    return X, Y          # (M, N) float pixel coords


class Ctx:
    def __init__(self, torch, dev, maps, cov, wf, night, fw_as, W, H):
        self.torch = torch; self.dev = dev
        N = maps.shape[0]
        self.N = N; self.W = W; self.H = H
        self.mflat = maps.reshape(-1)
        self.cflat = cov.reshape(-1)
        nu = np.unique(night)
        self.nidx = [np.flatnonzero(night == u) for u in nu]
        self.KN = len(nu)
        self.wf = torch.tensor(np.asarray(wf, np.float32), device=dev)
        A = np.zeros((self.KN, N), np.float32)
        for k, idx in enumerate(self.nidx):
            A[k, idx] = 1.0
        self.A = torch.tensor(A, device=dev)                          # (KN, N) night membership
        s = np.asarray(fw_as, np.float32) * math.sqrt(2.0) / 2.355     # matched-filter response sigma (arcsec = px)
        self.sig = torch.tensor(np.sqrt(s ** 2 + 0.12).astype(np.float32), device=dev)   # + 1" max-pool smear
        o = torch.arange(-RC, RC + 1, device=dev)
        self.oy = o[:, None].expand(2 * RC + 1, 2 * RC + 1).reshape(-1)
        self.ox = o[None, :].expand(2 * RC + 1, 2 * RC + 1).reshape(-1)

    def cut(self, X, Y, amp=None, jit=None):
        """X, Y (M,N) numpy float px -> features (M, KN, 2, 9, 9) and night mask (M, KN).  amp (M,) per-night S/N of an
        injected source (None = no injection); jit (M,N,2) px offset of the injected source."""
        torch = self.torch
        M, N = X.shape
        Xt = torch.tensor(X, dtype=torch.float32, device=self.dev); Yt = torch.tensor(Y, dtype=torch.float32, device=self.dev)
        ix = torch.round(Xt).long(); iy = torch.round(Yt).long()
        px = iy[..., None] + self.oy; qx = ix[..., None] + self.ox                  # (M,N,P)
        ok = (px >= 0) & (px < self.H) & (qx >= 0) & (qx < self.W)
        lin = torch.where(ok, px * self.W + qx, torch.zeros_like(px)) + (torch.arange(N, device=self.dev) * (self.H * self.W))[None, :, None]
        v = self.mflat[lin].float()
        c = self.cflat[lin] & ok
        if amp is not None:
            fx = Xt - ix; fy = Yt - iy
            if jit is not None:
                jt = torch.tensor(jit, dtype=torch.float32, device=self.dev)
                fx = fx + jt[..., 0]; fy = fy + jt[..., 1]
            d2 = (self.ox[None, None] - fx[..., None]) ** 2 + (self.oy[None, None] - fy[..., None]) ** 2
            prof = torch.exp(-0.5 * d2 / (self.sig[None, :, None] ** 2))
            # per-frame S/N: frame depth wf relative to the night's quadrature depth, night S/N = amp
            wn2 = (self.A * self.wf[None] ** 2).sum(1)                      # (KN,)
            fr_w = self.wf / torch.sqrt((self.A.t() @ wn2).clamp(min=1e-6))  # (N,)
            a_ = torch.tensor(amp, dtype=torch.float32, device=self.dev)[:, None] * fr_w[None]   # (M,N)
            v = torch.where(c & (v != 0), (v + a_[..., None] * prof).clamp(-3.0, 5.0), v)
        v = torch.where(c, v, torch.zeros_like(v))
        cf = c.float()
        w = self.wf[None, :, None]
        num = torch.einsum("kn,mnp->mkp", self.A, v * w)
        den = torch.einsum("kn,mnp->mkp", self.A, cf * w * w)
        z = torch.where(den > 1e-3, num / torch.sqrt(den.clamp(min=1e-3)), torch.zeros_like(num))      # (M,KN,P)
        nt = (v * w).sum(1); dt = (cf * w * w).sum(1)
        zt = torch.where(dt > 1e-3, nt / torch.sqrt(dt.clamp(min=1e-3)), torch.zeros_like(nt))         # (M,P)
        nm = (den[..., RC * (2 * RC + 1) + RC] > 1e-3)                                                  # night covered at centre
        P = 2 * RC + 1
        f = torch.stack([z, zt[:, None].expand_as(z)], 2).reshape(M, self.KN, 2, P, P)
        return f.clamp(-4.0, 8.0) / 3.0, nm


def make_model(torch):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
            self.e = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
            self.h = nn.Sequential(nn.Linear(64 + 1, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x, m):
            M, K = x.shape[:2]
            g = self.c(x.reshape(M * K, *x.shape[2:]))                         # (MK,32,9,9)
            ctr = g[:, :, RC - 1:RC + 2, RC - 1:RC + 2].mean((2, 3))
            gm = g.amax((2, 3))
            e = self.e(torch.cat([ctr, gm], 1)).reshape(M, K, 32)
            mf = m.float()[..., None]
            nk = mf.sum(1).clamp(min=1.0)
            mean = (e * mf).sum(1) / nk
            mx = torch.where(mf > 0, e, torch.full_like(e, -1e4)).amax(1)
            mx = torch.where(nk > 0, mx, torch.zeros_like(mx))
            return self.h(torch.cat([mean, mx, nk / K], 1))[:, 0]

    return Net()


def train(torch, ctx, orbit, S_seed, seed_sc, geo, log, tmax=25.0, nmax=24000, seed=106):
    """S_seed (n,6) seed states at t_ep (track shapes + negatives).  geo = (t_ep, t, obs, TF0, x0, y0, PIX)."""
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    S_seed = np.asarray(S_seed, float)
    ok_ = np.all(np.isfinite(S_seed), 1)
    S_seed = S_seed[ok_]; seed_sc = np.asarray(seed_sc, float)[ok_]
    if len(S_seed) < 50:
        raise RuntimeError(f"too few seed states {len(S_seed)}")
    X0, Y0 = _pix(orbit, S_seed, *geo)
    fin = np.all(np.isfinite(X0), 1) & np.all(np.isfinite(Y0), 1)
    X0, Y0, S_seed, seed_sc = X0[fin], Y0[fin], S_seed[fin], seed_sc[fin]
    ns = len(X0)
    # low-ranked seeds (bottom 85 % by seed score) = optimized noise/junk peaks
    lowi = np.argsort(-seed_sc)[int(0.15 * ns):]
    if len(lowi) < 30:
        lowi = np.arange(ns)
    n = nmax // 4
    # random-position tracks: a seed track's shape moved by a random offset to a random point on the grid
    def rand_tracks(m):
        j = rng.integers(0, ns, m)
        X = X0[j] - X0[j].mean(1, keepdims=True); Y = Y0[j] - Y0[j].mean(1, keepdims=True)
        cx = rng.uniform(0.05, 0.95, m) * ctx.W; cy = rng.uniform(0.05, 0.95, m) * ctx.H
        return X + cx[:, None], Y + cy[:, None]
    def low_tracks(m):
        j = lowi[rng.integers(0, len(lowi), m)]
        return X0[j].copy(), Y0[j].copy()
    feats, masks, labs = [], [], []
    CH = 4096
    for kind in range(4):     # 0 neg random, 1 neg low seed, 2 pos random, 3 pos low seed
        for s0 in range(0, n, CH):
            m = min(CH, n - s0)
            X, Y = rand_tracks(m) if kind in (0, 2) else low_tracks(m)
            amp = jit = None
            if kind >= 2:
                if ALOG:   # gen 117: log-uniform S/N up to AMAX so bright tracks are not out of distribution
                    amp = np.exp(rng.uniform(math.log(AMIN), math.log(AMAX), m))
                else:
                    amp = rng.uniform(AMIN, AMAX, m)
                jit = rng.normal(0, 0.45, (m, ctx.N, 2))
                if JITC > 0:   # gen 117: common per-track offset (refined orbits are off by up to ~1.5" as a whole)
                    jit = jit + rng.normal(0, JITC, (m, 1, 2))
            f, nm = ctx.cut(X, Y, amp, jit)
            feats.append(f); masks.append(nm); labs.append(torch.full((m,), float(kind >= 2), device=ctx.dev))
    Fx = torch.cat(feats); Mx = torch.cat(masks); Lx = torch.cat(labs)
    keep = Mx.sum(1) >= 2
    Fx, Mx, Lx = Fx[keep], Mx[keep], Lx[keep]
    perm = torch.randperm(len(Lx), device=ctx.dev)
    Fx, Mx, Lx = Fx[perm], Mx[perm], Lx[perm]
    nte = max(500, len(Lx) // 6)
    Fte, Mte, Lte = Fx[:nte], Mx[:nte], Lx[:nte]
    Ftr, Mtr, Ltr = Fx[nte:], Mx[nte:], Lx[nte:]
    tdata = time.time() - t0
    model = make_model(torch).to(ctx.dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    B = 512; step = 0; ntr = len(Ltr)
    t1 = time.time()
    while time.time() - t0 < tmax and step < 4000:
        idx = torch.randint(0, ntr, (B,), device=ctx.dev)
        lo = lossf(model(Ftr[idx], Mtr[idx]), Ltr[idx])
        opt.zero_grad(); lo.backward(); opt.step()
        step += 1
        if step == 2500:
            for g_ in opt.param_groups:
                g_["lr"] = 5e-4
    model.eval()
    with torch.no_grad():
        q = torch.cat([model(Fte[i:i + 4096], Mte[i:i + 4096]) for i in range(0, nte, 4096)]).cpu().numpy()
    y = Lte.cpu().numpy()
    acc = float(((q > 0) == (y > 0.5)).mean())
    auc = _auc(q, y)
    # held-out AUC split by injected S/N is not tracked (amp not kept); log overall numbers
    log(f"cnn: trained {step} steps on {ntr} samples (data {tdata:.1f}s, train {time.time() - t1:.1f}s), held-out "
        f"{nte}: acc {acc:.3f} AUC {auc:.3f}, final loss {float(lo.detach()):.3f}")
    return model


def _auc(s, y):
    o = np.argsort(s)
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    npos = (y > 0.5).sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y > 0.5].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def score(torch, ctx, orbit, model, S, geo):
    S = np.asarray(S, float)
    X, Y = _pix(orbit, S, *geo)
    X = np.nan_to_num(X, nan=-1e6); Y = np.nan_to_num(Y, nan=-1e6)
    out = []
    with torch.no_grad():
        for i in range(0, len(S), 2048):
            f, nm = ctx.cut(X[i:i + 2048], Y[i:i + 2048])
            out.append(torch.sigmoid(model(f, nm)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def rerank(cands, q, pstrong=0.8, alpha=0.5, lhi=None):
    """Weak candidates (p < pstrong) keep their multiset of p values, reassigned by logit(p) + alpha * logit(q).
    gen 117: logit(q) clipped to [-4, lhi] (TNO_CNN_LHI, default 8; gen106 used 4, which saturated: on inj7d 65 % of the
    refined junk has q > 0.9 and the truth/junk separation lives at logit 5-10)."""
    if lhi is None:
        lhi = float(os.environ.get("TNO_CNN_LHI", "8"))
    p = np.array([c["p"] for c in cands], float)
    weak = np.flatnonzero(p < pstrong)
    if len(weak) < 2:
        return cands, 0
    lg = lambda x, e=1e-4: np.log(np.clip(x, e, 1 - e) / (1 - np.clip(x, e, 1 - e)))
    comb = lg(p[weak]) + alpha * np.clip(lg(q[weak], 1e-7), -4.0, lhi)
    order = weak[np.argsort(-comb)]
    pv = np.sort(p[weak])[::-1]
    out = list(cands)
    base = weak[np.argsort(-p[weak])]
    moved = int((order != base).sum())
    for k, j in enumerate(order):
        out[j] = dict(cands[j], p=float(pv[k]), pcnn=float(q[j]))
    return out, moved
