"""gen141 armC toy: compare per-frame samplers of the 1" (4x4 max-pooled matched-filter S/N) maps.
Usage: python -I gen141_natinterp_toy.py SEED SN1"""
import math, sys, numpy as np, torch
torch.set_num_threads(4)
F = torch.nn.functional
dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
PXS = 0.263; NF = 12; SN1 = float(sys.argv[2]) if len(sys.argv) > 2 else 1.6
NPIX = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
NNULL = 4000


def gauss1d(s):
    k = int(math.ceil(3 * s)); x = np.arange(-k, k + 1)
    return np.exp(-x ** 2 / (2 * s * s)).astype(np.float32), k


def frame(fwhm_px, rot, off, bodies_sky):
    s = fwhm_px / 2.355
    imt = torch.from_numpy(rng.standard_normal((NPIX, NPIX)).astype(np.float32)).to(dev)
    c, sn = math.cos(rot), math.sin(rot)
    A = np.array([[c, -sn], [sn, c]]) / PXS
    det = bodies_sky @ A.T + off
    flux = SN1 * math.sqrt(4 * math.pi * s * s)
    X = torch.arange(NPIX, device=dev, dtype=torch.float32)
    for (dx, dy) in det:
        if not (14 < dx < NPIX - 14 and 14 < dy < NPIX - 14):
            continue
        ix, iy = int(dx), int(dy)
        x0, x1, y0, y1 = ix - 12, ix + 13, iy - 12, iy + 13
        gx = torch.exp(-(X[x0:x1] - float(dx)) ** 2 / (2 * s * s)); gy = torch.exp(-(X[y0:y1] - float(dy)) ** 2 / (2 * s * s))
        imt[y0:y1, x0:x1] += flux / (2 * math.pi * s * s) * gy[:, None] * gx[None, :]
    g, k = gauss1d(s)
    R = imt[None, None]
    gx_, gy_ = torch.from_numpy(g[None, :])[None, None].to(dev), torch.from_numpy(g[:, None])[None, None].to(dev)
    snr = F.conv2d(F.conv2d(R, gx_, padding=(0, k)), gy_, padding=(k, 0))
    s2 = snr[0, 0, 3::11, 3::11].flatten(); m2 = s2.median(); ns = 1.4826 * (s2 - m2).abs().median()
    snr = (snr - m2) / ns
    pm = F.max_pool2d(snr, 4, 4, ceil_mode=True)[0, 0]
    ph, pw = pm.shape
    Hs = int(NPIX * PXS) - 4
    jj, ii = np.meshgrid(np.arange(Hs), np.arange(Hs))
    sky = np.stack([jj.ravel(), ii.ravel()], 1).astype(float)
    d = sky @ A.T + off
    ux = (d[:, 0] - 1.5) / 4; uy = (d[:, 1] - 1.5) / 4
    cu = np.floor(ux + 0.5).astype(int); cv = np.floor(uy + 0.5).astype(int)
    ins = (cu >= 0) & (cu < pw) & (cv >= 0) & (cv < ph)
    pmn = pm.float().cpu().numpy()
    v = np.where(ins, pmn[cv.clip(0, ph - 1), cu.clip(0, pw - 1)], np.nan).reshape(Hs, Hs)
    cov = np.isfinite(v)
    sm = v[::7, ::7][cov[::7, ::7]]
    med = np.median(sm); sd = 1.4826 * np.median(np.abs(sm - med))
    v = np.where(cov, (v - med) / sd, 0.0)
    v = np.clip(v.astype(np.float16).astype(np.float32), -3, 5)
    return torch.from_numpy(v).to(dev)


def samp(m, u, v, mode, sig=0.7, clip=3.0):
    H, W = m.shape
    mc = m.clamp(-clip, clip)
    if mode == "nn":
        cu = torch.floor(u + 0.5).long().clamp(0, W - 1); cv = torch.floor(v + 0.5).long().clamp(0, H - 1)
        return mc[cv, cu]
    if mode == "bl":
        u0 = torch.floor(u); v0 = torch.floor(v); fu = u - u0; fv = v - v0
        out = 0
        for a, wa in ((0, 1 - fu), (1, fu)):
            for b, wb in ((0, 1 - fv), (1, fv)):
                out = out + wa * wb * mc[(v0 + b).long().clamp(0, H - 1), (u0 + a).long().clamp(0, W - 1)]
        return out
    if mode.startswith("g"):
        r = int(mode[1]) // 2
        cu = torch.floor(u + 0.5); cv = torch.floor(v + 0.5)
        out = 0; ws = 0
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                w = torch.exp(-((cu + a - u) ** 2 + (cv + b - v) ** 2) / (2 * sig * sig))
                out = out + w * mc[(cv + b).long().clamp(0, H - 1), (cu + a).long().clamp(0, W - 1)]
                ws = ws + w
        return out / ws
    if mode == "mx2":
        u0 = torch.floor(u); v0 = torch.floor(v)
        vals = [mc[(v0 + b).long().clamp(0, H - 1), (u0 + a).long().clamp(0, W - 1)] for a in (0, 1) for b in (0, 1)]
        return torch.stack(vals).max(0).values


MODES = [("nn", None), ("bl", None), ("g3", 0.5), ("g3", 0.7), ("g3", 1.0), ("g5", 0.9), ("mx2", None)]
GRID = [(a * 0.5, b * 0.5) for a in range(-2, 3) for b in range(-2, 3)]
tr = {m: [] for m in MODES}; nu = {m: [] for m in MODES}
trg = {m: [] for m in MODES}; nug = {m: [] for m in MODES}
for f in range(NF):
    fwhm = rng.uniform(3.4, 5.6); rot = math.radians(rng.uniform(-1, 1)); off = rng.uniform(0, 4, 2) + 4
    Hs = int(NPIX * PXS) - 4
    cen = np.array([(15 + 25 * i, 15 + 25 * j) for i in range(40) for j in range(40) if 15 + 25 * max(i, j) < Hs - 15], float)
    bod = cen + rng.uniform(-0.5, 0.5, cen.shape)
    m = frame(fwhm, rot, off, bod)
    nul = np.concatenate([cen + np.array([12.0, 0]) + rng.uniform(-3, 3, cen.shape) for _ in range(NNULL // len(cen) + 1)])[:NNULL]
    cal = rng.uniform(4, Hs - 4, (20000, 2))
    for md in MODES:
        mode, sg = md
        def S(p, dx=0.0, dy=0.0):
            return samp(m, torch.tensor(p[:, 0] + dx, dtype=torch.float32, device=dev),
                        torch.tensor(p[:, 1] + dy, dtype=torch.float32, device=dev), mode, sg or 0.7)
        c_ = S(cal); mu = float(c_.mean()); sd = float(c_.std())
        tr[md].append(((S(bod) - mu) / sd).cpu().numpy()); nu[md].append(((S(nul) - mu) / sd).cpu().numpy())
        trg[md].append(np.stack([((S(bod, dx, dy) - mu) / sd).cpu().numpy() for dx, dy in GRID]))
        nug[md].append(np.stack([((S(nul, dx, dy) - mu) / sd).cpu().numpy() for dx, dy in GRID]))
print(f"per-frame matched-filter S/N {SN1}, {NF} frames, pixel {PXS}, FWHM 3.4-5.6 px, {len(tr[MODES[0]][0])} bodies/frame")
for md in MODES:
    T = np.sum(tr[md], 0) / math.sqrt(NF); Nn = np.sum(nu[md], 0) / math.sqrt(NF)
    Tg = (np.sum(trg[md], 0) / math.sqrt(NF)).max(0); Ng = (np.sum(nug[md], 0) / math.sqrt(NF)).max(0)
    z = (T.mean() - Nn.mean()) / Nn.std(); zg = (Tg.mean() - Ng.mean()) / Ng.std()
    q = np.quantile(Ng, 0.99); q2 = np.quantile(Ng, 0.999)
    print(f"{md[0]:4s} sig={md[1]}: exact-pos Z={z:.2f}  grid25-max Z={zg:.2f}  P(true>null q99)={np.mean(Tg > q):.2f}  q99.9={np.mean(Tg > q2):.2f}")
