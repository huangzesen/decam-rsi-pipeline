import sys, math, numpy as np
sys.path.insert(0, '/app/submission')
from finder import MU, C_AU, AS
def kep(r0, v0, dt):
    r0n = np.linalg.norm(r0); inva = 2 / r0n - (v0 @ v0) / MU
    n = math.sqrt(MU * inva ** 3); s0 = (r0 @ v0) / math.sqrt(MU) * math.sqrt(inva)
    c1 = 1 - r0n * inva; M = n * dt; Ea = M.copy()
    for _ in range(12):
        Ea = Ea - (Ea - c1 * np.sin(Ea) + s0 * (1 - np.cos(Ea)) - M) / (1 - c1 * np.cos(Ea) + s0 * np.sin(Ea))
    f = 1 - (1 - np.cos(Ea)) / (r0n * inva); g = dt + (np.sin(Ea) - Ea) / n
    return f[:, None] * r0[None] + g[:, None] * v0[None]
w = 2 * math.pi / 365.25
def run(r, vrf, elong_off, t1, t2, cell=48.0, hv=180.0, gd0=0.0, verbose=False):
    # season oppositions at t = -182.6 and +182.6; observations offset by elong_off days from opposition
    ts = np.array([-182.6 + elong_off + t1[0], -182.6 + elong_off + t1[1], 182.6 + elong_off + t2[0], 182.6 + elong_off + t2[1]])
    vc = math.sqrt(MU / r)
    r0 = np.array([r, 0.0, 0.0]); v0 = np.array([vrf * vc, vc * math.cos(0.2), vc * math.sin(0.2)])
    ob = np.stack([np.cos(w * (ts + 182.6)), np.sin(w * (ts + 182.6)), 0 * ts], 1)
    lt = np.full(len(ts), (r - 1) / C_AU)
    for _ in range(3):
        p = kep(r0, v0, ts - lt); rel = p - ob; lt = np.linalg.norm(rel, axis=1) / C_AU
    d = rel / np.linalg.norm(rel, axis=1)[:, None]
    # frame: F2 = x, F0 = y, F1 = z
    th = np.stack([d[:, 1] / d[:, 0], d[:, 2] / d[:, 0]], 1)
    obf = np.stack([ob[:, 1], ob[:, 2], ob[:, 0]], 1)
    best = (9e9, None)
    for g in 1 / (r * np.exp(np.arange(-80, 81) * 0.0002)):
        ltg = 1 / (C_AU * g); k = 0.5 * MU * g ** 3
        tp = ts - ltg; u = 1 - k * tp * tp
        ncirc = math.sqrt(MU * g ** 3)
        f = u - g * obf[:, 2] + gd0 * ncirc * tp
        phi = th * f[:, None] + g * obf[:, :2]
        xs = []
        for I, J in ((0, 1), (2, 3)):
            D = u[I] * tp[J] - u[J] * tp[I]
            al = (phi[I] * tp[J] - phi[J] * tp[I]) / D
            ad = (u[I] * phi[J] - u[J] * phi[I]) / D
            xs.append(np.concatenate([al, ad * hv]) / (cell * AS))
        dd = np.abs(xs[0] - xs[1])
        if dd.max() < best[0]:
            best = (dd.max(), dd, 1 / g)
    return best
for eo in (0.0, 40.0):
    for vrf in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
        b = run(40.0, vrf, eo, (-10.0, 5.0), (-3.0, 12.0))
        print(f"elong offset {eo:5.1f} d  vr/vc {vrf:4.2f}: min over g of max |dx| = {b[0]:.2f} cells, dx = {np.round(b[1], 2)}, r = {b[2]:.3f}")
print('with gd0 = 0.15, body vr 0.15 / 0.2 / 0.1:')
for vrf in (0.1, 0.15, 0.2, 0.25):
    b = run(40.0, vrf, 0.0, (-10.0, 5.0), (-3.0, 12.0), gd0=0.15)
    print(f"  vr/vc {vrf:4.2f}: {b[0]:.2f} cells, dx = {np.round(b[1], 2)}")
