import sys, math, numpy as np
sys.path.insert(0, '/tmp/work')
from hashchk import *
def run2(r, vrf, ts, tref_shift=0.0, cell=48.0, hv=180.0, kpar=1.0, pairs=((0, 1), (2, 3)), opp=0.0):
    ts = np.asarray(ts, float)
    vc = math.sqrt(MU / r)
    r0 = np.array([r, 0.0, 0.0]); v0 = np.array([vrf * vc, vc * math.cos(0.2), vc * math.sin(0.2)])
    ob = np.stack([np.cos(w * (ts - opp)), np.sin(w * (ts - opp)), 0 * ts], 1)
    lt = np.full(len(ts), (r - 1) / C_AU)
    for _ in range(3):
        p = kep(r0, v0, ts - lt); rel = p - ob; lt = np.linalg.norm(rel, axis=1) / C_AU
    d = rel / np.linalg.norm(rel, axis=1)[:, None]
    th = np.stack([d[:, 1] / d[:, 0], d[:, 2] / d[:, 0]], 1)
    obf = np.stack([ob[:, 1], ob[:, 2], ob[:, 0]], 1)
    best = (9e9, None)
    for g in 1 / (r * np.exp(np.arange(-200, 201) * 0.0001)):
        ltg = 1 / (C_AU * g); k = 0.5 * MU * g ** 3
        tp = ts - ltg; u = 1 - k * tp * tp
        f = u - g * obf[:, 2]
        phi = th * f[:, None] + g * obf[:, :2]
        xs = []
        for I, J in pairs:
            D = u[I] * tp[J] - u[J] * tp[I]
            al = (phi[I] * tp[J] - phi[J] * tp[I]) / D
            ad = (u[I] * phi[J] - u[J] * phi[I]) / D
            x = np.concatenate([al, ad * hv]) / (cell * AS); x[2] /= kpar
            xs.append(x)
        dd = np.abs(xs[0] - xs[1])
        if dd.max() < best[0]:
            best = (dd.max(), dd, 1 / g)
    return best
print("P8 geometry (two seasons at opposition, tref asymmetric by 30 d), rate cell along the ecliptic widened by kpar")
for kpar in (1, 4, 6, 8):
    print(' kpar', kpar, ' '.join(f"vr {v:.2f}: {run2(40.0, v, [-212.6 - 10, -212.6 + 5, 152.6 - 3, 152.6 + 12], kpar=kpar, opp=152.6)[0]:.2f}" for v in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)))
print("P5 geometry (one season, 128-d window, cell 8, hv 64): pairs (-60,-35) and (30,60), opposition at 0")
for kpar in (1, 2, 4):
    print(' kpar', kpar, ' '.join(f"vr {v:.2f}: {run2(40.0, v, [-60, -35, 30, 60], cell=8.0, hv=64.0, kpar=kpar)[0]:.2f}" for v in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)))
print("P5 geometry: three nights (-50, 0, 50): pairs (0,1) and (1,2)")
for kpar in (1, 2, 4):
    print(' kpar', kpar, ' '.join(f"vr {v:.2f}: {run2(40.0, v, [-50, 0, 50], cell=8.0, hv=64.0, kpar=kpar, pairs=((0, 1), (1, 2)))[0]:.2f}" for v in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)))
print("P5 geometry: three nights (-50, 0, 50): pairs (0,1) and (0,2)")
print(' kpar 1', ' '.join(f"vr {v:.2f}: {run2(40.0, v, [-50, 0, 50], cell=8.0, hv=64.0, pairs=((0, 1), (0, 2)))[0]:.2f}" for v in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)))
print("P3 geometry (64-d window, cell 8, hv 32): nights (-30, 0, 30): pairs (0,1) and (1,2)")
print(' kpar 1', ' '.join(f"vr {v:.2f}: {run2(40.0, v, [-30, 0, 30], cell=8.0, hv=32.0, pairs=((0, 1), (1, 2)))[0]:.2f}" for v in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)))
print("P2 geometry (32-d window, cell 7, hv 16): nights (-15, 0, 15)")
print(' kpar 1', ' '.join(f"vr {v:.2f}: {run2(40.0, v, [-15, 0, 15], cell=7.0, hv=16.0, pairs=((0, 1), (1, 2)))[0]:.2f}" for v in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)))
