"""Audit of the catalogue around a candidate's orbit (dev tool, gen009).
audit.py CAND.json SCEN [PMIN] [CNT.npy]
For each candidate with p >= PMIN: fit a barycentric two-body orbit to its detections, predict every exposure whose
pointing is within 1.15 deg, and list every catalogue detection within 2 arcsec of the prediction with its S/N,
sharpness, flag, static count (if CNT.npy from sharpstat.py is given) and whether the candidate holds it.
Shows which detections of a real body the cuts of the pipeline throw away."""
import sys, json, math, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/app/submission')
from finder import sun_bary, MU, C_AU
cands = json.load(open(sys.argv[1])); scen = sys.argv[2]
RADX = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
pmin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
cnt = None
E = json.load(open(f'{scen}/exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); ob = np.array([e['obs_xyz'] for e in E]) + sun_bary(et)
night = np.floor(et - 0.5).astype(int)
L = lambda k: np.load(f'{scen}/det_{k}.npy', mmap_mode='r')
ra, dec, snr, sharp, flag, exp = L('ra'), L('dec'), L('snr'), L('sharp'), L('flag'), np.asarray(L('exp'))
st = np.searchsorted(exp, np.arange(nE)); en = np.searchsorted(exp, np.arange(nE), side='right')
ra0 = np.radians([e['ra0'] for e in E]); dec0 = np.radians([e['dec0'] for e in E])
pv = np.stack([np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)], 1)
truth = None
try:
    T = np.load(f'{scen}/truth.npz'); truth = -np.ones(int(T['ntot']), np.int64); truth[T['rows']] = T['body']
except Exception:
    pass


def kep(r0, v0, dt):
    r0n = np.linalg.norm(r0); inva = 2 / r0n - (v0 @ v0) / MU
    n = np.sqrt(MU * inva ** 3); s0 = (r0 @ v0) / math.sqrt(MU) * np.sqrt(inva)
    c1 = 1 - r0n * inva; M = n * dt; Ea = M.copy()
    for _ in range(10):
        Ea = Ea - (Ea - c1 * np.sin(Ea) + s0 * (1 - np.cos(Ea)) - M) / (1 - c1 * np.cos(Ea) + s0 * np.sin(Ea))
    f = 1 - (1 - np.cos(Ea)) / (r0n * inva); g = dt + (np.sin(Ea) - Ea) / n
    return f[:, None] * r0[None] + g[:, None] * v0[None]


def predict(p, t, o, t0):
    r0, v0 = p[:3], p[3:] * 1e-3
    lt = np.full(len(t), np.linalg.norm(r0) / C_AU)
    for _ in range(3):
        rel = kep(r0, v0, t - t0 - lt) - o; lt = np.linalg.norm(rel, axis=1) / C_AU
    return rel / np.linalg.norm(rel, axis=1)[:, None]


tot = dict(n=0, held=0)
rowsout = []
for ic, c in enumerate(cands):
    if c['p'] < pmin:
        continue
    d = np.array(c['det']); e = exp[d]
    if truth is not None and (truth[d] >= 0).any():
        continue
    u = np.stack([np.cos(np.radians(dec[d])) * np.cos(np.radians(ra[d])), np.cos(np.radians(dec[d])) * np.sin(np.radians(ra[d])), np.sin(np.radians(dec[d]))], 1)
    t0 = et[e].mean(); r = float(c.get('r', 40.0))
    eps = math.radians(23.43928); zecl = np.array([0, -math.sin(eps), math.cos(eps)])
    best = None
    for rr in (r, 0.97 * r, 1.03 * r):
        k0 = np.argmin(np.abs(et[e] - t0)); o = ob[e[k0]]; bq = o @ u[k0]; rho = -bq + math.sqrt(bq ** 2 - o @ o + rr ** 2)
        r0 = o + rho * u[k0]; rh = r0 / np.linalg.norm(r0); east = np.cross(zecl, rh); east /= np.linalg.norm(east)
        v0 = math.sqrt(MU / rr) * east
        p0 = np.concatenate([r0, v0 * 1e3])
        fun = lambda p: ((predict(p, et[e], ob[e], t0) - u) * 206265.0).ravel()
        try:
            s = least_squares(fun, p0, x_scale=np.array([1, 1, 1, .1, .1, .1]), loss='soft_l1', f_scale=0.5, max_nfev=400)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    if best is None:
        continue
    res = np.linalg.norm(fun(best.x).reshape(-1, 3), axis=1)
    pr = predict(best.x, et, ob, t0)
    sep = np.degrees(np.arccos(np.clip((pr * pv).sum(1), -1, 1)))
    print(f"\ncandidate {ic}: p {c['p']:.3f}, {c.get('nights')} nights / {c.get('seasons')} seasons, r {c.get('r')}, fit rms {np.sqrt(np.mean(res ** 2)):.2f}\", {len(d)} detections")
    held = set(d.tolist())
    for k in np.flatnonzero(sep < 1.15):
        s_, e_ = st[k], en[k]
        rk = np.radians(np.asarray(ra[s_:e_])); dk = np.radians(np.asarray(dec[s_:e_]))
        uk = np.stack([np.cos(dk) * np.cos(rk), np.cos(dk) * np.sin(rk), np.sin(dk)], 1)
        dd = np.degrees(np.arccos(np.clip(uk @ pr[k], -1, 1))) * 3600
        j = np.flatnonzero(dd < RADX)
        m5 = E[k].get('m5')
        if len(j) == 0:
            # is the prediction on a live CCD?  any detection within 60"
            live = (dd < 60).sum()
            print(f"   exp {k:4d} night {night[k]} {E[k]['band']:>2} {E[k]['exptime']:5.0f}s m5 {m5 if m5 is None else round(m5, 2)}: nothing within 2\" ({'on a CCD' if live else 'off the CCDs'})")
            continue
        for q in j:
            row = s_ + q
            tot['n'] += 1; tot['held'] += row in held
            rowsout.append((float(snr[row]), float(sharp[row]), int(flag[row]), int(cnt[row]) if cnt is not None else -1, row in held))
            print(f"   exp {k:4d} night {night[k]} {E[k]['band']:>2} {E[k]['exptime']:5.0f}s m5 {m5 if m5 is None else round(m5, 2)}: d {dd[q]:.2f}\" S/N {snr[row]:6.1f} sharp {sharp[row]:5.2f} flag {flag[row]} cnt {cnt[row] if cnt is not None else '-'} {'HELD' if row in held else 'not in the candidate'}")
print('\ncatalogue detections within 2" of the tracks:', tot['n'], ' held by the candidates:', tot['held'])
R = np.array(rowsout, float)
if len(R):
    lo = R[:, 0] < 8
    for nm, m in (('S/N < 8', lo), ('S/N >= 8', ~lo)):
        if m.any():
            sh = R[m, 1]
            print(nm, 'n', int(m.sum()), 'sharp outside 0.35-2.5: %.2f, outside 0.5-1.8: %.2f, zero: %.2f' % (((sh < 0.35) | (sh > 2.5)).mean(), ((sh < 0.5) | (sh > 1.8)).mean(), (sh == 0).mean()), ' held %.2f' % R[m, 4].mean())
