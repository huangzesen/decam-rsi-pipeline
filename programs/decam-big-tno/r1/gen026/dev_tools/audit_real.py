"""audit_real.py SCEN CANDIDATES.json [PREP_CACHE.npz] : what the catalogue holds along the tracks of the confident linkages.

For every candidate with p >= 0.9 and >= 7 nights (taken as real bodies): a two-body orbit is fitted to its members, the
track is predicted into every exposure of the patch, and the raw catalogue (before any cut of the prep) is searched within
1.5" of the prediction.  Prints, per body and in total: exposures that cover the track, raw detections on it, how many of
those are members, and for the ones that are not members their S/N, sharpness, flag and (with a prep cache of the same
scene, FINDER_DEVCACHE) whether the prep kept them and whether they may seed.  Development tool, not part of the search.
"""
import sys, json, math, numpy as np
sys.path.insert(0, '/app/submission')
from finder import sun_bary, MU, C_AU
from scipy.optimize import least_squares

scen, cf = sys.argv[1], sys.argv[2]
cache = np.load(sys.argv[3]) if len(sys.argv) > 3 else None
E = json.load(open(f'{scen}/exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); ob = np.array([e['obs_xyz'] for e in E]) + sun_bary(et)
band = np.array([e['band'] for e in E]); m5 = np.array([e['m5'] if e['m5'] is not None else np.nan for e in E], float)
mz = np.array([e['magzero'] if e['magzero'] is not None else np.nan for e in E], float)
ra0 = np.radians([e['ra0'] for e in E]); dec0 = np.radians([e['dec0'] for e in E])
pv = np.stack([np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)], 1)
L = lambda k: np.load(f'{scen}/det_{k}.npy', mmap_mode='r')
RA, DEC, EXP, SNR, SH, FL, FX = L('ra'), L('dec'), L('exp'), L('snr'), L('sharp'), L('flag'), L('flux')
st = np.searchsorted(EXP, np.arange(nE)); en = np.searchsorted(EXP, np.arange(nE), side='right')
night = np.floor(et - 0.5).astype(int)
keepset = seedset = None
if cache is not None:
    keepset = np.zeros(len(EXP), bool); keepset[cache['keep']] = True
    seedset = np.zeros(len(EXP), bool); seedset[cache['keep'][cache['seed']]] = True


def kep(r0, v0, dt):
    r0n = np.linalg.norm(r0); inva = 2 / r0n - (v0 @ v0) / MU
    n = math.sqrt(MU * inva ** 3); s0 = (r0 @ v0) / math.sqrt(MU) * math.sqrt(inva)
    c1 = 1 - r0n * inva; M = n * dt; Ea = M.copy()
    for _ in range(10):
        Ea = Ea - (Ea - c1 * np.sin(Ea) + s0 * (1 - np.cos(Ea)) - M) / (1 - c1 * np.cos(Ea) + s0 * np.sin(Ea))
    f = 1 - (1 - np.cos(Ea)) / (r0n * inva); g = dt + (np.sin(Ea) - Ea) / n
    return f[:, None] * r0[None] + g[:, None] * v0[None]


def predict(x, t, o, tref):
    r0, v0 = x[:3], x[3:] * 1e-3
    lt = np.full(len(t), np.linalg.norm(r0) / C_AU)
    for _ in range(3):
        rel = kep(r0, v0, t - tref - lt) - o; lt = np.linalg.norm(rel, axis=1) / C_AU
    return rel / np.linalg.norm(rel, axis=1)[:, None]


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)


cands = json.load(open(cf))
tot = dict(cover=0, raw=0, mem=0, nonmem=0, nonmem_kept=0, nonmem_seedable=0, mem_n=0)
rows_out = []
for ci, c in enumerate(cands):
    if c['p'] < 0.9 or c.get('nights', 0) < 7:
        continue
    d = np.array(sorted(c['det'])); e = np.asarray(EXP[d]); u = unit(np.asarray(RA[d]), np.asarray(DEC[d])); t = et[e]; o = ob[e]
    tref = float(t.mean()); r = float(c.get('r', 40.0))
    def helio(k):
        b = o[k] @ u[k]; rho = -b + math.sqrt(b * b - o[k] @ o[k] + r * r); return o[k] + rho * u[k]
    i0, i1 = int(np.argmin(t)), int(np.argmax(t))
    p0, p1 = helio(i0), helio(i1); v = (p1 - p0) / (t[i1] - t[i0]); pm = p0 + v * (tref - t[i0])
    x0 = np.concatenate([pm, v * 1e3])
    def res(x):
        up = predict(x, t, o, tref)
        return (np.cross(up, u)).ravel() * 206265.0
    sol = least_squares(res, x0, x_scale=np.concatenate([np.full(3, 0.1), np.full(3, 0.01)]))
    rms = float(np.sqrt(np.mean(np.sum(np.cross(predict(sol.x, t, o, tref), u) ** 2, 1))) * 206265)
    # member magnitudes per band (catalogue zero point)
    magm = {}
    for b in np.unique(band[e]):
        k = band[e] == b
        with np.errstate(all='ignore'):
            magm[b] = float(np.nanmedian(mz[e[k]] - 2.5 * np.log10(np.asarray(FX[d[k]], float))))
    up = predict(sol.x, et, ob, tref)
    sep = np.degrees(np.arccos(np.clip((up * pv).sum(1), -1, 1)))
    use = np.flatnonzero((sep < 1.25) & (et > t.min() - 500) & (et < t.max() + 500))
    mem = set(d.tolist()); ncover = nraw = nmem = 0; lines = []
    for k in use:
        a, b = st[k], en[k]
        if b <= a:
            continue
        uu = unit(np.asarray(RA[a:b]), np.asarray(DEC[a:b]))
        ds = np.arccos(np.clip(uu @ up[k], -1, 1)) * 206265.0
        if (ds < 45).sum() < 3:
            continue                                   # not on a live part of the camera
        ncover += 1
        j = np.flatnonzero(ds < 1.5)
        snp = 5 * 10 ** (0.4 * (m5[k] - magm[band[k]])) if band[k] in magm and np.isfinite(m5[k]) else np.nan
        inarc = t.min() - 1 <= et[k] <= t.max() + 1
        if len(j) == 0:
            lines.append((et[k], band[k], 'none', np.nan, snp, np.nan, -1, None, None, inarc))
            continue
        jj = j[np.argmin(ds[j])]; row = a + jj; nraw += 1
        ism = row in mem; nmem += ism
        lines.append((et[k], band[k], 'member' if ism else 'RAW', ds[jj], snp, float(SNR[row]), int(FL[row]), float(SH[row]),
                      (bool(keepset[row]), bool(seedset[row])) if keepset is not None else None, inarc))
    print(f"\ncandidate {ci}: p {c['p']:.3f}, {c.get('nights')} nights, r {r:.2f} au, fit rms {rms:.3f}\", {len(d)} members; "
          f"exposures covering the track {ncover}, raw detections within 1.5\" {nraw}, members among them {nmem}")
    for ln in lines:
        if ln[2] == 'RAW' or (ln[2] == 'none' and np.isfinite(ln[4]) and ln[4] >= 7):
            print(f"   mjd {ln[0]:.2f} {ln[1]:>2s} {ln[2]:6s} sep {ln[3]:.2f}\" predicted S/N {ln[4]:6.1f}  S/N {ln[5]:6.1f} flag {ln[6]} sharp {ln[7]} kept/seedable {ln[8]} in arc {ln[9]}")
    tot['cover'] += ncover; tot['raw'] += nraw; tot['mem'] += nmem; tot['mem_n'] += len(d)
    for ln in lines:
        if ln[2] == 'RAW':
            tot['nonmem'] += 1
            if ln[8] is not None:
                tot['nonmem_kept'] += ln[8][0]; tot['nonmem_seedable'] += ln[8][1]
    rows_out += [(ci,) + ln for ln in lines]
print('\ntotal:', tot)
import pickle
pickle.dump(rows_out, open('/tmp/w/audit_rows.pkl', 'wb'))
