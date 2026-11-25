"""For each kept detection: on how many OTHER nights could a body of its magnitude be seen (predicted S/N >= T in an
exposure whose pointing is within RAD deg of the detection)?"""
import sys, json, math, numpy as np
scen, cache = sys.argv[1], sys.argv[2]
T = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
RAD = float(sys.argv[4]) if len(sys.argv) > 4 else 1.3
Z = np.load(cache)
keep, seed, seed2, th, de, flux, snr_k, F = Z['keep'], Z['seed'], Z['seed2'], Z['th'], Z['de'], Z['flux'], Z['snr_k'], Z['F']
E = json.load(open(f'{scen}/exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int); nu, nidx = np.unique(night, return_inverse=True)
band = np.array([e['band'] for e in E])
mz = np.array([e['magzero'] for e in E], float)
col = dict(g=0.6, r=0.0, i=-0.25, z=-0.4, Y=-0.45, VR=0.1); cb = np.array([col.get(b, 0.0) for b in band])
# depth of each exposure from ALL detections: median flux error of faint ones
ex = np.asarray(np.load(f'{scen}/det_exp.npy', mmap_mode='r')); fe = np.load(f'{scen}/det_ferr.npy', mmap_mode='r'); sn = np.load(f'{scen}/det_snr.npy', mmap_mode='r')
st = np.searchsorted(ex, np.arange(nE)); en = np.searchsorted(ex, np.arange(nE), side='right')
sigf = np.full(nE, np.nan)
for k in range(nE):
    s_ = np.asarray(sn[st[k]:en[k]]); f_ = np.asarray(fe[st[k]:en[k]]); m = s_ < 10
    if m.sum() >= 20: sigf[k] = np.median(f_[m])
m5r = mz - 2.5 * np.log10(5 * sigf) - cb          # 5-sigma depth in r-equivalent magnitude of a body with the mean colours
print('r-equivalent depth by band:', {b: np.round(np.nanpercentile(m5r[band == b], [10, 50, 90]), 2).tolist() for b in np.unique(band)})
rp, dp = np.radians([e['ra0'] for e in E]), np.radians([e['dec0'] for e in E])
vp = np.stack([np.cos(dp) * np.cos(rp), np.cos(dp) * np.sin(rp), np.sin(dp)], 1)
den = np.maximum(vp @ F[2], 0.2); pt = np.stack([(vp @ F[0]) / den, (vp @ F[1]) / den], 1)
mag = mz[de] - 2.5 * np.log10(np.maximum(flux, 1e-9)) - cb[de]       # r-equivalent magnitude
n = len(keep); NN = len(nu)
nn_ok = np.zeros(n, np.int16)
o = np.argsort(de, kind='stable'); ds = de[o]; s2 = np.searchsorted(ds, np.arange(nE)); e2 = np.searchsorted(ds, np.arange(nE), side='right')
lim = m5r + 2.5 * np.log10(5.0 / T)                # magnitude seen at S/N T
for A in range(nE):
    ia = o[s2[A]:e2[A]]
    if len(ia) == 0: continue
    cand = np.flatnonzero((np.hypot(pt[:, 0] - pt[A, 0], pt[:, 1] - pt[A, 1]) < math.radians(RAD + 1.1)) & (night != night[A]) & np.isfinite(lim))
    if len(cand) == 0: continue
    # per other night: the deepest limit among exposures within RAD of the detection
    best = np.full((len(ia), NN), -np.inf, np.float32)
    for B in cand:
        near = np.hypot(th[ia, 0] - pt[B, 0], th[ia, 1] - pt[B, 1]) < math.radians(RAD)
        col_ = best[:, nidx[B]]; col_[near] = np.maximum(col_[near], lim[B])
    nn_ok[ia] = (best >= mag[ia][:, None]).sum(1)
np.save('/tmp/w/nn_ok.npy', nn_ok)
print('kept', n)
for nm, m in (('kept', np.ones(n, bool)), ('clean', seed2), ('seed', seed)):
    print(nm, int(m.sum()), 'share with >= k other usable nights, k=1..6:', np.round([(nn_ok[m] >= k).mean() for k in (1, 2, 3, 4, 5, 6)], 3))
Tr = np.load(f'{scen}/truth.npz')
bod = -np.ones(int(Tr['ntot']), np.int64); bod[Tr['rows']] = Tr['body']
b = bod[keep]; pl = b >= 0; sc = np.zeros(n, bool); sc[pl] = Tr['nights'][b[pl]] >= 5
print('planted kept', int(pl.sum()), 'k=1..6:', np.round([(nn_ok[pl] >= k).mean() for k in (1, 2, 3, 4, 5, 6)], 3))
print('planted of scored bodies', int(sc.sum()), 'k=1..6:', np.round([(nn_ok[sc] >= k).mean() for k in (1, 2, 3, 4, 5, 6)], 3))
for lo, hi in ((4.5, 5), (5, 6), (6, 8), (8, 12), (12, 1e9)):
    q = seed2 & (snr_k >= lo) & (snr_k < hi); q2 = sc & (snr_k >= lo) & (snr_k < hi)
    print('S/N %g-%g clean >=4: %.3f  >=2: %.3f | scored planted >=4: %.3f >=2: %.3f' % (lo, hi, (nn_ok[q] >= 4).mean(), (nn_ok[q] >= 2).mean(), (nn_ok[q2] >= 4).mean(), (nn_ok[q2] >= 2).mean()))
