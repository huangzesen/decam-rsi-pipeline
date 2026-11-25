import numpy as np, json, time
from scipy.spatial import cKDTree
Z = np.load('/tmp/w/cache_s.npz')
th, de, seed, seed2, snr = Z['th'], Z['de'], Z['seed'], Z['seed2'], Z['snr_k']
E = json.load(open('/app/data/sample/exposures.json'))
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
AS = np.pi / 180 / 3600
nd = night[de]
t0 = time.time()
# same-night partner in another exposure within vmax*dt + 0.5"
best = np.full(len(de), np.inf); bdt = np.zeros(len(de))
npair = 0
for n in np.unique(night):
    ex = np.flatnonzero(night == n)
    if len(ex) < 2: continue
    idx = np.flatnonzero(nd == n)
    tr = cKDTree(th[idx] / AS)
    pr = tr.query_pairs(12.0, output_type='ndarray')
    if len(pr) == 0: continue
    a, b = idx[pr[:, 0]], idx[pr[:, 1]]
    m = de[a] != de[b]
    a, b = a[m], b[m]
    d = np.hypot(*(th[a] - th[b]).T) / AS
    dt = np.abs(et[de[a]] - et[de[b]]) * 24
    ok = d <= 4.5 * dt + 0.6
    a, b, d, dt = a[ok], b[ok], d[ok], dt[ok]
    npair += len(a)
    for u, v in ((a, b), (b, a)):
        o = np.argsort(-d)
        best[u[o]] = d[o]; bdt[u[o]] = dt[o]
has = np.isfinite(best)
print('time', time.time() - t0)
print('kept', len(de), 'with same-night partner', has.sum(), has.mean())
print('seeds', seed.sum(), 'with partner', (has & seed).sum(), (has & seed).mean() / seed.mean())
print('seed2', seed2.sum(), 'with partner', (has & seed2).sum())
both = has & seed2
print('dt hist (h) of seed2 partners', np.histogram(bdt[both], [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 12])[0])
print('sep hist of seed2 partners', np.histogram(best[both], [0, 0.3, 0.6, 1, 2, 4, 8, 12, 50])[0])
# exposures with a same-night overlapping exposure
np.save('/tmp/w/has_s.npy', has)
