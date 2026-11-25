import sys, json, numpy as np, math, time
sys.path.insert(0, '/app/submission')
from finder import tangent
from scipy.spatial import cKDTree
src = '/tmp/p4'
Z = np.load('/tmp/w/cache_p4.npz')
keep, seed, th, de, snr_k = Z['keep'], Z['seed'], Z['th'], Z['de'], Z['snr_k']
T = np.load(src + '/truth.npz'); bod = -np.ones(int(T['ntot']), np.int64); bod[T['rows']] = T['body']
E = json.load(open(src + '/exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
m5 = np.array([e['m5'] if e['m5'] is not None else np.nan for e in E], float)
band = [e['band'] for e in E]
exp = np.load(src + '/det_exp.npy').astype(np.int64)
x, y, F = tangent(np.load(src + '/det_ra.npy'), np.load(src + '/det_dec.npy'))
st = np.searchsorted(exp, np.arange(nE)); en = np.searchsorted(exp, np.arange(nE), side='right')
AS = math.radians(1 / 3600)
pool = np.flatnonzero(seed & (snr_k >= 6.0))
pe = de[pool]; px = th[pool]; ptrue = bod[keep[pool]] >= 0
print('pool', len(pool), 'planted', ptrue.sum())
cen = np.array([[x[st[k]:en[k]].mean(), y[st[k]:en[k]].mean()] if en[k] > st[k] else [9, 9] for k in range(nE)])
ncov = np.zeros(len(pool), int); npart = np.zeros(len(pool), int); deep = np.zeros(len(pool), int); t0 = time.time()
trees = {}
for A in range(nE):
    ia = np.flatnonzero(pe == A)
    if not len(ia): continue
    for B in np.flatnonzero((night == night[A]) & (np.arange(nE) != A)):
        if np.hypot(*(cen[A] - cen[B])) > math.radians(2.4) or en[B] == st[B]: continue
        if B not in trees: trees[B] = cKDTree(np.stack([x[st[B]:en[B]], y[st[B]:en[B]]], 1))
        dt = abs(et[A] - et[B]) * 24
        c = np.array([len(v) for v in trees[B].query_ball_point(px[ia], 40 * AS)])
        cov = c >= 4
        d, _ = trees[B].query(px[ia])
        part = d < (1.0 + 5.0 * dt) * AS
        dm = (m5[B] - m5[A]) if np.isfinite(m5[A] + m5[B]) else -9
        ncov[ia] += cov; npart[ia] += cov & part
        deep[ia] += cov & (snr_k[pool[ia]] * 10 ** (0.4 * dm) >= 9) & ~part
print('time', time.time() - t0)
for nm, m in (('all', np.ones(len(pool), bool)), ('planted', ptrue), ('unplanted', ~ptrue)):
    print(nm, m.sum(), 'covered same night %.3f' % (ncov[m] > 0).mean(), 'partner %.3f' % (npart[m] > 0).mean(),
          'veto(deep, no partner in any) %.3f' % ((deep[m] > 0) & (npart[m] == 0)).mean(), 'covered no partner %.3f' % ((ncov[m] > 0) & (npart[m] == 0)).mean())
np.savez('/tmp/w/veto.npz', pool=pool, ncov=ncov, npart=npart, deep=deep)
