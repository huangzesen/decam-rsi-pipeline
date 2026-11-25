"""evalr.py CAND.json DST : found share by distance bin and nights (dev tool)."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']; r = T['r']
found = {}
for c in cands[:2000]:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5:
            found.setdefault(int(j), c['p'])
sc = np.flatnonzero(nn >= 5); f = np.array([b in found for b in sc])
print('scored', len(sc), 'found', f.sum())
for lo, hi in ((5, 6), (6, 8), (8, 99)):
    m = (nn[sc] >= lo) & (nn[sc] < hi); print(f'  nights {lo}-{hi - 1}: {f[m].sum()} / {m.sum()}')
