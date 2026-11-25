"""byr.py CAND.json DST [edges...] : found share of the scored planted bodies by planted distance."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']; r = T['r']
found = {}; nf = []
for c in cands[:2000]:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5:
            ok = True; found.setdefault(int(j), c['p'])
    if not ok: nf.append(c['p'])
edges = [float(v) for v in sys.argv[3:]] or [0, 1e9]
sc = np.flatnonzero(nn >= 5); f = np.array([b in found for b in sc])
print('found', int(f.sum()), 'of', len(sc), ' candidates', len(cands), ' non-matching', len(nf), 'p>=0.5/0.3/0.1:', [(np.array(nf) >= t).sum() for t in (0.5, 0.3, 0.1)])
for a, b in zip(edges[:-1], edges[1:]):
    m = (r[sc] >= a) & (r[sc] < b)
    print(f'  r {a:5.1f}-{b:5.1f}: {int(f[m].sum()):4d} / {int(m.sum()):4d}  ({f[m].mean() if m.any() else 0:.1%})')
pt = np.array(list(found.values()))
print('  true p>=0.9/0.5/0.3/0.1:', [(pt >= t).sum() for t in (0.9, 0.5, 0.3, 0.1)])
