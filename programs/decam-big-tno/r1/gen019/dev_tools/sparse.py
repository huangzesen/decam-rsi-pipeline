"""sparse.py CAND.json DST: candidates by largest number of nights in one season: true / non-matching, with p."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load(sys.argv[2] + '/det_exp.npy', mmap_mode='r')
tab = {}; ex = []
for c in cands[:2000]:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        ok = cnt.max() >= 0.8 * len(d) and len(np.unique(night[dexp[d[b == j]]])) >= 5
    un = np.unique(night[dexp[d]]); g = np.flatnonzero(np.diff(un) > 120); per = sorted([len(p) for p in np.split(un, g + 1)], reverse=True)
    k = ('max<=2' if per[0] <= 2 else 'max3' if per[0] == 3 else 'max>=4', min(len(un), 8))
    t = tab.setdefault(k, [0, 0, [], []]); t[0 if ok else 1] += 1; t[2 if ok else 3].append(c['p'])
    if per[0] <= 3 and not ok: ex.append((round(c['p'], 3), per, len(d), int((b >= 0).sum()), c.get('rms'), c.get('msc'), c.get('ndh'), c.get('ndm'), c.get('src')))
for k in sorted(tab):
    t = tab[k]; print(k, 'true', t[0], 'p med %.2f' % (np.median(t[2]) if t[2] else 0), [round(float(v), 2) for v in np.percentile(t[2], [10, 50, 90])] if t[2] else '', '| non-matching', t[1], [round(float(v), 2) for v in sorted(t[3], reverse=True)[:8]])
for e in sorted(ex, reverse=True)[:14]: print('   ', e)
