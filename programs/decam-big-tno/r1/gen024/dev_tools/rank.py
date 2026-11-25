"""rank.py CAND.json : true candidates above the k-th candidate that is not a found body; counts by p."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load('/tmp/p4/truth.npz')
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load('/tmp/p4/det_exp.npy', mmap_mode='r')
cands.sort(key=lambda c: -c['p']); seen = set(); lab = []
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()])
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[dexp[d[b == j]]])) >= 5 and j not in seen:
            ok = True; seen.add(j)
    lab.append(ok)
lab = np.array(lab); bad = np.cumsum(~lab); good = np.cumsum(lab)
print('found', int(lab.sum()), 'true above the k-th non-true, k = 3/5/10/15/20/30/40/60:', [int(good[np.searchsorted(bad, k)]) if bad[-1] >= k else int(good[-1]) for k in (3, 5, 10, 15, 20, 30, 40, 60)])
p = np.array([c['p'] for c in cands])
print('p >= 0.9/0.5/0.3/0.1: true', [int((lab & (p >= t)).sum()) for t in (0.9, 0.5, 0.3, 0.1)], 'not', [int((~lab & (p >= t)).sum()) for t in (0.9, 0.5, 0.3, 0.1)])
