"""rank2.py CAND.json : true candidates above the k-th candidate with NO planted detection (the real-sky chance candidates)."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load('/tmp/p4/truth.npz')
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load('/tmp/p4/det_exp.npy', mmap_mode='r')
cands.sort(key=lambda c: -c['p']); seen = set(); lab = []
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = 0
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()]); ok = 2
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[dexp[d[b == j]]])) >= 5 and j not in seen:
            ok = 1; seen.add(j)
    lab.append(ok)
lab = np.array(lab); bad = np.cumsum(lab == 0); good = np.cumsum(lab == 1)
ks = (3, 4, 5, 7, 10, 15, 20, 30, 40, 60)
print('true above the k-th no-planted candidate, k =', ks, [int(good[np.searchsorted(bad, k)]) if bad[-1] >= k else int(good[-1]) for k in ks])
print('p of the k-th no-planted candidate:', [round(float(cands[int(np.searchsorted(bad, k))]['p']), 3) if bad[-1] >= k else None for k in ks])
if len(sys.argv) > 2:
    for c, l in zip(cands, lab):
        if l == 0 and c['p'] > 0.1: print('  none p %.3f nights %d seasons %d src %d rms %.2f rr %s h3 %s ndh %s' % (c['p'], c['nights'], c['seasons'], c['src'], c['rms'], c.get('rr'), c.get('h3'), c.get('ndh')))
