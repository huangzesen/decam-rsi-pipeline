"""one_eval.py CAND.json DST : the candidates of the one-night carry (src 4) against the planted truth: true (>= 80 % of one
body, >= 5 of its nights), partial (the 4-night arc is a body's, the far night is not), chance; with p and chance level."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
allexp = np.load(sys.argv[2] + '/det_exp.npy', mmap_mode='r')
rows = []
for c in cands:
    if c.get('src') != 4:
        continue
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; kind = 'chance'
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        nn = len(np.unique(night[rexp[d[b == j]]]))
        kind = 'TRUE' if (cnt.max() >= 0.8 * len(d) and nn >= 5) else ('partial' if nn >= 3 else 'chance')
    rows.append((kind, round(c['p'], 3), c.get('nights'), c.get('seasons'), round(c.get('fa', -1), 5), len(d), int((b >= 0).sum()), round(c.get('rms', 0), 2), c.get('ndh'), c.get('ndm')))
rows.sort(key=lambda r: -r[1])
print('kind, p, nights, seasons, fa, ndet, planted dets, rms, nd hits, nd misses')
for r in rows: print(r)
for k in ('TRUE', 'partial', 'chance'):
    pp = np.array([r[1] for r in rows if r[0] == k])
    print(k, len(pp), 'p>=0.5/0.3/0.1/0.03:', [(pp >= t).sum() for t in (0.5, 0.3, 0.1, 0.03)])
