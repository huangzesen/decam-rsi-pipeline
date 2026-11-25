"""roc.py CAND.json DST [key] : planted bodies found before the k-th candidate that matches no planted body (ranked by `key`, default p).
The candidates with no planted detection at all at the top are the real bodies of the sample: they are left out of the count."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz'); key = sys.argv[3] if len(sys.argv) > 3 else 'p'
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load(sys.argv[2] + '/det_exp.npy', mmap_mode='r')
cands = sorted(cands, key=lambda c: -c.get(key, c['p']))
seen = set(); nf = 0; out = {}
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()])
        ok = cnt.max() >= 0.8 * len(d) and len(np.unique(night[dexp[d[b == j]]])) >= 5
    if ok:
        seen.add(j)
    elif not (len(bb) == 0 and c.get(key, c['p']) > 0.9):
        nf += 1
        if nf in (3, 5, 10, 20, 40, 80): out[nf] = len(seen)
print(f'ranked by {key}: planted bodies found before the k-th non-matching candidate:', out, ' all:', len(seen), ' non-matching:', nf)
