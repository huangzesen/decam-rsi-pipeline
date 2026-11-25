"""Soft non-detection statistic (ndz) of true and non-matching candidates.  ndz.py candidates.json [planted dir]"""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1]))
true = np.zeros(len(cands), bool)
if len(sys.argv) > 2:
    T = np.load(sys.argv[2] + '/truth.npz'); E = json.load(open(sys.argv[2] + '/exposures.json'))
    night = np.floor(np.array([e['mjd_mid'] for e in E]) - 0.5).astype(int)
    body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
    rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
    for k, c in enumerate(cands):
        d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
        if len(bb):
            u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
            true[k] = cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5
f = lambda key: np.array([c.get(key, 0) for c in cands], float)
p, nights, ndz, ndx, ndn, ndd, ndh, ndm, s = [f(k) for k in ('p', 'nights', 'ndz', 'ndx', 'ndn', 'ndd', 'ndh', 'ndm', 's')]
print('candidates', len(cands), 'true', int(true.sum()))
for name, m in (('true', true), ('other', ~true)):
    for lo, hi in ((5, 5), (6, 7), (8, 99)):
        q = m & (nights >= lo) & (nights <= hi)
        if q.sum():
            print(f"{name:5s} nights {lo}-{hi}: n {int(q.sum()):4d}  ndz 10/25/50/75/90 % " + ' '.join(f"{v:6.2f}" for v in np.percentile(ndz[q], [10, 25, 50, 75, 90])) +
                  f"   expected nights (median) {np.median(ndx[q]):.1f} of {np.median(ndn[q]):.0f}, member nights there {np.median(ndd[q]):.0f}; hard hits/misses {np.mean(ndh[q]):.1f}/{np.mean(ndm[q]):.1f}")
for thr in (-1, -1.5, -2, -2.5, -3):
    print(f"ndz < {thr}: true {np.mean(ndz[true] < thr) if true.any() else 0:.3f}  other {np.mean(ndz[~true] < thr):.3f}   other with p >= 0.1: {np.mean(ndz[~true & (p >= 0.1)] < thr) if (~true & (p >= 0.1)).any() else 0:.3f} (n {int((~true & (p >= 0.1)).sum())})")
o = np.argsort(-p)
print('top non-matching: p nights seasons r ndz ndx ndn ndd ndh ndm src')
k = 0
for i in o:
    if not true[i]:
        c = cands[i]; print('  ', c['p'], c['nights'], c['seasons'], c['r'], c.get('ndz'), c.get('ndx'), c.get('ndn'), c.get('ndd'), c.get('ndh'), c.get('ndm'), c.get('src'), 'dq', c.get('dq'))
        k += 1
        if k >= int(sys.argv[3]) if len(sys.argv) > 3 else k >= 25: break
