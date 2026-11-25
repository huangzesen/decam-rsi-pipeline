import pickle, sys, json, numpy as np, math
out = pickle.load(open('/tmp/w/dens.pkl', 'rb'))
cands = json.load(open('/tmp/o/base/candidates.json')); Z = np.load('/tmp/w/cache_p4.npz'); keep, snr_k, de = Z['keep'], Z['snr_k'], Z['de']
E = json.load(open('/tmp/p4/exposures.json')); night = np.floor(np.array([e['mjd_mid'] for e in E]) - 0.5).astype(int)
sl = []
for c in cands:
    k = np.searchsorted(keep, np.array(c['det'])); nn = night[de[k]]; un, inv = np.unique(nn, return_inverse=True)
    l = np.log(np.maximum(snr_k[k], 4.5) / 4.5); sl.append((np.bincount(inv, weights=l) / np.bincount(inv)).sum())
sl = np.array(sl)
cls = np.array([x[0] for x in out]); a = np.array([x[1:] for x in out], float)
p = a[:, 0]; nights = a[:, 1]; sumL = a[:, 2]; sumLall = a[:, 4]
tr = cls == 'true'
def report(name, score):
    o = np.argsort(-score); t = tr[o]; bad = np.cumsum(~t); good = np.cumsum(t)
    res = [int(good[np.searchsorted(bad, k)]) if bad[-1] >= k else int(good[-1]) for k in (3, 5, 10, 20, 40)]
    m = nights <= 6
    st, sf = score[m & tr], score[m & ~tr]
    print('%-40s %s   AUC(<=6 nights) %.3f' % (name, res, (st[:, None] > sf[None, :]).mean()))
lo = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
report('p as written', lo)
for w in (0.3, 0.6, 1.0, 1.5): report('+ %.1f * sum log(snr/4.5)' % w, lo + w * sl)
for w in (0.15, 0.3, 0.5): report('+ %.2f * sum(4.8 - log n_all)' % w, lo + w * (4.8 * nights - sumLall))
for w in (0.3,): 
    for w2 in (0.15, 0.3): report('+ %.1f snr + %.2f n_all' % (w, w2), lo + w * sl + w2 * (4.8 * nights - sumLall))
report('+ 0.3 * sum(4.5 - log n(>=0.7s))', lo + 0.3 * (4.5 * nights - sumL))
print('corr sum log snr vs sumL: %.3f' % np.corrcoef(sl, 4.5 * nights - sumL)[0, 1])
