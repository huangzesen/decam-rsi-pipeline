import pickle, sys, numpy as np, math
out = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/w/dens.pkl', 'rb'))
cls = np.array([x[0] for x in out]); a = np.array([x[1:] for x in out], float)
p = a[:, 0]; nights = a[:, 1]; sumL = a[:, 2]; meanL = a[:, 5]; s = a[:, 7]; src = a[:, 8]
tr = cls == 'true'
def report(name, score):
    o = np.argsort(-score); t = tr[o]; bad = np.cumsum(~t); good = np.cumsum(t)
    res = [int(good[np.searchsorted(bad, k)]) if bad[-1] >= k else int(good[-1]) for k in (3, 5, 10, 20, 40)]
    # top-N purity: true among first N, N = n_true and 1.25 n_true... and AUC over <=6 nights
    m = nights <= 6
    st, sf = score[m & tr], score[m & ~tr]
    auc = (st[:, None] > sf[None, :]).mean() if len(st) and len(sf) else float('nan')
    print('%-34s true before the 3rd/5th/10th/20th/40th non-true: %s   AUC(<=6 nights) %.3f' % (name, res, auc))
lo = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
report('p as written', lo)
for L0 in (4.0, 4.5):
    for w in (0.15, 0.3, 0.5, 0.8):
        report('+ w=%.2f sum(L0=%.1f - log n)' % (w, L0), lo + w * (L0 * nights - sumL))
for w in (0.5, 1.0, 2.0):
    report('+ w=%.1f * (4.3 - mean log n)' % w, lo + w * (4.3 - meanL))
    report('+ w=%.1f * clip(4.3 - mean log n)*5' % w, lo + w * 5 * np.clip(4.3 - meanL, -1.0, 1.5))
