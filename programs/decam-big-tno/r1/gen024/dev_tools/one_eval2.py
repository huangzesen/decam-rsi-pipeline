import sys, pickle, json, numpy as np
D = pickle.load(open(sys.argv[1] + '/carry.pkl', 'rb')); new = D['new']
T = np.load('/tmp/p4/truth.npz'); ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
snr = np.load('/tmp/p4/det_snr.npy', mmap_mode='r')
rows = []
for c in new:
    if c.get('src') != 4: continue
    d = np.asarray(c['det']); b = body[d]; s = np.asarray(snr[d], float)
    pure = (b >= 0).all() and (b == b[0]).all(); npl = int((b >= 0).sum())
    rexp = float(np.sqrt(2 * np.mean((1.1 / np.maximum(s, 4.5)) ** 2 + 0.04 ** 2)))
    rows.append(('pure' if pure else 'none' if npl == 0 else 'mixed', c['fa'], c['rms'], rexp, c['rms'] / rexp, float(np.median(s)), float(s.min())))
rows.sort(key=lambda r: r[1])
for r in rows:
    if r[1] < 0.3: print('%-5s fa %.4f rms %.2f rexp %.2f ratio %.2f medsnr %.1f minsnr %.1f' % r)
for lim in (0.5, 0.6, 0.7, 0.8, 1.0):
    for fthr in (0.03, 0.05, 0.1, 0.2, 0.3):
        q = [r for r in rows if r[1] < fthr and r[4] <= lim]
        print('ratio<=%.1f fa<%.2f: pure %d none %d mixed %d' % (lim, fthr, sum(r[0] == 'pure' for r in q), sum(r[0] == 'none' for r in q), sum(r[0] == 'mixed' for r in q)))
