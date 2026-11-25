import sys, json, math, numpy as np
T = np.load('/tmp/p4/truth.npz')
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
cands = json.load(open(sys.argv[1]))
def cls_of(c):
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb) == 0: return 'none'
    u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
    return 'true' if (cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5) else 'mixed'
rows = []
for c in cands:
    if c.get('src', 0) and c.get('fa', 0) > 0.2:
        rows.append((c['fa'], cls_of(c), c['p'], c['nights'], c['seasons'], c['sn'], c.get('vl', 0), c['s'], c['rms'], len(c['det'])))
rows.sort()
for r in rows: print('fa=%.2f %-5s p=%.3f nights=%d seasons=%d sn=%d vl=%.2f s=%.2f rms=%.2f ndet=%d' % r)
for lo, hi in ((0.2, 0.5), (0.5, 1.0), (1.0, 2.0)):
    q = [r for r in rows if lo < r[0] <= hi]
    print('fa %.1f-%.1f: true %d, mixed %d, none %d' % (lo, hi, sum(r[1] == 'true' for r in q), sum(r[1] == 'mixed' for r in q), sum(r[1] == 'none' for r in q)))
