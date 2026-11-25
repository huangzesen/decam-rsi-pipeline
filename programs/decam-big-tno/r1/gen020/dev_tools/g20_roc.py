import sys, json, numpy as np
T = np.load('/tmp/p4/truth.npz')
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
for f in sys.argv[1:]:
    cands = json.load(open(f)); cands.sort(key=lambda c: -c['p'])
    cls = []; seen = set()
    for c in cands:
        d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False; j = -1
        if len(bb):
            u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()])
            ok = cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5
        if ok and j in seen: cls.append('dup')
        elif ok: cls.append('true'); seen.add(j)
        else: cls.append('none' if len(bb) == 0 else 'mixed')
    cls = np.array(cls); p = np.array([c['p'] for c in cands])
    t = cls == 'true'
    for nm, bad in (('none only', cls == 'none'), ('none+mixed+dup', ~t)):
        cb = np.cumsum(bad); ct = np.cumsum(t)
        res = [int(ct[np.searchsorted(cb, k)]) if cb[-1] >= k else int(ct[-1]) for k in (3, 5, 10, 20, 40)]
        print('%-28s %-15s true before the 3rd/5th/10th/20th/40th: %s  (true %d, none %d, mixed %d, dup %d)' % (f.split('/')[-2], nm, res, t.sum(), (cls == 'none').sum(), (cls == 'mixed').sum(), (cls == 'dup').sum()))
    # top-N: true among the first N = 1.2 x found
    for N in (int(t.sum()), int(1.1 * t.sum())):
        print('     true among the first %d by p: %d' % (N, t[:N].sum()))
    # carried
    src = np.array([c.get('src', 0) for c in cands])
    for sname, m in (('carried true', (src > 0) & t), ('carried none', (src > 0) & (cls == 'none'))):
        print('     %s: n %d, p>=0.5/0.3/0.1/0.03: %s' % (sname, m.sum(), [(p[m] >= q).sum() for q in (0.5, 0.3, 0.1, 0.03)]))
