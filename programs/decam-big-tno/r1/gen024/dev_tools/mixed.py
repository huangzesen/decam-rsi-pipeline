import sys, json, numpy as np, collections
cands = json.load(open(sys.argv[1])); T = np.load('/tmp/p4/truth.npz')
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load('/tmp/p4/det_exp.npy', mmap_mode='r')
found = set(np.load(sys.argv[2]).tolist())
res = collections.Counter(); ex = []
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if not len(bb): continue
    u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
    tn = len(np.unique(night[dexp[d[b == j]]])); an = len(np.unique(night[dexp[d]]))
    pure = cnt.max() >= 0.8 * len(d)
    if pure and tn >= 5: continue
    if int(j) in found: res['body found by another candidate'] += 1; continue
    key = 'true nights %d of %d nights, %d of %d dets' % (tn, an, cnt.max(), len(d))
    res['true nights >=5, impure' if tn >= 5 else 'true nights %d' % tn] += 1
    if tn >= 4: ex.append((round(c['p'], 3), key, c.get('src')))
for k in sorted(res): print(k, res[k])
for e in sorted(ex, reverse=True)[:25]: print(e)
