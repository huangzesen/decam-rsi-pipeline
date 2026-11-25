import sys, json, math, numpy as np
T = np.load('/tmp/p4/truth.npz')
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
cands = json.load(open(sys.argv[1])); cands.sort(key=lambda c: -c['p'])
def cls_of(c):
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb) == 0: return 'none'
    u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
    return 'true' if (cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5) else 'mixed'
keys = ['p', 'nights', 'seasons', 'neff', 'rms', 'msc', 'r', 'hv', 'arc', 'src', 'fa', 'ndh', 'ndm', 'ndo', 'ndh2', 'ndm2', 'vl', 'vn', 's']
fmt = lambda c: ' '.join('%s=%s' % (k, (('%.3g' % c[k]) if isinstance(c.get(k), float) else c.get(k))) for k in keys) + ' ndet=%d' % len(c['det'])
cl = [cls_of(c) for c in cands]
print('--- top none'); n = 0
for c, k in zip(cands, cl):
    if k == 'none':
        print(fmt(c)); n += 1
        if n >= 22: break
print('--- lowest true'); 
for c, k in list(zip(cands, cl))[::-1]:
    if k == 'true':
        print(fmt(c)); n += 1
        if n >= 22 + 14: break
