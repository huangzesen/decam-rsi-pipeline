import sys, json, math, numpy as np
cands = json.load(open(sys.argv[1])); scen = '/tmp/p4'; Z = np.load('/tmp/w/cache_p4.npz')
keep, th, de, snr_k, seed2 = Z['keep'], Z['th'], Z['de'], Z['snr_k'], Z['seed2']
AS = math.pi / 180 / 3600
E = json.load(open(f'{scen}/exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
T = np.load(scen + '/truth.npz'); ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
pk = body[keep] >= 0
o = np.argsort(de, kind='stable'); ds = de[o]
st = np.searchsorted(ds, np.arange(nE)); en = np.searchsorted(ds, np.arange(nE), side='right')
R = math.radians(0.1)
def dens(k):
    A = de[k]; i = o[st[A]:en[A]]; i = i[~pk[i]]
    d2 = (th[i, 0] - th[k, 0]) ** 2 + (th[i, 1] - th[k, 1]) ** 2
    near = i[d2 < R * R]
    s = snr_k[k]
    n1 = (snr_k[near] >= 0.7 * s).sum()                      # as bright or brighter (within 0.4 mag below)
    n2 = ((snr_k[near] >= s / 1.6) & (snr_k[near] <= s * 1.6)).sum()
    return n1, n2, len(near)
out = []
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        ok = cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5
    cls = 'true' if ok else ('none' if len(bb) == 0 else 'mixed')
    k = np.searchsorted(keep, d); nn = night[de[k]]
    D = np.array([dens(q) for q in k], float)
    un, inv = np.unique(nn, return_inverse=True)
    l1 = np.log(np.maximum(D[:, 0], 0.5)); l2 = np.log(np.maximum(D[:, 1], 0.5)); l3 = np.log(np.maximum(D[:, 2], 0.5))
    f = lambda l: (np.bincount(inv, weights=l) / np.bincount(inv))
    out.append((cls, c['p'], c['nights'], f(l1).sum(), f(l2).sum(), f(l3).sum(), f(l1).mean(), f(l2).mean(), c.get('s', 0), c.get('src', 0), c.get('fa', 0)))
import pickle; pickle.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else '/tmp/w/dens.pkl', 'wb'))
for cls in ('true', 'none', 'mixed'):
    a = np.array([x[1:] for x in out if x[0] == cls], float)
    if len(a) == 0: continue
    print(cls, len(a))
    for nm, j in (('sum log n(>=0.7s)', 2), ('sum log n(band)', 3), ('sum log n(all)', 4), ('mean log n(>=0.7s)', 5), ('mean log n(band)', 6)):
        print('   %-20s all: %s   <=6 nights: %s' % (nm, np.round(np.percentile(a[:, j], [10, 25, 50, 75, 90]), 2), np.round(np.percentile(a[a[:, 1] <= 6, j], [10, 25, 50, 75, 90]), 2)))
