import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); night = np.floor(np.array([e['mjd_mid'] for e in E]) - 0.5).astype(int)
body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']; sc = nn >= 5; vr = np.abs(T['vr'])
found = {}; false = []
for k, c in enumerate(sorted(cands, key=lambda c: -c['p'])[:2000]):
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5:
            ok = True; found.setdefault(int(j), c['p'])
    if not ok:
        false.append((c['p'], c.get('nights'), int((b >= 0).sum()), len(d)))
fb = np.array(sorted(found)); pt = np.array([found[b] for b in fb]); pf = np.array([f[0] for f in false])
print(f"candidates {len(cands)}, scored {int(sc.sum())}, found {len(fb)} ({len(fb) / sc.sum():.1%}), not matching {len(false)}")
edges = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.46]
print('found by |vr|/vc:', '  '.join(f"{a:.2f}-{b:.2f}: {int(((vr[fb] >= a) & (vr[fb] < b)).sum())}/{int((sc & (vr >= a) & (vr < b)).sum())}" for a, b in zip(edges[:-1], edges[1:])))
print('true / not matching at p >=', '  '.join(f"{t}: {int((pt >= t).sum())}/{int((pf >= t).sum())}" for t in (0.9, 0.7, 0.5, 0.3, 0.1)))
np.save(sys.argv[1] + '.found.npy', fb)
if len(sys.argv) > 3:
    f0 = np.load(sys.argv[3]); new = np.setdiff1d(fb, f0); lost = np.setdiff1d(f0, fb)
    print(f"against {sys.argv[3]}: +{len(new)} -{len(lost)}; |vr| of the new ones: {np.round(np.sort(vr[new]), 2).tolist()[:60]}")
    print(f"  |vr| of the lost ones: {np.round(np.sort(vr[lost]), 2).tolist()[:60]}")
