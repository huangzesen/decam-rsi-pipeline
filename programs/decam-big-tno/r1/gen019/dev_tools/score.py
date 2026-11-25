import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); night = np.floor(np.array([e['mjd_mid'] for e in E]) - 0.5).astype(int)
body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']; sc = nn >= 5
found = {}; false = []; 
for k, c in enumerate(cands[:2000]):
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5:
            ok = True; found.setdefault(int(j), (c['p'], c.get('src', 0)))
    if not ok:
        false.append((c['p'], c.get('nights'), c.get('seasons'), c.get('src', 0), int((b >= 0).sum()), len(d), c.get('rms'), c.get('r')))
fb = np.array(sorted(found)); 
print(f"candidates {len(cands)}, scored bodies {int(sc.sum())}, found {len(fb)} ({len(fb) / max(sc.sum(), 1):.1%}), not matching {len(false)}")
bins = [(5, 5), (6, 7), (8, 10), (11, 99)]
print('by nights:', '  '.join(f"{a}-{b}: {int(((nn[fb] >= a) & (nn[fb] <= b)).sum())}/{int(((nn >= a) & (nn <= b)).sum())}" for a, b in bins))
src = np.array([found[b][1] for b in fb]); print('found by src:', {int(s): int((src == s).sum()) for s in np.unique(src)})
pf = np.array([f[0] for f in false]); pt = np.array([found[b][0] for b in fb])
for th in (0.9, 0.7, 0.5, 0.3, 0.1):
    print(f"  p>={th}: true {int((pt >= th).sum())} not-matching {int((pf >= th).sum())}")
fs = sorted(false, reverse=True)
print('top not-matching (p, nights, seasons, src, planted dets, ndet, rms, r):'); [print('   ', f) for f in fs[:int(sys.argv[3]) if len(sys.argv) > 3 else 12]]
fq = [f for f in false if f[3]]; print('not-matching from quads:', len(fq), ' with 0 planted dets:', sum(1 for f in fq if f[4] == 0))
np.save('/tmp/found.npy', fb)
# carried candidates: true/false by seed nights and chance level
tr = {}; 
for k, c in enumerate(cands[:2000]):
    if not c.get('src'): continue
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        ok = cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5
    fa = c.get('fa', 0); cl = (c.get('sn', 0), 'fa<1e-6' if fa < 1e-6 else 'fa<1e-4' if fa < 1e-4 else 'fa<1e-3' if fa < 1e-3 else 'fa<1e-2' if fa < 1e-2 else 'fa>=1e-2', 'true' if ok else ('none' if len(bb) == 0 else 'mixed'))
    tr[cl] = tr.get(cl, 0) + 1
for k in sorted(tr): print('  carried', k, tr[k])
