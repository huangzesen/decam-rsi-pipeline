import sys, json, numpy as np
# usage: incr.py CAND DST BASEFOUND : bodies found by CAND that are not in BASEFOUND, with their night pattern
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz'); base = set(np.load(sys.argv[3]).tolist())
E = json.load(open(sys.argv[2] + '/exposures.json')); et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']
found = {}; nfalse = 0; fal = []
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5:
            ok = True; found.setdefault(int(j), c['p'])
    if not ok:
        nfalse += 1; fal.append((c['p'], c.get('nights'), c.get('seasons'), int((b >= 0).sum()), len(d)))
new = sorted(set(found) - base)
print(f"candidates {len(cands)}, found {len(found)}, not matching {nfalse}, found and not in base: {len(new)}; union with base {len(set(found) | base)}")
ps = np.array([found[b] for b in new]) if new else np.zeros(0)
print('p of the new ones: >=0.9', int((ps >= 0.9).sum()), '>=0.5', int((ps >= 0.5).sum()), '>=0.1', int((ps >= 0.1).sum()), '<0.1', int((ps < 0.1).sum()))
pf = np.array([f[0] for f in fal]) if fal else np.zeros(0)
print('not matching: p>=0.9', int((pf >= 0.9).sum()), '>=0.5', int((pf >= 0.5).sum()), '>=0.1', int((pf >= 0.1).sum()))
# season pattern of the new ones
bi = T['body']; ei = T['exp']
yr = np.floor((et - et.min() + 100) / 365.25).astype(int)
pat = {}
for b in new:
    ee = ei[bi == b]; ns = {}
    for e in ee:
        ns.setdefault(yr[e], set()).add(night[e])
    k = '+'.join(str(v) for v in sorted((len(v) for v in ns.values()), reverse=True))
    pat[k] = pat.get(k, 0) + 1
print('patterns of the new ones:', sorted(pat.items(), key=lambda kv: -kv[1])[:15])
print('top not matching:', sorted(fal, reverse=True)[:8])
