import json, sys, numpy as np
cands = json.load(open(sys.argv[1])); dst = sys.argv[2] if len(sys.argv) > 2 else None
if dst:
    T = np.load(dst + '/truth.npz'); E = json.load(open(dst + '/exposures.json')); night = np.floor(np.array([e['mjd_mid'] for e in E]) - 0.5).astype(int)
    body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']; rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
rows = []
for c in cands:
    ok = None
    if dst:
        d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False; npl = len(bb)
        if len(bb):
            u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
            ok = bool(cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5)
        cls = 'true' if ok else ('none' if npl == 0 else 'mixed')
    else:
        cls = 'unpl'
    rows.append((cls, c['nights'], c['ndh'], c['ndm'], c['ndo'], c['p'], c['seasons'], c['rms'], c['msc']))
for cls in ('true', 'mixed', 'none', 'unpl'):
    R = [r for r in rows if r[0] == cls]
    if not R: continue
    print('==', cls, len(R))
    for lo, hi in ((5, 5), (6, 7), (8, 99)):
        Q = [r for r in R if lo <= r[1] <= hi]
        if not Q: continue
        h = np.array([r[2] for r in Q]); m = np.array([r[3] for r in Q]); o = np.array([r[4] for r in Q])
        print(f'  nights {lo}-{hi}: n={len(Q)}  opp>0: {np.mean(o > 0):.2f}  mean hit {h.mean():.2f} miss {m.mean():.2f} opp {o.mean():.2f};  miss==0: {np.mean(m == 0):.2f} miss>=2: {np.mean(m >= 2):.2f} miss>=3: {np.mean(m >= 3):.2f}  missfrac(h+m>0) {m.sum() / max((h + m).sum(), 1):.2f}')
if len(sys.argv) > 3:
    for r in sorted(rows, key=lambda r: -r[5])[:int(sys.argv[3])]:
        print(r)
