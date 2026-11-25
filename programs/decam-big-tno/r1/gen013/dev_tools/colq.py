"""Found share of the planted bodies by their redness (plant_colour.py / plant_zp.py sets).  colq.py candidates.json planted_dir seed"""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz'); seed = int(sys.argv[3])
E = json.load(open(sys.argv[2] + '/exposures.json')); night = np.floor(np.array([e['mjd_mid'] for e in E]) - 0.5).astype(int)
body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']; NB = len(nn)
q = np.random.default_rng(seed + 1000).random(NB)
found = np.zeros(NB, bool); pf = np.zeros(NB); dq = np.full(NB, np.nan); msc = np.full(NB, np.nan)
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5 and not found[j]:
            found[j] = True; pf[j] = c['p']; dq[j] = c.get('dq', np.nan); msc[j] = c.get('msc', np.nan)
sc = nn >= 5
for lo, hi in ((0, 0.15), (0.15, 0.35), (0.35, 0.65), (0.65, 0.85), (0.85, 1.0)):
    m = sc & (q >= lo) & (q < hi)
    print(f"redness {lo:.2f}-{hi:.2f}: found {int(found[m].sum()):3d} of {int(m.sum()):3d} ({found[m].mean():.1%}), at p >= 0.7: {int((pf[m] >= 0.7).sum()):3d}, median msc {np.nanmedian(msc[m]):.3f}, median fitted dq {np.nanmedian(dq[m]):+.2f}")
print('all: found', int(found[sc].sum()), 'of', int(sc.sum()))
