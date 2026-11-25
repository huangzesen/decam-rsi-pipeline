import sys, pickle, json, numpy as np
out, P = sys.argv[1], sys.argv[2]
T = np.load(P + '/truth.npz'); E = json.load(open(P + '/exposures.json'))
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']
D = pickle.load(open(out + '/carry.pkl', 'rb')); dbg = D['dbg']
bo = np.argsort(T['body'], kind='stable'); bs = T['body'][bo]
rows = []
for s in dbg:
    d = np.asarray(s['det']); b = body[d]; bb = b[b >= 0]
    if len(bb) < 0.8 * len(d): rows.append(None); continue
    u, c = np.unique(bb, return_counts=True); j = u[c.argmax()]
    if c.max() < 0.8 * len(d): rows.append(None); continue
    ex = T['exp'][bo[np.searchsorted(bs, j):np.searchsorted(bs, j, side='right')]]
    sn = np.unique(night[rexp[d[b == j]]]); tn = np.unique(night[ex])
    tmin, tmax = et[rexp[d]].min(), et[rexp[d]].max()
    far = [n for n in tn if n not in sn and tmin - 800 <= n <= tmax + 800]
    rows.append((int(j), len(sn), len(far)))
true = [(s, r) for s, r in zip(dbg, rows) if r is not None and nn[r[0]] >= 5]
false = [s for s, r in zip(dbg, rows) if r is None]
print('short arcs', len(dbg), ' of scored planted bodies', len(true), ' no planted body', len(false))
print('distinct bodies among true seeds', len(set(r[0] for s, r in true)))
import collections
for far in (1, 2, 3, 4, 5):
    S = [(s, r) for s, r in true if (r[2] == far if far < 5 else r[2] >= 5)]
    if not S: continue
    print(f' far nights {far}: seeds {len(S)}  cnt>=far {sum(s["cnt"] >= far for s, r in S)}  m>=far {sum(s.get("m", 0) >= far for s, r in S)}  m>=2 {sum(s.get("m", 0) >= 2 for s, r in S)} acc {sum(bool(s.get("acc")) for s, r in S)}',
          ' median ncell %.0f mu %.3f fa %.3g K2 %d nb %d m1 %.0f" m2 %.0f"' % tuple(np.median([s[k] if k in s else np.nan for s, r in S]) for k in ('ncell', 'mu', 'fa', 'K2', 'nb', 'm1', 'm2')))
print('false seeds: cnt hist', np.bincount([min(s['cnt'], 8) for s in false]), ' m hist', np.bincount([int(min(s.get('m', 0), 8)) for s in false]), 'acc', sum(bool(s.get('acc')) for s in false))
print('false seeds median ncell %.0f mu %.3f m1 %.0f m2 %.0f smaj %.4f smin %.4f' % tuple(np.median([s.get(k, np.nan) for s in false]) for k in ('ncell', 'mu', 'm1', 'm2', 'smaj', 'smin')))
tt = [s for s, r in true]
print('true seeds: m1 pct', np.percentile([s['m1'] for s in tt], [10, 50, 90]).round(0), 'm2 pct', np.percentile([s['m2'] for s in tt], [10, 50, 90]).round(0), 'ncell pct', np.nanpercentile([s.get('ncell', np.nan) for s in tt], [10, 50, 90]).round(0), 'mu pct', np.percentile([s['mu'] for s in tt], [10, 50, 90]).round(4))
