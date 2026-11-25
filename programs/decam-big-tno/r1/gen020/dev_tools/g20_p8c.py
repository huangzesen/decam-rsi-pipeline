import pickle, json, numpy as np
T = np.load('/tmp/p4/truth.npz')
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
r4 = pickle.load(open('/tmp/o/base/raw4.pkl', 'rb')); r5 = pickle.load(open('/tmp/o/base/raw5.pkl', 'rb'))
new = r5[len(r4):]
print('P8 raw linkages', len(new))
pure = {}; none = 0; mixed = 0
for c in new:
    d = np.asarray(c['det']); b = body[d]
    if (b < 0).all(): none += 1
    elif (b >= 0).all() and len(set(b.tolist())) == 1: pure.setdefault(int(b[0]), []).append(len(np.unique(night[rexp[d]])))
    else: mixed += 1
found = set(np.load('/tmp/o/rB/found.npy').tolist())
print('pure bodies', len(pure), 'of them found at the end', sum(b in found for b in pure), 'no planted det', none, 'mixed', mixed)
print('nights of P8 raw linkages (pure):', np.bincount([max(v) for v in pure.values()]))
print('keys of a raw linkage:', sorted(new[0].keys()) if new else None)
