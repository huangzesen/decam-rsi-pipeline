"""raw.py RAW.pkl DST : planted scored bodies that have a raw linkage (>= 80 % of its detections of one body), by nights of the linkage."""
import sys, pickle, json, numpy as np
raw = pickle.load(open(sys.argv[1], 'rb')); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']
best = {}
for c in raw:
    d = np.asarray(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb) == 0: continue
    u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()])
    if cnt.max() >= 0.8 * len(d):
        k = len(np.unique(night[rexp[d[b == j]]]))
        best[j] = max(best.get(j, 0), k)
sc = set(np.flatnonzero(nn >= 5).tolist())
v = np.array([best[j] for j in best if j in sc])
print(sys.argv[1], 'raw linkages', len(raw), ' scored bodies with a pure raw linkage of >=3 / >=4 / >=5 nights:', int((v >= 3).sum()), int((v >= 4).sum()), int((v >= 5).sum()), 'of', len(sc))
np.save(sys.argv[1] + '.b4.npy', np.array(sorted(j for j in best if j in sc and best[j] >= 4)))
