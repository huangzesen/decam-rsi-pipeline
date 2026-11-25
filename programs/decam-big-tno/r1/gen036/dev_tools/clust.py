import numpy as np, json, math
from scipy.spatial import cKDTree
Z = np.load('/tmp/w/cache_p4.npz'); seed, th, de, snr = Z['seed'], Z['th'], Z['de'], Z['snr_k']
E = json.load(open('/tmp/p4/exposures.json')); et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
AS = math.radians(1 / 3600)
pool = np.flatnonzero(seed & (snr >= 6)); pn = night[de[pool]]
un, cnt = np.unique(pn, return_counts=True)
tot = np.zeros(6); exp_ = np.zeros(6); rs = np.array([3, 10, 30, 100, 300, 1000.0])
npair = 0
for i in range(len(un)):
    for j in range(i + 1, len(un)):
        if not (5 <= un[j] - un[i] <= 60): continue
        A = th[pool[pn == un[i]]]; B = th[pool[pn == un[j]]]
        ta = cKDTree(A); tb = cKDTree(B)
        c = ta.count_neighbors(tb, rs * AS)
        if c[-1] < 50: continue
        # local expectation from the 1000" count (area scaling)
        tot += c; exp_ += c[-1] * (rs / rs[-1]) ** 2; npair += 1
        if npair >= 60: break
    if npair >= 60: break
print('night pairs', npair); print('r arcsec', rs); print('pairs within r', tot); print('ratio to uniform scaled from 1000"', np.round(tot / exp_, 2))
