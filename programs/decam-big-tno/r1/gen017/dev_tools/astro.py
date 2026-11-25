import numpy as np, json
from scipy.spatial import cKDTree
d = '/app/data/sample/'
ra = np.load(d + 'det_ra.npy', mmap_mode='r'); dec = np.load(d + 'det_dec.npy', mmap_mode='r')
snr = np.load(d + 'det_snr.npy', mmap_mode='r'); exp = np.load(d + 'det_exp.npy', mmap_mode='r'); fl = np.load(d + 'det_flag.npy', mmap_mode='r')
sh = np.load(d + 'det_sharp.npy', mmap_mode='r')
E = json.load(open(d + 'exposures.json')); band = np.array([e['band'] for e in E])
r = np.asarray(ra); de = np.asarray(dec)
m = (np.abs(r - 28.9) < 1.2) & (np.abs(de - 17.6) < 1.2)
i = np.flatnonzero(m); print('box detections', len(i))
x = (r[i] - 28.9) * np.cos(np.radians(17.6)) * 3600; y = (de[i] - 17.6) * 3600
s = np.asarray(snr)[i]; e = np.asarray(exp)[i]; f = np.asarray(fl)[i]; h = np.asarray(sh)[i]
ref = np.flatnonzero((s > 60) & (f == 0) & (h > 0.7) & (h < 1.3))
# reference position of a star: mean over its bright detections (cluster by 0.5")
t = cKDTree(np.stack([x[ref], y[ref]], 1))
pairs = t.query_pairs(0.6, output_type='ndarray')
# union-find light: use connected components
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
n = len(ref)
A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
nc, lab = connected_components(A, directed=False)
cntc = np.bincount(lab, minlength=nc)
mx = np.bincount(lab, weights=x[ref], minlength=nc) / cntc; my = np.bincount(lab, weights=y[ref], minlength=nc) / cntc
good = cntc >= 3
print('reference stars', good.sum())
tr = cKDTree(np.stack([mx[good], my[good]], 1))
dd, j = tr.query(np.stack([x, y], 1), distance_upper_bound=2.0)
ok = np.isfinite(dd) & (f == 0)
dx = x[ok] - mx[good][j[ok]]; dy = y[ok] - my[good][j[ok]]; so = s[ok]; eo = e[ok]
# bright stars: scatter of high S/N detections around the mean = systematic floor
for lo, hi in ((4.5, 5), (5, 6), (6, 8), (8, 12), (12, 25), (25, 60), (60, 200), (200, 1e9)):
    q = (so >= lo) & (so < hi)
    if q.sum() < 20: continue
    rr = np.hypot(dx[q], dy[q])
    sx = 0.5 * (np.percentile(dx[q], 84) - np.percentile(dx[q], 16)); sy = 0.5 * (np.percentile(dy[q], 84) - np.percentile(dy[q], 16))
    print('S/N %g-%g n %d  sigma_x %.3f sigma_y %.3f  model %.3f  r68 %.3f r95 %.3f  frac r>0.8: %.3f  >1.5: %.3f' % (lo, hi, q.sum(), sx, sy, np.sqrt(0.04**2 + (1.1 / np.median(so[q]))**2), np.percentile(rr, 68), np.percentile(rr, 95), (rr > 0.8).mean(), (rr > 1.5).mean()))
