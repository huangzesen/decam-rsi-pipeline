import sys, time, json, os, numpy as np
sys.path.insert(0, '/app/submission')
import finder as F
D = '/app/data/sample/'
E = json.load(open(D + 'exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); night_e = np.floor(et - 0.5).astype(np.int64)
_, nidx = np.unique(night_e, return_inverse=True)
ra = np.asarray(np.load(D + 'det_ra.npy', mmap_mode='r')); dec = np.asarray(np.load(D + 'det_dec.npy', mmap_mode='r'))
m = np.flatnonzero((np.abs(dec - 17.6) < 1.5) & (np.abs(ra - 28.9) < 1.5))
exp = np.load(D + 'det_exp.npy', mmap_mode='r')[m].astype(np.int64)
x, y, Fr = F.tangent(ra[m], dec[m])
L = lambda k: np.load(D + f'det_{k}.npy', mmap_mode='r')[m]
snr, sharp, flag, ccd = L('snr'), L('sharp'), L('flag'), L('ccd')
AS = F.AS
cnt = F.static_count(x, y, nidx[exp].astype(np.int64), 'cpu')
a0 = F.astro_calib(exp, ccd, x, y, cnt, snr, sharp, flag, nE)
# inject: the exposure with most detections gets (+0.8", -0.5"); one CCD of the second gets (+0.5", 0)
ne = np.bincount(exp, minlength=nE); k1, k2 = np.argsort(-ne)[:2]
c2 = np.bincount(ccd[exp == k2]).argmax()
x2 = x.copy(); y2 = y.copy()
x2[exp == k1] += 0.8 * AS; y2[exp == k1] -= 0.5 * AS
sel = (exp == k2) & (ccd == c2); x2[sel] += 0.5 * AS
cnt2 = F.static_count(x2, y2, nidx[exp].astype(np.int64), 'cpu')
print('static share of exposure k1 before / after the injected offset: %.3f / %.3f' % (np.mean(cnt[exp == k1] >= 3), np.mean(cnt2[exp == k1] >= 3)))
a1 = F.astro_calib(exp, ccd, x2, y2, cnt2, snr, sharp, flag, nE)
d = (a1[0] - a0[0]) / AS, (a1[1] - a0[1]) / AS
print('k1 recovered: (%.3f, %.3f) expected (0.8, -0.5)' % (np.median(d[0][exp == k1]), np.median(d[1][exp == k1])))
print('k2/ccd recovered: (%.3f, %.3f) expected (0.5, 0.0)' % (np.median(d[0][sel]), np.median(d[1][sel])))
oth = (exp != k1) & ~sel
print('others changed by: rms %.4f" max %.4f"' % (np.sqrt(np.mean(d[0][oth] ** 2 + d[1][oth] ** 2)), np.hypot(d[0][oth], d[1][oth]).max()), ' max offset reported', a1[2])
x3 = x2 - a1[0]; y3 = y2 - a1[1]
cnt3 = F.static_count(x3, y3, nidx[exp].astype(np.int64), 'cpu')
print('static share of exposure k1 after correction + recount: %.3f' % np.mean(cnt3[exp == k1] >= 3))
