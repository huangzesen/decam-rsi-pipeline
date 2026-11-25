import sys, time, json, numpy as np
sys.path.insert(0, '/app/submission')
import finder as F
t = time.time()
D = '/app/data/sample/'
E = json.load(open(D + 'exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); night_e = np.floor(et - 0.5).astype(np.int64)
_, nidx = np.unique(night_e, return_inverse=True)
exp = np.asarray(np.load(D + 'det_exp.npy', mmap_mode='r')).astype(np.int64)
x, y, Fr = F.tangent(np.asarray(np.load(D + 'det_ra.npy', mmap_mode='r')), np.asarray(np.load(D + 'det_dec.npy', mmap_mode='r')))
print('loaded', time.time() - t)
cnt = F.static_count(x, y, nidx[exp].astype(np.int64), 'cpu')
print('cnt', time.time() - t)
snr = np.asarray(np.load(D + 'det_snr.npy', mmap_mode='r')); sharp = np.asarray(np.load(D + 'det_sharp.npy', mmap_mode='r')); flag = np.asarray(np.load(D + 'det_flag.npy', mmap_mode='r'))
ccd = np.load(D + 'det_ccd.npy', mmap_mode='r')
t1 = time.time()
ac = F.astro_calib(exp, ccd, x, y, cnt, snr, sharp, flag, nE)
print('astro_calib', time.time() - t1, 's')
dx, dy = ac[0], ac[1]
AS = F.AS
print('per-detection correction: rms %.3f" max %.3f" share nonzero %.3f' % (np.sqrt(np.mean(dx**2 + dy**2)) / AS, np.hypot(dx, dy).max() / AS, np.mean((dx != 0) | (dy != 0))))
# independent check: stars with a different selection (snr 12-25), 1.5" cells, residual per year before / after
for lab, xx, yy in (('before', x, y), ('after', x - dx, y - dy)):
    sel = np.flatnonzero((cnt >= 5) & (flag == 0) & (snr >= 12) & (snr < 25) & (sharp > 0.7) & (sharp < 1.3))
    xs, ys, es = xx[sel], yy[sel], exp[sel]
    kx = np.floor((xs - xs.min()) / (3 * AS)).astype(np.int64); ky = np.floor((ys - ys.min()) / (3 * AS)).astype(np.int64)
    _, gid = np.unique(kx * (int(ky.max()) + 2) + ky, return_inverse=True)
    n = np.bincount(gid); mx = np.bincount(gid, weights=xs) / n; my = np.bincount(gid, weights=ys) / n
    g = n[gid] >= 5; f = (n / np.maximum(n - 1, 1))[gid]
    rx = ((xs - mx[gid]) * f)[g] / AS; ry = ((ys - my[gid]) * f)[g] / AS; eg = es[g]
    tight = np.hypot(rx, ry) < 1.0
    yr = ((et[eg] - 56658) / 365.25 + 2014).astype(int)
    print(lab, 'n', int(tight.sum()), 'scatter/axis %.3f"' % (1.4826 * np.median(np.abs(np.r_[rx[tight], ry[tight]]))))
    print('   year medians:', ' '.join('%d:(%+.3f,%+.3f)' % (Y, np.median(rx[tight & (yr == Y)]), np.median(ry[tight & (yr == Y)])) for Y in np.unique(yr)))
    med = np.array([[np.median(rx[tight & (eg == k)]), np.median(ry[tight & (eg == k)])] for k in range(nE) if (tight & (eg == k)).sum() >= 50])
    print('   per-exposure median offset: median %.3f" p90 %.3f" max %.3f" (%d exposures)' % (np.median(np.hypot(*med.T)), np.percentile(np.hypot(*med.T), 90), np.hypot(*med.T).max(), len(med)))
