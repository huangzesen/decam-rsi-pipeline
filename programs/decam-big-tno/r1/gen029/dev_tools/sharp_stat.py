import sys, json, numpy as np
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
snr, sharp, flag = L('snr'), L('sharp'), L('flag')
cnt = F.static_count(x, y, nidx[exp].astype(np.int64), 'cpu')
st = (cnt >= 5) & (flag == 0)
for lo, hi in ((20, 100), (8, 12), (5, 6)):
    k = st & (snr >= lo) & (snr < hi)
    med = np.array([np.median(sharp[k & (exp == e)]) for e in np.unique(exp) if (k & (exp == e)).sum() >= 100])
    sc = np.array([1.4826 * np.median(np.abs(sharp[k & (exp == e)] - np.median(sharp[k & (exp == e)]))) for e in np.unique(exp) if (k & (exp == e)).sum() >= 100])
    print(f'S/N {lo}-{hi}: per-exposure median sharp of static sources: pct 0/5/50/95/100 = {np.percentile(med, [0, 5, 50, 95, 100]).round(3)}; scatter med {np.median(sc):.3f}; n exp {len(med)}')
    sg = 3.4 / np.maximum(snr[k], 4.5)
    keepw = (sharp[k] >= np.minimum(0.35, 1 - 2.5 * sg)) & (sharp[k] <= np.maximum(2.5, 1 + 2.5 * sg)) & (sharp[k] != 0)
    seedw = (sharp[k] >= np.minimum(0.5, 1 - 2.0 * sg)) & (sharp[k] <= np.maximum(1.8, 1 + 2.0 * sg)) & (sharp[k] != 0)
    print(f'     share of static detections passing the keep window {keepw.mean():.3f}, the seed window {seedw.mean():.3f}')
