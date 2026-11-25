import sys, json, numpy as np, torch
sys.path.insert(0, '/app/submission')
import finder as Fd
scen = sys.argv[1]
torch.set_num_threads(3)
L = lambda k: np.load(f'{scen}/det_{k}.npy', mmap_mode='r')
E = json.load(open(f'{scen}/exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(np.int64)
nu, nidx = np.unique(night, return_inverse=True)
exp = np.asarray(L('exp')).astype(np.int64)
x, y, F = Fd.tangent(np.asarray(L('ra')), np.asarray(L('dec')))
cnt = Fd.static_count(x, y, nidx[exp].astype(np.int64), 'cpu')
cnt = np.asarray(cnt)
snr, sharp, flag = np.asarray(L('snr')), np.asarray(L('sharp')), np.asarray(L('flag'))
np.save('/tmp/w/cnt.npy', cnt.astype(np.int16))
print('cnt hist', np.bincount(np.minimum(cnt, 6)) / len(cnt))
edges = [-5, -0.5, 0.0, 0.2, 0.35, 0.5, 0.8, 1.2, 1.8, 2.5, 3.0, 4.0, 6.0, 1e9]
print('sharp bins', edges)
for lo, hi in ((4.5, 5), (5, 6), (6, 8), (8, 12), (12, 25), (25, 1e9)):
    ms = (snr >= lo) & (snr < hi) & ((flag & 1) == 0)
    for nm, m in (('static(cnt>=3)', ms & (cnt >= 3)), ('cnt==2', ms & (cnt == 2)), ('single(cnt<2)', ms & (cnt < 2))):
        h = np.histogram(sharp[m], edges)[0]
        print('S/N %g-%g %-15s n %8d ' % (lo, hi, nm, m.sum()), np.round(h / max(m.sum(), 1), 3))
