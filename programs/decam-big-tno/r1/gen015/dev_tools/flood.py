import sys, json, math, numpy as np
sys.path.insert(0, '/app/submission')
import finder as F
scen = sys.argv[1]
L = lambda k: np.load(f'{scen}/det_{k}.npy', mmap_mode='r')
E = json.load(open(f'{scen}/exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(np.int64)
nu, nidx = np.unique(night, return_inverse=True)
exp = np.asarray(L('exp')).astype(np.int64)
x, y, FF = F.tangent(np.asarray(L('ra')), np.asarray(L('dec')))
cnt = F.static_count(x, y, nidx[exp].astype(np.int64), 'cpu')
snr, sharp, flag = np.asarray(L('snr')), np.asarray(L('sharp')), np.asarray(L('flag'))
keep = (cnt < 3) & ((flag & 1) == 0) & (sharp >= 0.35) & (sharp <= 2.5) & (snr >= 4.5)
np.savez('/tmp/w/flood_tmp.npz', cnt=cnt, keep=keep)
per = np.bincount(exp[keep], minlength=nE); med = np.median(per[per > 0])
flooded = (per > 4 * med) & (per > 3000)
print('median kept', med, 'flooded', int(flooded.sum()))
good = (flag & 1) == 0
br = snr >= 20
nb_all = np.bincount(exp[br & good], minlength=nE); nb_keep = np.bincount(exp[br & keep], minlength=nE)
fb = nb_keep / np.maximum(nb_all, 1)
print('bright non-static share, all exposures pct', np.percentile(fb[nb_all > 50], [5, 25, 50, 75, 90, 95, 99]))
cnt1 = np.bincount(exp[keep & (cnt < 2)], minlength=nE)
for k in np.flatnonzero(flooded):
    i = np.flatnonzero((exp == k) & keep); s = np.sort(snr[i])[::-1]
    tgt = int(4 * med)
    print(k, E[k]['band'], E[k]['exptime'], E[k]['m5'], 'kept', per[k], 'cnt<2', cnt1[k], 'S/N at 4xmed', round(float(s[min(tgt, len(s) - 1)]), 1), 'at 1.5xmed', round(float(s[min(int(1.5 * med), len(s) - 1)]), 1),
          'kept S/N pct', np.round(np.percentile(s, [50, 90, 99]), 1), 'bright share', round(float(fb[k]), 3), 'n bright all', nb_all[k], 'kept', nb_keep[k])
