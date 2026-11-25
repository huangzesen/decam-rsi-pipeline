import sys, json, math, numpy as np
sys.path.insert(0, '/app/submission')
import finder as F
scen = '/app/data/sample'
L = lambda k: np.load(f'{scen}/det_{k}.npy', mmap_mode='r')
E = json.load(open(f'{scen}/exposures.json')); nE = len(E)
exp = np.asarray(L('exp')).astype(np.int64)
Z = np.load('/tmp/w/flood_tmp.npz'); cnt = Z['cnt']; keep = Z['keep']
snr = np.asarray(L('snr')); flag = np.asarray(L('flag')); ccd = np.asarray(L('ccd')).astype(np.int64)
m5 = np.array([e.get('m5') if e.get('m5') is not None else np.nan for e in E], float); m5m = np.nanmedian(m5)
sb = 20.0 * 10 ** (0.4 * np.clip(np.nan_to_num(m5 - m5m, nan=0.0), 0.0, 3.5))
good = (flag & 1) == 0
br = (snr >= sb[exp]) & good
key = exp * 128 + ccd
nall = np.bincount(key[br], minlength=nE * 128); nk = np.bincount(key[br & keep], minlength=nE * 128)
kk = np.bincount(key[keep], minlength=nE * 128)
ok = nall >= 15
sh = nk / np.maximum(nall, 1)
print('ccd-exposures with >=15 bright:', int(ok.sum()), 'of', int((kk > 0).sum()), 'with kept')
print('share pct', np.round(np.percentile(sh[ok], [10, 25, 50, 75, 90, 95, 99]), 3))
tot = kk.sum()
for lim in (0.1, 0.2, 0.3, 0.5, 0.8):
    bad = ok & (sh > lim)
    print(f'limit {lim}: bad ccd-exposures {int(bad.sum())} ({bad.sum() / ok.sum():.1%}), kept detections in them {int(kk[bad].sum())} ({kk[bad].sum() / tot:.1%})')
print('kept in ccd-exposures with <15 bright:', int(kk[~ok].sum()), f'({kk[~ok].sum() / tot:.1%})')
# per exposure: kept before/after at limit 0.3
bad = ok & (sh > 0.3)
per0 = np.bincount(exp[keep], minlength=nE); per1 = np.bincount(exp[keep & ~bad[key]], minlength=nE)
print('per-exposure kept median before/after', np.median(per0[per0 > 0]), np.median(per1[per1 > 0]))
for k in (216, 243, 244, 245, 246, 313, 316, 317, 322, 189, 170, 13):
    print(k, E[k]['band'], 'kept', per0[k], 'after', per1[k], 'bad ccds', int(bad[k * 128:(k + 1) * 128].sum()), 'of', int(ok[k * 128:(k + 1) * 128].sum()))
# seeds (cnt<2) share in bad
seedish = keep & (cnt < 2) & (snr >= 5)
print('seed-like in bad cells:', int((seedish & bad[key]).sum()), 'of', int(seedish.sum()))
