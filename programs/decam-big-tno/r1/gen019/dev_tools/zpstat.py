import numpy as np, json, time
t0 = time.time()
D = '/app/data/sample'
E = json.load(open(D + '/exposures.json')); nE = len(E)
band = np.array([e['band'] for e in E]); mz = np.array([e['magzero'] if e['magzero'] is not None else np.nan for e in E])
snr = np.load(D + '/det_snr.npy', mmap_mode='r'); 
sel = np.flatnonzero(np.asarray(snr) >= 30)
exp = np.load(D + '/det_exp.npy', mmap_mode='r')[sel].astype(np.int64)
ra = np.load(D + '/det_ra.npy', mmap_mode='r')[sel]; dec = np.load(D + '/det_dec.npy', mmap_mode='r')[sel]
flux = np.load(D + '/det_flux.npy', mmap_mode='r')[sel].astype(np.float64); flag = np.load(D + '/det_flag.npy', mmap_mode='r')[sel]
ok = (flag & 1) == 0
print(len(sel), 'bright', ok.sum(), time.time() - t0)
bi = np.unique(band, return_inverse=True)[1]
cell = 1.5 / 3600
kx = np.floor(ra * np.cos(np.radians(17.6)) / cell).astype(np.int64); ky = np.floor(dec / cell).astype(np.int64)
key = (kx * 200000 + ky) * 8 + bi[exp]
mag = mz[exp] - 2.5 * np.log10(np.maximum(flux, 1e-9))
good = ok & np.isfinite(mag)
key, mag, ex = key[good], mag[good], exp[good]
o = np.argsort(key, kind='stable'); key, mag, ex = key[o], mag[o], ex[o]
u, st, cnt = np.unique(key, return_index=True, return_counts=True)
gid = np.repeat(np.arange(len(u)), cnt)
# group median via sort within group
o2 = np.lexsort((mag, gid)); ms = mag[o2]
med = ms[st + cnt // 2]
res = mag - med[gid]
use = cnt[gid] >= 3
print('stars with >=3 dets in band:', int((cnt >= 3).sum()), 'dets used', int(use.sum()))
de = np.full(nE, np.nan); ne = np.zeros(nE, int); sc = np.full(nE, np.nan)
for e in range(nE):
    m = use & (ex == e)
    ne[e] = m.sum()
    if ne[e] >= 30:
        de[e] = np.median(res[m]); sc[e] = 1.48 * np.median(np.abs(res[m] - de[e]))
f = np.isfinite(de)
print('exposures with offset:', f.sum(), 'of', nE)
print('offset percentiles 1/5/25/50/75/95/99:', np.round(np.percentile(de[f], [1, 5, 25, 50, 75, 95, 99]), 3))
print('|offset| > 0.1:', int((np.abs(de[f]) > 0.1).sum()), ' > 0.2:', int((np.abs(de[f]) > 0.2).sum()), ' > 0.4:', int((np.abs(de[f]) > 0.4).sum()))
print('scatter within exposure (median):', np.nanmedian(sc))
for e in np.argsort(-np.abs(np.nan_to_num(de)))[:15]:
    print(e, E[e]['band'], E[e]['exptime'], round(E[e]['mjd_mid'], 2), 'offset %.3f' % de[e], 'n', ne[e], 'magzero', E[e]['magzero'], 'm5', E[e]['m5'])
np.save('/tmp/t/zp_off.npy', de)
print(time.time() - t0)
