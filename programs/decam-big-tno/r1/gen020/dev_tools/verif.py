"""How many clean (seed-quality, any S/N) detections can be verified as non-static: another night's exposure of the
same band covers the position, is deep enough to show the source at S/N >= T, and shows nothing (cnt < 2 already)."""
import sys, json, math, numpy as np
sys.path.insert(0, '/app/submission')
from finder import occ_has, COV_C, AS
scen, cache = sys.argv[1], sys.argv[2]
T = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0
Z = np.load(cache)
keep, seed, seed2, th, de, flux, snr_k = Z['keep'], Z['seed'], Z['seed2'], Z['th'], Z['de'], Z['flux'], Z['snr_k']
cov0, covc = Z['cov0'], Z['covc']
E = json.load(open(f'{scen}/exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
band = np.array([e['band'] for e in E])
mz = np.array([e['magzero'] if e['magzero'] is not None else np.nan for e in E], float)
sigf = np.full(nE, np.nan)
ferr = flux / np.maximum(snr_k, 1e-3)
o = np.argsort(de, kind='stable'); ds = de[o]
st = np.searchsorted(ds, np.arange(nE)); en = np.searchsorted(ds, np.arange(nE), side='right')
for k in range(nE):
    i = o[st[k]:en[k]]; i = i[snr_k[i] < 10]
    if len(i) >= 20:
        sigf[k] = np.median(ferr[i])
m5 = mz - 2.5 * np.log10(5 * sigf)
print('m5 by band:', {b: np.round(np.nanpercentile(m5[band == b], [10, 50, 90]), 2).tolist() for b in np.unique(band)})
# pointing in tangent plane: use mean position of detections per exposure
cx = np.array([th[o[st[k]:en[k]], 0].mean() if en[k] > st[k] else np.nan for k in range(nE)])
cy = np.array([th[o[st[k]:en[k]], 1].mean() if en[k] > st[k] else np.nan for k in range(nE)])
n = len(keep)
best = np.zeros(n, np.float32)          # best expected S/N in another night's same-band exposure covering the position
nopp = np.zeros(n, np.int16)            # number of other nights with same-band coverage (any depth)
# de-biased S/N (Eddington): sources just over the threshold are mostly up-scattered
s = snr_k.astype(np.float64)
s_true = np.where(s < 9, s - 1.0 / np.maximum(s - 3.5, 0.6), s)
for A in range(nE):
    ia = o[st[A]:en[A]]
    if len(ia) == 0 or not np.isfinite(sigf[A]):
        continue
    for B in range(nE):
        if B == A or night[B] == night[A] or band[B] != band[A] or not np.isfinite(sigf[B]):
            continue
        if math.hypot(cx[A] - cx[B], cy[A] - cy[B]) > math.radians(2.3):
            continue
        cov = occ_has(covc, np.full(len(ia), B), th[ia], cov0, COV_C)
        if not cov.any():
            continue
        j = ia[cov]
        sb = s_true[j] * (sigf[A] / sigf[B]) * 10 ** (0.4 * (mz[B] - mz[A]))
        best[j] = np.maximum(best[j], sb); nopp[j] += 1
np.savez('/tmp/w/verif.npz', best=best, nopp=nopp)
cl = seed2
print('kept', n, 'clean', cl.sum(), 'seed', seed.sum())
for nm, m in (('kept', np.ones(n, bool)), ('clean', cl), ('seed', seed)):
    print(nm, 'with same-band other-night coverage: %.3f' % (nopp[m] > 0).mean(), ' verified (best >= %.1f): %.3f' % (T, (best[m] >= T).mean()),
          ' >=5: %.3f  >=6: %.3f  >=8: %.3f  >=10: %.3f' % tuple((best[m] >= t).mean() for t in (5, 6, 8, 10)))
for lo, hi in ((4.5, 5), (5, 6), (6, 8), (8, 12), (12, 25), (25, 1e9)):
    m = cl & (snr_k >= lo) & (snr_k < hi)
    print('clean S/N %g-%g: n %d, covered %.3f, verified %.3f' % (lo, hi, m.sum(), (nopp[m] > 0).mean(), (best[m] >= T).mean()))
for b in np.unique(band):
    m = cl & (band[de] == b)
    print('clean band', b, 'n', m.sum(), 'covered %.3f verified %.3f' % ((nopp[m] > 0).mean(), (best[m] >= T).mean()))
try:
    Tr = np.load(f'{scen}/truth.npz')
    bod = -np.ones(int(Tr['ntot']), np.int64); bod[Tr['rows']] = Tr['body']
    pl = bod[keep] >= 0
    print('planted kept', pl.sum(), 'clean %.3f' % cl[pl].mean(), 'covered %.3f verified %.3f' % ((nopp[pl] > 0).mean(), (best[pl] >= T).mean()))
    for lo, hi in ((4.5, 5), (5, 6), (6, 8), (8, 12), (12, 25), (25, 1e9)):
        m = pl & (snr_k >= lo) & (snr_k < hi)
        print('planted S/N %g-%g: n %d, covered %.3f, verified %.3f' % (lo, hi, m.sum(), (nopp[m] > 0).mean(), (best[m] >= T).mean()))
    np.save('/tmp/w/planted_kept.npy', pl)
except Exception as ex:
    print('no truth', ex)
