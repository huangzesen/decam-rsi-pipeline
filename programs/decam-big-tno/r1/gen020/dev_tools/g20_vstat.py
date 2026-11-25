"""vstat.py CAND.json SCEN CACHE : 'verified non-static' members of candidates: true (planted) vs not matching."""
import sys, json, math, numpy as np
sys.path.insert(0, '/app/submission')
from finder import occ_has, COV_C, COV_F, AS
cands = json.load(open(sys.argv[1])); scen = sys.argv[2]; Z = np.load(sys.argv[3])
keep, th, de, flux, snr_k = Z['keep'], Z['th'], Z['de'], Z['flux'], Z['snr_k']
cov0, covc, covf = Z['cov0'], Z['covc'], Z['covf']
E = json.load(open(f'{scen}/exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
band = np.array([e['band'] for e in E])
mz = np.array([e['magzero'] if e['magzero'] is not None else np.nan for e in E], float)
if 'mzc' in Z: mz = Z['mzc']
sigf = np.full(nE, np.nan); ferr = flux / np.maximum(snr_k, 1e-3)
o = np.argsort(de, kind='stable'); ds = de[o]
st = np.searchsorted(ds, np.arange(nE)); en = np.searchsorted(ds, np.arange(nE), side='right')
fc = np.zeros((nE, 2)); fr = np.full(nE, -1.0)
for k in range(nE):
    i = o[st[k]:en[k]]
    if len(i):
        fc[k] = 0.5 * (th[i].min(0) + th[i].max(0)); fr[k] = np.hypot(*(th[i] - fc[k]).T).max() + math.radians(0.02)
    i = i[snr_k[i] < 10]
    if len(i) >= 20: sigf[k] = np.median(ferr[i])
bi = np.unique(band, return_inverse=True)[1]
usable = np.isfinite(sigf) & np.isfinite(mz) & (sigf > 0)

def static_check(d):
    A = de[d]; p = th[d]; s = snr_k[d].astype(float)
    s_true = np.where(s < 9, s - 1.0 / np.maximum(s - 3.5, 0.6), s)
    f = flux[d] * s_true / np.maximum(s, 1e-3)
    best = np.zeros(len(d)); nopp = np.zeros(len(d), int); nblk = np.zeros(len(d), int)
    M = (bi[A][:, None] == bi[None, :]) & (night[A][:, None] != night[None, :]) & usable[None, :] & usable[A][:, None]
    M &= np.hypot(p[:, 0][:, None] - fc[None, :, 0], p[:, 1][:, None] - fc[None, :, 1]) < fr[None, :]
    mi, Bi = np.nonzero(M)
    if len(mi) == 0: return best, nopp, nblk
    cov = occ_has(covc, Bi, p[mi], cov0, COV_C)
    mi, Bi = mi[cov], Bi[cov]
    blk = occ_has(covf, Bi, p[mi], cov0, COV_F, nb=1)
    np.add.at(nblk, mi[blk], 1)
    mi, Bi = mi[~blk], Bi[~blk]
    sp = f[mi] * 10 ** (0.4 * (mz[Bi] - mz[A[mi]])) / sigf[Bi]
    np.maximum.at(best, mi, sp); np.add.at(nopp, mi, 1)
    return best, nopp, nblk

T = np.load(scen + '/truth.npz'); ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
rows_all = np.concatenate([np.array(c['det']) for c in cands]); kd = np.searchsorted(keep, rows_all)
assert (keep[kd] == rows_all).all()
best, nopp, nblk = static_check(kd)
np.save('/tmp/w/vbest.npy', best)
ptr = 0; out = []
for c in cands:
    d = np.array(c['det']); n = len(d); b = body[d]; bb = b[b >= 0]; ok = False
    vb = best[ptr:ptr + n]; no = nopp[ptr:ptr + n]; nb_ = nblk[ptr:ptr + n]; k = kd[ptr:ptr + n]; ptr += n
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        ok = cnt.max() >= 0.8 * n and len(np.unique(night[rexp[d[b == j]]])) >= 5
    cls = 'true' if ok else ('none' if len(bb) == 0 else 'mixed')
    nn = night[de[k]]
    feats = {}
    for Tt in (5, 6, 7, 8, 10):
        vn = len(np.unique(nn[vb >= Tt])); feats[Tt] = (vn, len(np.unique(nn)) - vn)
    out.append((cls, c['p'], c['nights'], feats, int((nb_ > 0).sum()), c.get('src', 0)))
for cls in ('true', 'none', 'mixed'):
    S = [x for x in out if x[0] == cls]
    if not S: continue
    print(cls, len(S))
    for Tt in (5, 6, 7, 8, 10):
        v = np.array([x[3][Tt][0] for x in S]); u = np.array([x[3][Tt][1] for x in S]); nn_ = v + u
        print('  T=%g: verified nights share %.3f; per-linkage verified nights 0/1/2/3/4+: %s ; share of nights with a blocked opp %.3f' % (
            Tt, v.sum() / nn_.sum(), np.round(np.bincount(np.minimum(v, 4), minlength=5) / len(S), 2).tolist(), np.mean([x[4] for x in S]) / nn_.mean()))
    lo = [x for x in S if x[2] <= 6]
    if lo:
        v = np.array([x[3][7][0] for x in lo]); u = np.array([x[3][7][1] for x in lo])
        print('  <=6 nights (n=%d), T=7: verified share %.3f, 0/1/2/3/4+: %s' % (len(lo), v.sum() / (v + u).sum(), np.round(np.bincount(np.minimum(v, 4), minlength=5) / len(lo), 2).tolist()))
import pickle; pickle.dump(out, open('/tmp/w/vstat.pkl', 'wb'))
