"""Offline: kind-3 clipped z with / without the time-scrambled null subtraction, real vs junk.
python ana.py SDDIR TRUTH EXPOSURES"""
import sys, json, csv
import numpy as np
sys.path.insert(0, '$RSI_SANDBOX/r1/gen113/tmp/exp')
import score_inject as si

sd, tr, ex = sys.argv[1:4]
Z = np.load(sd + '/refined_vc.npz')
R = json.load(open(sd + '/refined.json'))['candidates']
truth = json.load(open(tr)); rows = list(csv.DictReader(open(ex)))
kd = Z['rkd']; scq = Z['scq']; sm = Z['scr_m']; ss = Z['scr_s']; sx = Z['scr_x']
n = len(kd)
for i, c in enumerate(R):
    c['p'] = 1.0 - i / (n + 1); c['i'] = i
_, res, _, _ = si.score(truth, R, rows)
hit = np.full(n, -1)
for k, h in enumerate(res):
    if h is not None:
        hit[R[k]['i']] = h[0]   # sorted by p = original order
cls = np.array([truth['bodies'][j]['cls'] if j >= 0 else '-' for j in hit])


def zc(x):
    ref = x[kd == 2] if (kd == 2).sum() >= 30 else x[kd >= 2]
    m = np.median(ref); s = 1.4826 * np.median(np.abs(ref - m)) + 1e-3
    return (x - m) / s


z0 = zc(scq)
print('n', n, 'kind counts', {int(k): int((kd == k).sum()) for k in np.unique(kd)})
print('scr_m by kind (median, 95%):', {int(k): (round(float(np.median(sm[kd == k])), 2), round(float(np.percentile(sm[kd == k], 95)), 2)) for k in np.unique(kd)})
for W in (0.5, 1.0, 1.5):
    z1 = zc(scq - W * sm)
    k3 = np.flatnonzero(kd == 3)
    o0 = k3[np.argsort(-z0[k3])]; o1 = k3[np.argsort(-z1[k3])]
    print(f'W={W}: kind-3 tracks {len(k3)}')
    for name, o, zz in (('base', o0, z0), ('scr', o1, z1)):
        rk = {}
        for r_, i in enumerate(o):
            if hit[i] >= 0 and hit[i] not in rk:
                rk[hit[i]] = (r_, cls[i], round(float(zz[i]), 2))
        top = o[:60]
        print('  ', name, 'body ranks among kind-3 (first 15):', sorted(rk.values())[:15])
        print('  ', name, 'junk among top 30 kind-3:', int((hit[o[:30]] < 0).sum()), 'top 60:', int((hit[top] < 0).sum()))
# real vs junk at high z0
k3 = kd == 3
for lab, msk in (('real', k3 & (hit >= 0)), ('junk z0>=3', k3 & (hit < 0) & (z0 >= 3)), ('junk z0>=2.5', k3 & (hit < 0) & (z0 >= 2.5)), ('junk all', k3 & (hit < 0))):
    if msk.sum():
        print(lab, int(msk.sum()), 'scq med %.2f scr_m med %.2f p95 %.2f, excess sig med %.2f' % (
            np.median(scq[msk]), np.median(sm[msk]), np.percentile(sm[msk], 95), np.median((scq[msk] - sm[msk]) / (ss[msk] + 0.3))))
for i in np.flatnonzero(k3 & (hit >= 0) & (cls == 'C')):
    print('C', hit[i], 'z0 %.2f scq %.2f scr_m %.2f scr_s %.2f scr_x %.2f' % (z0[i], scq[i], sm[i], ss[i], sx[i]))
hi = np.flatnonzero(k3 & (hit < 0) & (z0 >= 3.0))
print('junk z0>=3 detail (z0, scq, scr_m, scr_x):', [(round(float(z0[i]), 2), round(float(scq[i]), 2), round(float(sm[i]), 2), round(float(sx[i]), 2)) for i in hi[np.argsort(-z0[hi])][:25]])
