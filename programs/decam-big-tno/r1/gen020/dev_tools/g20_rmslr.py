import sys, json, math, numpy as np
from scipy.stats import chi2
T = np.load('/tmp/p4/truth.npz'); Z = np.load('/tmp/w/cache_p4.npz'); keep = Z['keep']; snr_k = Z['snr_k']
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
cands = json.load(open(sys.argv[1]))
def cls_of(c):
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb) == 0: return 'none'
    u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
    return 'true' if (cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5) else 'mixed'
cl = np.array([cls_of(c) for c in cands]); t = cl == 'true'
p = np.array([c['p'] for c in cands]); lo = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
nights = np.array([c['nights'] for c in cands])
def llr(c, floor, R):
    k = np.searchsorted(keep, np.array(c['det'])); s = np.maximum(snr_k[k], 4.5); n = len(k)
    sa2 = float(np.mean((1.1 / s) ** 2 + floor ** 2))            # per-axis variance (mean over members)
    dof = max(2 * n - 6, 2); SS = max(n * c['rms'] ** 2, 1e-6)
    lr = chi2.logpdf(SS / sa2, dof) - math.log(sa2)
    lc = math.log(dof / 2) + (dof / 2 - 1) * math.log(SS) - (dof / 2) * math.log(n * R * R)
    return lr - lc, dof, math.sqrt(SS / (sa2 * dof))
def metric(score, name):
    o = np.argsort(-score); tt = t[o]; nn = (cl == 'none')[o]; bad = ~tt
    r1 = [int(np.cumsum(tt)[np.searchsorted(np.cumsum(nn), k)]) if nn.sum() >= k else int(tt.sum()) for k in (3, 5, 10, 20, 40)]
    r2 = [int(np.cumsum(tt)[np.searchsorted(np.cumsum(bad), k)]) if bad.sum() >= k else int(tt.sum()) for k in (3, 5, 10, 20, 40)]
    m = nights <= 6; st, sf = score[m & t], score[m & (cl == 'none')]
    print('%-38s vs none %s  vs all non-true %s  AUC<=6n (vs none) %.4f' % (name, r1, r2, (st[:, None] > sf[None, :]).mean()))
metric(lo, 'as written')
for floor, R in ((0.06, 0.5), (0.09, 0.5)):
    L = np.array([llr(c, floor, R) for c in cands])
    for nm, m in (('true', t), ('none', cl == 'none'), ('mixed', cl == 'mixed')):
        print('   floor %.2f %s: logLR pct 10/25/50/75/90 %s | reduced chi pct %s' % (floor, nm, np.round(np.percentile(L[m, 0], [10, 25, 50, 75, 90]), 2), np.round(np.percentile(L[m, 2], [10, 25, 50, 75, 90]), 2)))
        m6 = m & (nights <= 6)
        print('        <=6 nights: logLR pct %s' % np.round(np.percentile(L[m6, 0], [10, 25, 50, 75, 90]), 2))
    for w in (0.25, 0.5, 0.75):
        for cap in (3.0, 5.0):
            metric(lo + w * np.clip(L[:, 0], -cap, cap), 'floor %.2f w=%.2f cap=%.0f' % (floor, w, cap))
