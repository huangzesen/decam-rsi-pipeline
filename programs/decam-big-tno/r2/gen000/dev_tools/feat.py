"""feat.py CAND.json [DST] : features of the candidates, split into planted-true / non-matching when DST (truth) is given.
Prints, per class, how the candidates fall over the non-detection counts (strong and marginal band), the magnitude
scatter and the fitted redness; and lists the non-matching ones."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1]))
true = np.zeros(len(cands), bool)
if len(sys.argv) > 2:
    T = np.load(sys.argv[2] + '/truth.npz')
    E = json.load(open(sys.argv[2] + '/exposures.json')); night = np.floor(np.array([e['mjd_mid'] for e in E]) - 0.5).astype(int)
    body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
    rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
    for k, c in enumerate(cands):
        d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
        if len(bb):
            u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
            true[k] = cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5
            c['npl'] = int(len(bb))
        else:
            c['npl'] = 0
def tab(sel, name):
    cs = [c for c, t in zip(cands, sel) if t]
    print(f"== {name}: {len(cs)}")
    if not cs: return
    for key in ('ndh', 'ndm', 'ndh2', 'ndm2'):
        v = np.array([c.get(key, 0) for c in cs]); print(f"   {key}: " + ' '.join(f"{q}:{int((v == q).sum())}" for q in range(0, 6)) + f" >5:{int((v > 5).sum())}")
    for key in ('msc', 'rms', 'rho', 'p'):
        v = np.array([c.get(key, 0) for c in cs]); print(f"   {key}: 10/50/90% {np.percentile(v, 10):.3f} {np.percentile(v, 50):.3f} {np.percentile(v, 90):.3f}")
nn = np.array([c.get('nights', 0) for c in cands])
tab(true, 'true'); tab(~true, 'non-matching')
tab(true & (nn <= 6), 'true, <= 6 nights'); tab(~true & (nn <= 6), 'non-matching, <= 6 nights')
if '-l' in sys.argv:
    for c, t in zip(cands, true):
        if not t and c['p'] >= 0.03:
            print({k: c[k] for k in ('p', 'nights', 'seasons', 'rms', 'r', 'msc', 'hv', 'src', 'ndh', 'ndm', 'ndo', 'ndh2', 'ndm2', 'ndo2', 'rho', 's', 'neff') if k in c}, 'npl', c.get('npl'), 'ndet', len(c['det']))
