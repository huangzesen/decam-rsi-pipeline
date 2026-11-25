import sys, json, math, numpy as np
T = np.load('/tmp/p4/truth.npz')
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
def metric(p, name):
    o = np.argsort(-p); tt = t[o]; nn = (cl == 'none')[o]; bad = ~tt
    r1 = [int(np.cumsum(tt)[np.searchsorted(np.cumsum(nn), k)]) if nn.sum() >= k else int(tt.sum()) for k in (3, 5, 10, 20, 40)]
    r2 = [int(np.cumsum(tt)[np.searchsorted(np.cumsum(bad), k)]) if bad.sum() >= k else int(tt.sum()) for k in (3, 5, 10, 20, 40)]
    src = np.array([c.get('src', 0) for c in cands]) > 0
    print('%-30s vs none %s  vs all non-true %s | carried true p>=0.5/0.3/0.1: %s carried none: %s' % (name, r1, r2, [(p[src & t] >= q).sum() for q in (0.5, 0.3, 0.1)], [(p[src & (cl == 'none')] >= q).sum() for q in (0.5, 0.3, 0.1)]))
p_old = np.array([c['p'] for c in cands]); metric(p_old, 'as written')
for kexp, cap in ((1.0, 0.3), (1.0, 0.5), (1.5, 0.5), (2.0, 0.5)):
    pn = p_old.copy()
    for i, c in enumerate(cands):
        if c.get('src', 0) and c.get('fa', 0) > 0:
            hf = 1 - 0.3 * min(max((c['hv'] - 8.7) / 1.0, 0.0), 1.0)
            p0 = hf / (1 + math.exp(-0.7 * c['s']))
            ratio = min(c['p'] / max(p0, 1e-12), 0.999999)
            pr = c['fa'] * ratio / (1 - ratio)
            pe = min(max(pr * math.exp(kexp * max(c.get('vl', 0.0), 0.0)), pr), max(cap, pr))
            pn[i] = p0 * pe / (pe + c['fa'])
    metric(pn, 'prior x exp(%.1f vl), cap %.1f' % (kexp, cap))
# print carried ones
for c, k in zip(cands, cl):
    if c.get('src', 0) and (k != 'true' or c['p'] < 0.3):
        print('   ', k, 'p=%.3f nights=%d seasons=%d sn=%d fa=%.3g vl=%.2f s=%.2f rms=%.2f' % (c['p'], c['nights'], c['seasons'], c['sn'], c['fa'], c.get('vl', 0), c['s'], c['rms']))
