import sys, json, math, numpy as np
scen = sys.argv[2]
E = json.load(open(scen + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
dexp = np.load(scen + '/det_exp.npy', mmap_mode='r')
cands = json.load(open(sys.argv[1]))
planted = scen != '/app/data/sample'
if planted:
    T = np.load(scen + '/truth.npz'); ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
    rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
def cls_of(c):
    if not planted: return 'un'
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb) == 0: return 'none'
    u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
    return 'true' if (cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5) else 'mixed'
cl = np.array([cls_of(c) for c in cands]); p = np.array([c['p'] for c in cands]); nights = np.array([c['nights'] for c in cands])
h3 = []
for c in cands:
    d = np.sort(np.array(c['det'])); nn = night[np.asarray(dexp[d])]; u, cnt = np.unique(nn, return_counts=True); h3.append(int((cnt >= 2).sum()))
h3 = np.array(h3)
if planted:
    t = cl == 'true'
    for nm, m in (('true', t), ('none', cl == 'none'), ('mixed', cl == 'mixed')):
        for nm2, m2 in (('all', m), ('<=6 nights', m & (nights <= 6))):
            print('%-6s %-10s n=%d  nights with two members 0/1/2/3+: %s' % (nm, nm2, m2.sum(), np.round(np.bincount(np.minimum(h3[m2], 3), minlength=4) / max(m2.sum(), 1), 2)))
    lo = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    def metric(score, name):
        o = np.argsort(-score); tt = t[o]; nn = (cl == 'none')[o]; bad = ~tt
        r1 = [int(np.cumsum(tt)[np.searchsorted(np.cumsum(nn), k)]) if nn.sum() >= k else int(tt.sum()) for k in (3, 5, 10, 20, 40)]
        r2 = [int(np.cumsum(tt)[np.searchsorted(np.cumsum(bad), k)]) if bad.sum() >= k else int(tt.sum()) for k in (3, 5, 10, 20, 40)]
        print('%-30s vs none %s  vs all non-true %s' % (name, r1, r2))
    metric(lo, 'as written')
    for w in (0.3, 0.56, 0.8): metric(lo + w * np.minimum(h3, 2), '+ %.2f per double night (<=2)' % w)
else:
    o = np.argsort(-p)
    for i in o[:26]: print('p=%.3f nights=%d seasons=%d ndet=%d double nights=%d r=%.1f' % (p[i], nights[i], cands[i]['seasons'], len(cands[i]['det']), h3[i], cands[i]['r']))
