import sys, json, math, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load('/tmp/p4/truth.npz')
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load('/tmp/p4/det_exp.npy', mmap_mode='r')
info = []
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = 0; j = -1
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()]); ok = 2
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[dexp[d[b == j]]])) >= 5: ok = 1
    info.append((ok, j))
sig = lambda s: 1 / (1 + math.exp(-0.7 * s))
ks = (4, 5, 7, 10, 15, 20, 30, 40)
def ev(name, f):
    pn = []
    for c in cands:
        s0 = c['s']; s1 = s0 + (f(c) if c['src'] != 4 else 0.0)
        pn.append(c['p'] * sig(s1) / max(sig(s0), 1e-12))
    o = np.argsort(-np.array(pn)); seen = set(); lab = []
    for i in o:
        ok, j = info[i]
        if ok == 1:
            if j in seen: ok = 3
            else: seen.add(j)
        lab.append(ok)
    lab = np.array(lab); bad = np.cumsum(lab == 0); good = np.cumsum(lab == 1); bad2 = np.cumsum((lab == 0) | (lab == 2))
    print(f'{name:34s} vs none k={ks}:', [int(good[np.searchsorted(bad, k)]) for k in ks], '| vs none+mixed:', [int(good[np.searchsorted(bad2, k)]) for k in (5, 10, 20, 30, 40)])
ev('off', lambda c: 0.0)
for W in (0.5, 1.0, 1.5, 2.0, 3.0):
    ev(f'W {W} clip(1.5(1.5-rr),-1.5,.9)', lambda c, W=W: W * min(max(1.5 * (1.5 - c['rr']), -1.5), 0.9))
for W in (1.0, 2.0):
    ev(f'W {W} penalty only', lambda c, W=W: W * min(max(1.5 * (1.5 - c['rr']), -1.5), 0.0))
    ev(f'W {W} nights<=6 only', lambda c, W=W: W * min(max(1.5 * (1.5 - c['rr']), -1.5), 0.9) if c['nights'] <= 6 else 0.0)
    ev(f'W {W} centre 1.3', lambda c, W=W: W * min(max(1.5 * (1.3 - c['rr']), -1.5), 0.9))
print('--- penalty only')
for W, c0 in ((1.5, 1.5), (1.5, 1.3), (2.0, 1.3), (1.0, 1.3), (3.0, 1.5), (1.5, 1.7)):
    ev(f'W {W} centre {c0} penalty only', lambda c, W=W, c0=c0: W * min(max(1.5 * (c0 - c['rr']), -1.5), 0.0))
