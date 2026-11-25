"""cls.py CAND.json DST : found share of the scored planted bodies by night pattern (largest season) and brightness."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']; mr = T['mr']; NB = len(nn)
snr = np.load(sys.argv[2] + '/det_snr.npy', mmap_mode='r')
found = {}; nf = []
for c in cands[:2000]:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5:
            ok = True; found.setdefault(int(j), c['p'])
    if not ok: nf.append((c['p'], c.get('nights'), c.get('seasons'), c.get('src'), int((b >= 0).sum()), len(d)))
# pattern per body
rows = T['rows']; bi = T['body']; ei = T['exp']
o = np.argsort(bi, kind='stable'); bs = bi[o]; es = ei[o]; rs = rows[o]
st = np.searchsorted(bs, np.arange(NB)); en = np.searchsorted(bs, np.arange(NB), side='right')
cls = {}; 
def key(b):
    nb = np.unique(night[es[st[b]:en[b]]])
    if len(nb) < 5: return None
    # seasons: split at gaps > 120 d
    g = np.flatnonzero(np.diff(nb) > 120); parts = np.split(nb, g + 1)
    per = sorted([len(p) for p in parts], reverse=True)
    m = per[0]
    far2 = any(v >= 2 for v in per[1:])
    if m >= 5: k = 'a: >=5 in a season'
    elif m == 4: k = 'b: 4 + far (some far season >=2)' if far2 else 'c: 4 + single far nights'
    elif m == 3: k = 'd: 3 + far (some far season >=2)' if far2 else 'e: 3 + single far nights'
    else: k = 'f: <=2 per season'
    sm = float(np.median(snr[rs[st[b]:en[b]]]))
    return k, ('faint<6' if sm < 6 else 'mid6-9' if sm < 9 else 'bright>=9')
tab = {}
for b in range(NB):
    k = key(b)
    if k is None: continue
    for kk in (k[0], k[0] + ' | ' + k[1], 'all | ' + k[1], 'all'):
        t = tab.setdefault(kk, [0, 0, 0]); t[0] += 1; t[1] += b in found; t[2] += (b in found and found[b] >= 0.3)
for kk in sorted(tab): print(f"{kk:55s} {tab[kk][1]:4d} / {tab[kk][0]:4d}  ({tab[kk][1] / tab[kk][0]:.0%})   p>=0.3: {tab[kk][2]}")
pf = np.array([f[0] for f in nf]); pt = np.array(list(found.values()))
print('not matching:', len(nf), ' p>=0.9/0.5/0.3/0.1: true', [(pt >= t).sum() for t in (0.9, 0.5, 0.3, 0.1)], 'non-matching', [(pf >= t).sum() for t in (0.9, 0.5, 0.3, 0.1)])
print('top non-matching (p, nights, seasons, src, planted dets, ndet):', sorted(nf, reverse=True)[:12])
np.save(sys.argv[3] if len(sys.argv) > 3 else '/tmp/w/found_last.npy', np.array(sorted(found)))
