"""funnel.py DST DBGDUMP CAND.json : per night-pattern class, how far the planted bodies get in the sparse pass."""
import sys, json, pickle, numpy as np
dst, dump, cand = sys.argv[1:4]
T = np.load(dst + '/truth.npz'); E = json.load(open(dst + '/exposures.json'))
mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
snr = np.load(dst + '/det_snr.npy', mmap_mode='r')
NB = len(T['nights'])
rows = T['rows']; bi = T['body']; ei = T['exp']
o = np.argsort(bi, kind='stable'); bs = bi[o]; es = ei[o]; rs = rows[o]
st = np.searchsorted(bs, np.arange(NB)); en = np.searchsorted(bs, np.arange(NB), side='right')
D = pickle.load(open(dump, 'rb')); tri = set(pickle.load(open(dump + '.tri', 'rb'))) if len(sys.argv) > 3 else set()
found = set()
for c in json.load(open(cand))[:2000]:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5: found.add(int(j))
tab = {}
for b in range(NB):
    nb = np.unique(night[es[st[b]:en[b]]])
    if len(nb) < 5: continue
    g = np.flatnonzero(np.diff(nb) > 120); parts = np.split(nb, g + 1)
    per = sorted([len(p) for p in parts], reverse=True); m = per[0]
    nu = sum(1 for v in per if v >= 2)
    sm = float(np.median(snr[rs[st[b]:en[b]]]))
    k = ('a' if m >= 5 else 'b4' if m == 4 else 'd3' if m == 3 else 'f2') + f' useasons{min(nu,3)}' + (' bright' if sm >= 7 else ' faint')
    t = tab.setdefault(k, [0, 0, 0, 0, 0, 0]); t[0] += 1
    t[1] += b in D['ideal']; t[2] += b in D['gate']; t[3] += b in D['fit']; t[4] += b in tri; t[5] += b in found
print('class: n, units in 2 seasons, pair through gates, quad fit, pure triple, found')
for k in sorted(tab): print(f'{k:28s}', tab[k])
