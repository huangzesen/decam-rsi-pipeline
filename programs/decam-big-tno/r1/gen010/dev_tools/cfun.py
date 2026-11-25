"""cfun.py OUT DST CACHE: where the bodies of the carry classes (best season 3 or 4 nights) are lost."""
import sys, json, pickle, numpy as np
out, dst, cache = sys.argv[1:4]
T = np.load(dst + '/truth.npz'); E = json.load(open(dst + '/exposures.json'))
mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load(dst + '/det_exp.npy', mmap_mode='r'); snr = np.load(dst + '/det_snr.npy', mmap_mode='r')
Z = np.load(cache); keep = Z['keep']; seed = Z['seed']
kept = np.zeros(ntot, bool); kept[keep] = True; isseed = np.zeros(ntot, bool); isseed[keep[seed]] = True
nn = T['nights']; NB = len(nn)
rows = T['rows']; bi = T['body']; ei = T['exp']
o = np.argsort(bi, kind='stable'); bs = bi[o]; es = ei[o]; rs = rows[o]
st = np.searchsorted(bs, np.arange(NB)); en = np.searchsorted(bs, np.arange(NB), side='right')
cands = json.load(open(out + '/candidates.json'))
found = {}
for c in cands[:2000]:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[dexp[d[b == j]]])) >= 5: found.setdefault(int(j), c)
def owner(d):
    b = body[np.asarray(d)]
    return int(b[0]) if (b >= 0).all() and (b == b[0]).all() else -1
g3 = pickle.load(open(out + '/g3.pkl', 'rb'))
has3 = {}
for c in g3:
    j = owner(keep[np.asarray(c['kdet'])])
    if j >= 0: has3[j] = has3.get(j, 0) + 1
C = pickle.load(open(out + '/carry.pkl', 'rb'))
voted = {}
for d in (C['dbg'] or []):
    j = owner(d['det'])
    if j >= 0:
        v = voted.setdefault(j, dict(n=0, acc=0, mmax=0, famin=9e9, nseed=0))
        v['n'] += 1; v['acc'] += bool(d.get('acc')); v['mmax'] = max(v['mmax'], d.get('m', 0)); v['famin'] = min(v['famin'], d.get('fa', 9e9)); v['nseed'] = max(v['nseed'], len(np.unique(night[dexp[d['det']]])))
newb = {}
for c in C['new']:
    d = np.asarray(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()])
        newb.setdefault(j, []).append((cnt.max(), len(d), c.get('fa'), c.get('sn'), c.get('src')))
try:
    grown = pickle.load(open(out + '/grown.pkl', 'rb')); gb = {}
    for v in grown.values():
        if v is None: continue
        d = keep[np.asarray(v['det'])]; j = owner(d)
        if j >= 0: gb[j] = max(gb.get(j, 0), len(np.unique(night[dexp[d]])))
except Exception as ex:
    print('grown.pkl:', ex); gb = {}
tab = {}; ex = {}
for b in range(NB):
    r = rs[st[b]:en[b]]; nb = night[es[st[b]:en[b]]]
    un = np.unique(nb)
    if len(un) < 5: continue
    g = np.flatnonzero(np.diff(un) > 120); parts = np.split(un, g + 1); per = sorted([len(p) for p in parts], reverse=True)
    m = per[0]; far2 = any(v >= 2 for v in per[1:])
    if m >= 5: continue
    k = ('4' if m == 4 else '3' if m == 3 else '<=2') + ('+far2' if far2 else '+singles')
    # seed-quality nights in best season
    sn_per = sorted([len(np.unique(nb[isseed[r] & np.isin(nb, p)])) for p in parts], reverse=True)
    kn = len(np.unique(nb[kept[r]]))
    t = tab.setdefault(k, dict(n=0, found=0, kept5=0, seed3=0, g3=0, grown34=0, voted=0, m2=0, acc=0, new=0))
    t['n'] += 1; t['found'] += b in found; t['kept5'] += kn >= 5; t['seed3'] += sn_per[0] >= 3; t['g3'] += b in has3
    t['grown34'] += gb.get(b, 0) in (3, 4); t['voted'] += b in voted; t['m2'] += (b in voted and voted[b]['mmax'] >= 2); t['acc'] += (b in voted and voted[b]['acc'] > 0); t['new'] += b in newb
    if b not in found and b in voted and voted[b]['mmax'] >= 2: ex.setdefault(k, []).append((b, per, float(np.median(snr[r])), voted[b], newb.get(b)))
for k in sorted(tab): print(k, tab[k])
for k in sorted(ex):
    print('lost though voted with m>=2:', k)
    for e in ex[k][:12]: print('   ', e)
