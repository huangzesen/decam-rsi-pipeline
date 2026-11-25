import sys, json, numpy as np
dst = '/tmp/p4'
Z = np.load('/tmp/cache_1.npz'); keep = Z['keep']; seed = Z['seed']; snr_k = Z['snr_k']
T = np.load(dst + '/truth.npz'); E = json.load(open(dst + '/exposures.json'))
mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot'])
iskept = np.zeros(ntot, bool); iskept[keep] = True
isseed = np.zeros(ntot, bool); isseed[keep[seed]] = True
snr = np.load(dst + '/det_snr.npy', mmap_mode='r')
NB = len(T['nights']); rows = T['rows']; bi = T['body']; ei = T['exp']
o = np.argsort(bi, kind='stable'); bs = bi[o]; es = ei[o]; rs = rows[o]
st = np.searchsorted(bs, np.arange(NB)); en = np.searchsorted(bs, np.arange(NB), side='right')
rules = {'seed>=6 dt60': (6, 6, 60, 1), 'seed>=6 dt30': (6, 6, 30, 1), 'seed>=6 dt100': (6, 6, 100, 1), 'seed 6+5 dt60': (6, 5, 60, 1), 'seed>=5 dt60': (5, 5, 60, 1),
         'seed>=5 dt100': (5, 5, 100, 1), 'kept>=5 dt100': (5, 5, 100, 0), 'seed>=5.5 dt100': (5.5, 5.5, 100, 1), 'seed 6+5 dt100': (6, 5, 100, 1)}
tab = {}
for b in range(NB):
    e = es[st[b]:en[b]]; r = rs[st[b]:en[b]]
    nb = np.unique(night[e])
    if len(nb) < 5: continue
    g = np.flatnonzero(np.diff(nb) > 120); parts = np.split(nb, g + 1)
    per = sorted([len(p) for p in parts], reverse=True); m = per[0]
    nu = sum(1 for v in per if v >= 2)
    if nu < 2 or m >= 5: continue
    sm = float(np.median(snr[r]))
    k = ('b4' if m == 4 else 'd3' if m == 3 else 'f2') + (' bright' if sm >= 7 else ' faint')
    t = tab.setdefault(k, [0] + [0] * (2 * len(rules))); t[0] += 1
    sea = np.searchsorted(np.r_[nb[g + 1]], night[e], side='right')
    s = np.asarray(snr[r]); ng = night[e]; tt = mjd[e]
    for ri, (hi, lo, dtm, needseed) in enumerate(rules.values()):
        okd = (isseed[r] if needseed else iskept[r])
        nus = 0; ntri = 0; units = []
        for q in np.unique(sea):
            idx = np.flatnonzero((sea == q) & okd & (s >= lo))
            has = False
            for a_ in idx:
                for b_ in idx:
                    if ng[b_] > ng[a_] and tt[b_] - tt[a_] <= dtm and max(s[a_], s[b_]) >= hi and min(s[a_], s[b_]) >= lo:
                        has = True; units.append((q, a_, b_))
            nus += has
        # triple-capable: units in >=2 seasons and >= 5 distinct nights among units of some pair of quads sharing a unit
        tri = False
        if nus >= 2:
            for u in units:
                oth = [v for v in units if v[0] != u[0]]
                for i1 in range(len(oth)):
                    for i2 in range(i1 + 1, len(oth)):
                        nts = set(ng[[u[1], u[2], oth[i1][1], oth[i1][2], oth[i2][1], oth[i2][2]]])
                        if len(nts) >= 5: tri = True
        t[1 + 2 * ri] += nus >= 2; t[2 + 2 * ri] += tri
print('class: n | per rule: bodies with units in 2 seasons / triple-capable (5 nights from two quads sharing a unit)')
print('rules:', list(rules))
for k in sorted(tab): print(f'{k:12s}', tab[k][0], ' '.join(f'{tab[k][1 + 2 * i]}/{tab[k][2 + 2 * i]}' for i in range(len(rules))))
