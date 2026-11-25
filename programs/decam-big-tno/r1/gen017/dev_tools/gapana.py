import numpy as np, json, pickle, collections, sys
out = sys.argv[1]
T = np.load('/tmp/p3/truth.npz'); E = json.load(open('/tmp/p3/exposures.json'))
mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd-0.5).astype(int)
Z = np.load('/tmp/w/cache_p3.npz'); keep = Z['keep']; seed = Z['seed']; snr_k = Z['snr_k']
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']; rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
isb = np.zeros(ntot, bool); isb[keep[seed & (snr_k >= 6.5)]] = True       # bright seeds
iss = np.zeros(ntot, bool); iss[keep[seed]] = True
isk = np.zeros(ntot, bool); isk[keep] = True
raw = []
for k in (5, 6, 7):
    try: raw += [(k, c) for c in pickle.load(open(f'{out}/raw{k}.pkl', 'rb'))]
    except Exception as ex: print('no raw', k)
D = pickle.load(open(f'{out}/debug.pkl', 'rb'))['links']
cands = json.load(open(f'{out}/candidates.json'))
def best(lst, getdet):
    b = {}
    for it in lst:
        d = np.asarray(getdet(it)); bb = body[d]; bb = bb[bb >= 0]
        if len(bb) == 0: continue
        u, c = np.unique(bb, return_counts=True)
        for uu, cc in zip(u, c):
            nn = len(np.unique(night[rexp[d[body[d] == uu]]]))
            if nn > b.get(int(uu), (0, 0, 0))[0]: b[int(uu)] = (nn, len(d), it[0] if isinstance(it, tuple) else 0)
    return b
braw = best(raw, lambda it: it[1]['det']); bgr = best(D, lambda it: it['det']); bc = best(cands, lambda it: it['det'])
bi, ei, rows = T['body'], T['exp'], T['rows']; NB = len(T['nights'])
o = np.argsort(bi, kind='stable'); bs = bi[o]; es = ei[o]; rs = rows[o]
st = np.searchsorted(bs, np.arange(NB)); en = np.searchsorted(bs, np.arange(NB), side='right')
stat = collections.defaultdict(lambda: collections.Counter())
ex = collections.defaultdict(list)
for b in range(NB):
    r = rs[st[b]:en[b]]; n = night[es[st[b]:en[b]]]
    nb = np.unique(n)
    if len(nb) < 5: continue
    g = np.flatnonzero(np.diff(nb) > 120); parts = np.split(nb, g+1)
    per = tuple(sorted([len(p) for p in parts], reverse=True))
    if per[0] >= 5: continue
    # seasons with a bright pair (two nights >= 0.7 d apart with a bright seed)
    bp = []
    for p in parts:
        nbr = np.unique(n[np.isin(n, p) & isb[r]])
        bp.append(len(nbr) >= 2)
    nbp = sum(bp)
    adj = any(bp[i] and bp[i+1] and parts[i+1][0] - parts[i][-1] < 400 for i in range(len(parts)-1))
    anyg = nbp >= 2
    key = per if per in ((2,2,1),(3,2),(3,2,1),(4,2),(4,2,1),(2,2,2),(2,2,1,1)) else 'other'
    s = stat[key]; s['n'] += 1; s['bp2'] += anyg; s['bp2adj'] += adj
    rw = braw.get(b, (0,0,0)); gr = bgr.get(b, (0,0,0)); cc = bc.get(b, (0,0,0))
    s['raw4'] += rw[0] >= 4; s['grown5'] += gr[0] >= 5; s['cand5'] += cc[0] >= 5
    if anyg: s['bp2_raw4'] += rw[0] >= 4; s['bp2_grown5'] += gr[0] >= 5
    if anyg and rw[0] >= 4 and gr[0] < 5: ex[key].append((b, per, rw, gr, [len(p) for p in parts], [int(x) for x in bp]))
    if anyg and rw[0] < 4: ex['noraw'].append((b, per, [len(p) for p in parts], [int(x) for x in bp], [int(p[0]) for p in parts]))
for k, s in stat.items(): print(k, dict(s))
for k in ex: print(k, len(ex[k]), ex[k][:6])
print('---- pair lengths of bodies with bright pairs in two seasons at most 3 seasons apart')
cnt = collections.Counter()
for b in range(NB):
    r = rs[st[b]:en[b]]; n = night[es[st[b]:en[b]]]; tt = mjd[es[st[b]:en[b]]]
    nb = np.unique(n)
    if len(nb) < 5: continue
    g = np.flatnonzero(np.diff(nb) > 120); parts = np.split(nb, g+1)
    per = tuple(sorted([len(p) for p in parts], reverse=True))
    if per[0] >= 5: continue
    sp = []
    for p in parts:
        m = np.isin(n, p) & isb[r]
        nbr = np.unique(n[m])
        if len(nbr) >= 2: sp.append((p[0], float(tt[m].max() - tt[m].min())))
    best_ = None
    for i in range(len(sp)):
        for j in range(i+1, len(sp)):
            if sp[j][0] - sp[i][0] < 1250:
                lo, hi = sorted([sp[i][1], sp[j][1]])
                cl = 'both>=3d' if lo >= 3 else ('one<3d' if hi >= 3 else 'both<3d')
                if best_ is None or ['both>=3d', 'one<3d', 'both<3d'].index(cl) < ['both>=3d', 'one<3d', 'both<3d'].index(best_): best_ = cl
    if best_ is None: continue
    rw = braw.get(b, (0,0,0))
    cnt[(best_, 'raw4' if rw[0] >= 4 else 'noraw')] += 1
for k in sorted(cnt): print(k, cnt[k])
