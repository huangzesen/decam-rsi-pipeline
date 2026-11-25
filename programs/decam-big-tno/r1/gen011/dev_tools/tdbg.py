import pickle, numpy as np, json, collections, sys
out = sys.argv[1]
T = np.load('/tmp/p2/truth.npz'); E = json.load(open('/tmp/p2/exposures.json'))
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
Z = np.load('/tmp/w/cache_p2.npz'); keep = Z['keep']
C = pickle.load(open(out + '/carry.pkl', 'rb')); dbg = C['dbg']; tr = C['truth']
Zc = 2.5
per = collections.defaultdict(list)
for t in tr:
    if t['pool'] or t['n3']: continue
    for i in range(len(t['seed'])):
        per[int(t['seed'][i])].append((int(t['det'][i]), t['ok'][i], t['z'][i], t['hw'][i], t['rp'][i], t['K2'], t['nb']))
tab = collections.Counter(); ex = []
for s, lst in per.items():
    d = dbg[s]; bseed = body[d['det']]
    rows = keep[[l[0] for l in lst]]
    if len(set(bseed.tolist())) != 1 or bseed[0] < 0 or (body[rows] != bseed[0]).any(): tab['misaligned'] += 1; continue
    nts = night[rexp[rows]]
    un = np.unique(nts)
    if len(un) < 2: continue
    okn = {n: np.any([l[1] for l, nn in zip(lst, nts) if nn == n], axis=0) for n in un}
    nok = sum(v.any() for v in okn.values())
    # joint over two nights
    joint = False; K2 = lst[0][5]; nb = lst[0][6]
    for a in range(len(lst)):
        for b in range(a + 1, len(lst)):
            if nts[a] == nts[b]: continue
            both = lst[a][1] & lst[b][1]
            if both.any():
                dz = np.abs(lst[a][2] - lst[b][2])[both]; lim = (lst[a][3] + lst[b][3])[both] + 2 * Zc / nb
                if (dz <= lim).any(): joint = True
    m = int(d.get('m', 0))
    tab[(f'farnights_in_box={min(len(un),3)}', f'nights_ok={min(nok,3)}', 'joint' if joint else 'nojoint', f'm={min(m,3)}')] += 1
    if nok < len(un) and len(ex) < 12:
        # nearest approach of the never-ok night
        for n in un:
            if not okn[n].any():
                l = [l for l, nn in zip(lst, nts) if nn == n][0]
                ex.append((s, 'rpmin', round(float(l[4].min()), 2), 'z at rpmin', round(float(l[2][l[4].argmin()]), 2), 'K2', K2, 'nb', nb, 'm1', round(d['m1']), 'm2', round(d['m2']), 'smaj', round(d['smaj'], 3), 'smin', round(d['smin'], 3)))
for k in sorted(tab, key=str): print(k, tab[k])
for e in ex: print(e)
