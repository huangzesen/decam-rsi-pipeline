"""found by cadence class x brightness for a planted run: python3 ana.py OUT"""
import sys, json, pickle, numpy as np, collections
out = sys.argv[1]; dst = '/tmp/p2'
T = np.load(dst + '/truth.npz'); E = json.load(open(dst + '/exposures.json'))
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
snr = np.load(dst + '/det_snr.npy', mmap_mode='r')
body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
info = pickle.load(open('/tmp/w/info.pkl', 'rb'))
cands = json.load(open(out + '/candidates.json'))
found = {}; nfalse = 0; pf = []
for c in cands[:2000]:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5:
            ok = True; found.setdefault(int(j), c)
    if not ok:
        nfalse += 1; pf.append(c['p'])
rows = T['rows']; tb = T['body']
srow = np.asarray(snr[rows])
def cls(i):
    return 'ge5in130' if i['m130']>=5 else ('4in64' if i['m64']>=4 else ('4in130' if i['m130']>=4 else ('3in32' if i['m32']>=3 else ('3in64' if i['m64']>=3 else ('3in130' if i['m130']>=3 else 'max2')))))
tab = collections.defaultdict(lambda: [0, 0]); tb2 = collections.defaultdict(lambda: [0, 0]); tb3 = collections.defaultdict(lambda: [0, 0])
for b, i in info.items():
    m = tb == b
    # median over nights of the best S/N of the night
    nb = night[T['exp'][m]]; sb = srow[m]
    best = np.array([sb[nb == n].max() for n in np.unique(nb)])
    med = np.median(best)
    br = 'faint(<6)' if med < 6 else ('mid(6-9)' if med < 9 else 'bright')
    i['br'] = br; i['cls'] = cls(i)
    tab[(i['cls'], br)][0] += 1; tab[(i['cls'], br)][1] += b in found
    tb2[i['cls']][0] += 1; tb2[i['cls']][1] += b in found
    tb3[br][0] += 1; tb3[br][1] += b in found
print('found', len(found), 'of', len(info), f'({len(found) / len(info):.1%})', 'not matching', nfalse, 'with p>=0.5:', sum(p >= 0.5 for p in pf), 'p>=0.2:', sum(p >= 0.2 for p in pf))
pt = sorted([c['p'] for c in found.values()], reverse=True); pfs = sorted(pf, reverse=True)
for k in (1, 3, 5, 10, 20, 40):
    if len(pfs) >= k: print(f'  true above the {k}th non-matching: {sum(p > pfs[k - 1] for p in pt)}', end=';')
print()
print('by class:', {k: tuple(v) for k, v in sorted(tb2.items())})
print('by brightness:', {k: tuple(v) for k, v in sorted(tb3.items())})
for k in sorted(tab): print('  ', k, tab[k])
src = collections.Counter(c.get('src', 0) for c in found.values()); print('found by src', dict(src))
pickle.dump(dict(found=set(found)), open(out + '/found.pkl', 'wb'))
