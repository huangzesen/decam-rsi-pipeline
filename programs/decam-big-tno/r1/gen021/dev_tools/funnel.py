import sys, json, pickle, glob, numpy as np
out, P = sys.argv[1], sys.argv[2]
T = np.load(P + '/truth.npz'); E = json.load(open(P + '/exposures.json'))
night = np.floor(np.array([e['mjd_mid'] for e in E]) - 0.5).astype(int)
body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']; sc = np.flatnonzero(nn >= 5)
S = {int(r[0]): r for r in np.load(P + '/stats.npy')} if glob.glob(P + '/stats.npy') else {}
def best(links):
    """per body: max number of own nights in a linkage that is >= 80% pure"""
    bn = np.zeros(len(nn), int)
    for L in links:
        d = np.asarray(L['det']); b = body[d]; bb = b[b >= 0]
        if len(bb) == 0: continue
        u, c = np.unique(bb, return_counts=True); j = u[c.argmax()]
        if c.max() >= 0.8 * len(d):
            bn[j] = max(bn[j], len(np.unique(night[rexp[d[b == j]]])))
    return bn
raw = []
for f in sorted(glob.glob(out + '/raw*.pkl')):
    raw = pickle.load(open(f, 'rb'))            # cumulative
bn_raw = best(raw)
D = pickle.load(open(out + '/debug.pkl', 'rb')); bn_g = best(D['links'])
c = json.load(open(out + '/candidates.json')); bn_c = best(c)
seedable = np.load(P + '/seedable.npy') if glob.glob(P + '/seedable.npy') else np.ones(len(nn), bool)
print('scored', len(sc), ' seedable(3in64+4th40)', int(seedable[sc].sum()))
print('raw linkage with >=4 own nights:', int((bn_raw[sc] >= 4).sum()), ' grown link >=4:', int((bn_g[sc] >= 4).sum()), ' grown >=5:', int((bn_g[sc] >= 5).sum()), ' final cand >=5:', int((bn_c[sc] >= 5).sum()))
for a, b in [(5, 5), (6, 7), (8, 10), (11, 99)]:
    m = sc[(nn[sc] >= a) & (nn[sc] <= b)]
    print(f'  nights {a}-{b}: n {len(m)} seedable {int(seedable[m].sum())} raw>=4 {int((bn_raw[m] >= 4).sum())} grown>=5 {int((bn_g[m] >= 5).sum())} found {int((bn_c[m] >= 5).sum())}')
lost = sc[(bn_raw[sc] >= 4) & (bn_g[sc] < 5)]
print('seeded (raw>=4) but not grown to 5:', len(lost), ' their total nights hist', np.bincount(np.minimum(nn[lost], 15)))
ls = sc[seedable[sc] & (bn_raw[sc] < 4)]
print('seedable but no raw linkage:', len(ls), ' nights hist', np.bincount(np.minimum(nn[ls], 15)), ' median mag', np.median(T['mr'][ls]), 'vs found', np.median(T['mr'][sc[bn_c[sc] >= 5]]))
np.save(out + '/lost_seedable.npy', ls); np.save(out + '/lost_grow.npy', lost)
