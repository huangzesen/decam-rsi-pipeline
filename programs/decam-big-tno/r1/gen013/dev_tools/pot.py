"""what could seed the lost bodies: python3 pot.py OUT"""
import sys, json, pickle, numpy as np, collections
out = sys.argv[1]; dst = '/tmp/p2'
T = np.load(dst + '/truth.npz'); E = json.load(open(dst + '/exposures.json'))
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
Z = np.load('/tmp/w/cache_p2.npz'); keep, seed, snr_k = Z['keep'], Z['seed'], Z['snr_k']
info = pickle.load(open('/tmp/w/info.pkl', 'rb')); found = pickle.load(open(out + '/found.pkl', 'rb'))['found']
ntot = int(T['ntot']); isseed = np.zeros(ntot, bool); isseed[keep[seed]] = True
iskept = np.zeros(ntot, bool); iskept[keep] = True
snr = np.load(dst + '/det_snr.npy', mmap_mode='r')
rows, tb, te = T['rows'], T['body'], T['exp']
sr = np.asarray(snr[rows]); sd = isseed[rows]; kp = iskept[rows]
print('planted detections kept', kp.mean().round(3), 'seed', sd.mean().round(3))
def maxin(ns, w):
    return max([int(((ns >= a) & (ns <= a + w)).sum()) for a in ns], default=0)
tab = collections.defaultdict(lambda: collections.Counter())
for b, i in info.items():
    i['cls'] = 'ge5in130' if i['m130']>=5 else ('4in64' if i['m64']>=4 else ('4in130' if i['m130']>=4 else ('3in32' if i['m32']>=3 else ('3in64' if i['m64']>=3 else ('3in130' if i['m130']>=3 else 'max2')))))
    if b in found: continue
    m = tb == b
    nb = night[te[m]]
    ns_all = np.unique(nb)
    ns_seed = np.unique(nb[sd[m]]); ns_b = np.unique(nb[sd[m] & (sr[m] >= 6.5)]); ns_55 = np.unique(nb[sd[m] & (sr[m] >= 5.5)])
    c = tab[i['cls']]
    c['lost'] += 1
    c['seed3in32'] += maxin(ns_seed, 32) >= 3
    c['seed3in64'] += maxin(ns_seed, 64) >= 3
    c['seed3in130'] += maxin(ns_seed, 130) >= 3
    c['b55_3in130'] += maxin(ns_55, 130) >= 3
    c['bright3in130'] += maxin(ns_b, 130) >= 3
    c['seed4in64'] += maxin(ns_seed, 64) >= 4
    c['seed4in130'] += maxin(ns_seed, 130) >= 4
    c['b55_4in130'] += maxin(ns_55, 130) >= 4
    c['bright4in130'] += maxin(ns_b, 130) >= 4
    # two seed pairs in different seasons (pair: 2 seed nights within 60 d)
    grp = np.split(ns_seed, np.flatnonzero(np.diff(ns_seed) > 150) + 1) if len(ns_seed) else []
    c['2seasons_with_pair'] += sum(len(g) >= 2 for g in grp) >= 2
    c['3seasons_with_pair'] += sum(len(g) >= 2 for g in grp) >= 3
for k in sorted(tab): print(k, dict(tab[k]))
