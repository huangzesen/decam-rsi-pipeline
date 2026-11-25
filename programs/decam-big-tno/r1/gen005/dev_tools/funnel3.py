import sys, json, pickle, numpy as np, collections
out = sys.argv[1]; dst = '/tmp/p2'
T = np.load(dst + '/truth.npz'); E = json.load(open(dst + '/exposures.json'))
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
info = pickle.load(open('/tmp/w/info.pkl', 'rb'))
cands = json.load(open(out + '/candidates.json'))
found = {}
for c in cands[:2000]:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5:
            found.setdefault(int(j), c)
def own(det):
    b = body[det]; bb = b[b >= 0]
    if len(bb) == 0: return -1, 0
    u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
    return int(j), len(np.unique(night[rexp[det[b == j]]]))
import glob, os
rawbest = collections.defaultdict(int)
fs = sorted(glob.glob(out + '/raw*.pkl'))
if fs:
    for c in pickle.load(open(fs[-1], 'rb')):
        j, n = own(np.asarray(c['det']))
        if j >= 0 and n >= 0.8 * c['nights']: rawbest[j] = max(rawbest[j], n)
grownbest = collections.defaultdict(int)
if os.path.exists(out + '/grown.pkl'):
    keep = np.load('/tmp/w/cache_p2.npz')['keep']
    for v in pickle.load(open(out + '/grown.pkl', 'rb')).values():
        if v is None: continue
        j, n = own(keep[np.asarray(v['det'])])
        if j >= 0: grownbest[j] = max(grownbest[j], n)
for b, i in info.items():
    i['cls'] = 'ge5in130' if i['m130']>=5 else ('4in64' if i['m64']>=4 else ('4in130' if i['m130']>=4 else ('3in32' if i['m32']>=3 else ('3in64' if i['m64']>=3 else ('3in130' if i['m130']>=3 else 'max2')))))
tab = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
for b, i in info.items():
    nb = '5' if i['n'] == 5 else ('6-7' if i['n'] <= 7 else '8+')
    t = tab[(i['cls'], nb)]
    t[0] += 1; t[1] += b in found; t[2] += rawbest[b] >= 4; t[3] += grownbest[b] >= 4; t[4] += grownbest[b] >= 5
print('class, nights: total, found, raw>=4 own nights, grown>=4, grown>=5')
for k in sorted(tab): print(k, tab[k])
print('found', len(found), 'of', len(info))
pickle.dump(dict(found=set(found), rawbest=dict(rawbest), grownbest=dict(grownbest)), open(out + '/funnel3.pkl', 'wb'))
