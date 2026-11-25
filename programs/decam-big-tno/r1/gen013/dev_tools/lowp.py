import sys, json, numpy as np
def load(cf, D):
    cands = json.load(open(cf)); T = np.load(D + '/truth.npz')
    E = json.load(open(D + '/exposures.json')); night = np.floor(np.array([e['mjd_mid'] for e in E]) - 0.5).astype(int)
    body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
    rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
    out=[]
    for c in cands:
        d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False; npl=len(bb)
        if len(bb):
            u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
            ok = cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5
        c['true']=bool(ok); c['npl']=npl; c['nd']=len(d); out.append(c)
    return out
C=load(sys.argv[1], sys.argv[2])
keys=['p','nights','seasons','rms','r','arc','msc','hv','src','fa','sn','ndh','ndm','ndo','s','neff','npl','nd']
lo=[c for c in C if c['p']<float(sys.argv[3])]
lo.sort(key=lambda c:-c['p'])
print(' T '+' '.join(f'{k:>7s}' for k in keys))
for c in lo:
    print((' T ' if c['true'] else ' . ')+' '.join(f"{c[k]:7.3g}" for k in keys))
