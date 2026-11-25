import sys, json, numpy as np
def found(cj, dst):
    cands = json.load(open(cj)); T = np.load(dst + '/truth.npz')
    E = json.load(open(dst + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
    ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
    rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
    f = {}; nf = []
    for c in cands[:2000]:
        d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False
        if len(bb):
            u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
            if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5:
                ok = True; f.setdefault(int(j), c)
        if not ok: nf.append(c)
    return f, nf, T, night
if __name__ == '__main__':
    fa, na, T, night = found(sys.argv[1], sys.argv[3]); fb, nb, _, _ = found(sys.argv[2], sys.argv[3])
    print('A', len(fa), 'B', len(fb), 'only A', len(set(fa) - set(fb)), 'only B', len(set(fb) - set(fa)))
    snr = np.load(sys.argv[3] + '/det_snr.npy', mmap_mode='r')
    def desc(b):
        rows = T['rows'][T['body'] == b]; ex = T['exp'][T['body'] == b]
        nb_ = np.unique(night[ex]); g = np.flatnonzero(np.diff(nb_) > 120); per = [len(p) for p in np.split(nb_, g + 1)]
        return per, round(float(np.median(snr[rows])), 1)
    for lab, s, f in (('only A', set(fa) - set(fb), fa), ('only B', set(fb) - set(fa), fb)):
        for b in sorted(s):
            print(' ', lab, b, desc(b), 'p=%.2f' % f[b]['p'], 'src', f[b].get('src'), 'nights', f[b].get('nights'))
