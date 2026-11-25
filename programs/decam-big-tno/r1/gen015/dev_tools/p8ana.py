"""p8ana.py OUT DST CACHE: the raw linkages of the season-pair pass against the planted truth, and its potential by class."""
import sys, json, pickle, numpy as np
out, dst, cache = sys.argv[1:4]
T = np.load(dst + '/truth.npz'); E = json.load(open(dst + '/exposures.json'))
mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load(dst + '/det_exp.npy', mmap_mode='r'); snr = np.load(dst + '/det_snr.npy', mmap_mode='r')
Z = np.load(cache); keep = Z['keep']; seed = Z['seed']; snk = Z['snr_k']
sb = np.zeros(ntot, bool); sb[keep[seed & (snk >= float(sys.argv[4]) if len(sys.argv) > 4 else seed & (snk >= 6.5))]] = True
raw = [c for c in pickle.load(open(out + '/raw5.pkl', 'rb')) if c.get('h', 0) >= 200]
print('P8 raw linkages', len(raw))
pure = {}; nmix = 0; nnone = 0
for c in raw:
    d = np.asarray(c['det']); b = body[d]
    if (b >= 0).all() and (b == b[0]).all(): pure[int(b[0])] = max(pure.get(int(b[0]), 0), len(np.unique(night[dexp[d]])))
    elif (b >= 0).any(): nmix += 1
    else: nnone += 1
print('bodies with a pure P8 linkage', len(pure), ' mixed linkages', nmix, ' linkages with no planted detection', nnone)
found = set(np.load(sys.argv[5]).tolist()) if len(sys.argv) > 5 else set()
nn = T['nights']; NB = len(nn); rows = T['rows']; bi = T['body']; ei = T['exp']
o = np.argsort(bi, kind='stable'); bs = bi[o]; es = ei[o]; rs = rows[o]
st = np.searchsorted(bs, np.arange(NB)); en = np.searchsorted(bs, np.arange(NB), side='right')
tab = {}
for b in range(NB):
    r = rs[st[b]:en[b]]; nb = night[es[st[b]:en[b]]]; un = np.unique(nb)
    if len(un) < 5: continue
    g = np.flatnonzero(np.diff(un) > 120); parts = np.split(un, g + 1); per = [len(p) for p in parts]
    m = max(per); far2 = sorted(per, reverse=True)[1] >= 2 if len(per) > 1 else False
    k = ('>=5' if m >= 5 else '4' if m == 4 else '3' if m == 3 else '<=2') + ('+far2' if far2 else '+singles')
    # potential: two consecutive seasons (gap of first nights < 520 d) each with a bright-seed pair >= 3 d apart
    bn = [np.unique(nb[sb[r] & np.isin(nb, p)]) for p in parts]
    haspair = [len(q) >= 2 and q[-1] - q[0] >= 3 for q in bn]
    pot = any(haspair[i] and haspair[i + 1] and parts[i + 1][-1] - parts[i][0] <= 520 for i in range(len(parts) - 1))
    t = tab.setdefault(k, dict(n=0, potential=0, pure=0, pure_of_pot=0, found=0))
    t['n'] += 1; t['potential'] += pot; t['pure'] += b in pure; t['pure_of_pot'] += (pot and b in pure); t['found'] += b in found
for k in sorted(tab): print(k, tab[k])
