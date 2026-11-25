"""pur.py CAND.json DST: purity of candidates that hold a planted body (why some fail the 80 % / 5-night rule)."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load(sys.argv[2] + '/det_exp.npy', mmap_mode='r'); snr = np.load(sys.argv[2] + '/det_snr.npy', mmap_mode='r')
nn = T['nights']
out = {}; ex = []
for c in cands[:2000]:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if not len(bb): out['none'] = out.get('none', 0) + 1; continue
    u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]; pur = cnt.max() / len(d)
    nj = len(np.unique(night[dexp[d[b == j]]]))
    k = ('pure>=0.8' if pur >= 0.8 else 'pure<0.8') + (' nights>=5' if nj >= 5 else ' nights<5') + (' scored' if nn[j] >= 5 else ' unscored')
    out[k] = out.get(k, 0) + 1
    if pur < 0.8 or nj < 5:
        bad = d[b != j]
        ex.append((round(c['p'], 3), c.get('nights'), c.get('seasons'), c.get('src'), int(cnt.max()), len(d), nj, int(nn[j]), [round(float(s), 1) for s in snr[bad]][:6], len(np.unique(night[dexp[d]])) ))
for k in sorted(out): print(k, out[k])
print('(p, nights, seasons, src, body dets, ndet, body nights in cand, body nights total, snr of others, cand nights)')
for e in sorted(ex, reverse=True)[:40]: print('  ', e)
