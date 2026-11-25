import sys, json, numpy as np, collections
cands = json.load(open(sys.argv[1])); T = np.load('/tmp/p4/truth.npz')
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load('/tmp/p4/det_exp.npy', mmap_mode='r'); snr = np.load('/tmp/p4/det_snr.npy', mmap_mode='r')
print(sorted(cands[0].keys()))
rows = []
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    cls = 'none'
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
        tn = len(np.unique(night[dexp[d[b == j]]]))
        cls = 'true' if (cnt.max() >= 0.8 * len(d) and tn >= 5) else 'mixed'
    s = np.maximum(np.asarray(snr[d], float), 4.5); rexp = float(np.sqrt(2 * np.mean((1.1 / s) ** 2 + 0.04 ** 2)))
    rows.append((cls, c['p'], c.get('nights'), c.get('src', 0), c['rms'] / rexp, c.get('s', 0)))
for nn in (5, 6, 7):
    for cls in ('true', 'mixed', 'none'):
        r = np.array([x[4] for x in rows if x[0] == cls and x[2] == nn and x[3] == 0])
        if len(r): print(f'src0 nights {nn} {cls:5s} n {len(r):4d} ratio 10/25/50/75/90 %:', np.round(np.percentile(r, [10, 25, 50, 75, 90]), 2))
# within the band p 0.05-0.6, src 0
band = [x for x in rows if x[3] == 0 and 0.05 <= x[1] <= 0.6]
for lim in (0.6, 0.7, 0.8, 0.9, 1.0, 9):
    print('band p 0.05-0.6, ratio <=', lim, collections.Counter(x[0] for x in band if x[4] <= lim))
