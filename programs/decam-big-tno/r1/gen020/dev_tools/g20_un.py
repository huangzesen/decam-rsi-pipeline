import sys, json, math, numpy as np
cands = json.load(open(sys.argv[1])); cands.sort(key=lambda c: -c['p'])
Z = np.load('/tmp/w/cache_un.npz'); keep = Z['keep']; snr_k = Z['snr_k']
p = np.array([c['p'] for c in cands])
print(len(cands), 'candidates; p >= 0.9/0.5/0.3/0.2/0.1/0.05:', [(p >= q).sum() for q in (0.9, 0.5, 0.3, 0.2, 0.1, 0.05)])
keys = ['p', 'nights', 'seasons', 'rms', 'msc', 'r', 'hv', 'arc', 'src', 'fa', 'ndh', 'ndm', 'ndh2', 'ndm2', 'vl', 'rl', 's']
for c in cands[:int(sys.argv[2]) if len(sys.argv) > 2 else 30]:
    k = np.searchsorted(keep, np.array(c['det'])); s = np.maximum(snr_k[k], 4.5); n = len(k)
    out = []
    for floor in (0.06, 0.09):
        sa2 = float(np.mean((1.1 / s) ** 2 + floor ** 2)); dof = max(2 * n - 6, 2); SS = n * c['rms'] ** 2
        out.append(math.sqrt(SS / (sa2 * dof)))
    print(' '.join('%s=%s' % (q, (('%.3g' % c[q]) if isinstance(c.get(q), float) else c.get(q))) for q in keys), 'ndet=%d medsnr=%.1f redchi(0.06/0.09)=%.2f/%.2f' % (n, np.median(s), out[0], out[1]))
