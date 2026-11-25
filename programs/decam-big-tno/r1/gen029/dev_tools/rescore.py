import sys, json, math, numpy as np
sys.path.insert(0, '/tmp/w')
from cmp import found
cj, dst = sys.argv[1], sys.argv[2]
f, nf, T, night = found(cj, dst)
snr = np.load(dst + '/det_snr.npy', mmap_mode='r')
def newp(c, fl, sl, k=1.15):
    s = np.maximum(np.asarray(snr[np.array(c['det'])], float), 4.5)
    r_old = math.sqrt(2 * np.mean((1.1 / s) ** 2 + 0.04 ** 2)); r_new = math.sqrt(2 * np.mean((sl / s) ** 2 + fl ** 2))
    rms = c['rms']
    ds = 4.0 * (max(0.0, rms - max(0.22, 1.15 * r_old)) - max(0.0, rms - max(0.22, k * r_new))) / 0.2
    p = min(max(c['p'], 1e-9), 1 - 1e-9)
    return 1 / (1 + math.exp(-(math.log(p / (1 - p)) + 0.7 * ds))), ds
tr = list(f.values())
print('true %d, non-matching %d' % (len(tr), len(nf)))
print('penalised now (rms over the limit): true %d, non-matching %d' % (sum(newp(c, 0.04, 1.1)[1] == 0 and c['rms'] > max(0.22, 1.15 * math.sqrt(2 * np.mean((1.1 / np.maximum(np.asarray(snr[np.array(c['det'])], float), 4.5)) ** 2 + 0.04 ** 2))) for c in tr),
      sum(c['rms'] > max(0.22, 1.15 * math.sqrt(2 * np.mean((1.1 / np.maximum(np.asarray(snr[np.array(c['det'])], float), 4.5)) ** 2 + 0.04 ** 2))) for c in nf)))
for fl, sl, k in ((0.04, 1.1, 1.15), (0.09, 1.3, 1.0), (0.09, 1.3, 1.15), (0.09, 1.2, 1.15)):
    pt = np.array([newp(c, fl, sl, k)[0] for c in tr]); pf = np.array([newp(c, fl, sl, k)[0] for c in nf])
    print('model floor %.2f slope %.1f x %.2f: true at p >= 0.9/0.5/0.3/0.1: %s   non-matching: %s' % (fl, sl, k, [int((pt >= t).sum()) for t in (0.9, 0.5, 0.3, 0.1)], [int((pf >= t).sum()) for t in (0.9, 0.5, 0.3, 0.1)]))
