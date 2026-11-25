import numpy as np, json
T = np.load('/tmp/p5/truth.npz')
E = json.load(open('/tmp/p5/exposures.json'))
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
body, ex = T['body'], T['exp']
found = set(np.load('/tmp/w/found_a.npy').tolist())
snr = np.load('/tmp/p5/det_snr.npy', mmap_mode='r')[T['rows']]
from collections import Counter
cc = Counter(); cf = Counter()
for b in range(int(body.max()) + 1):
    m = body == b
    nb = np.unique(night[ex[m]])
    if len(nb) < 5: continue
    g = np.flatnonzero(np.diff(nb) > 120); parts = np.split(nb, g + 1)
    per = sorted([len(p) for p in parts], reverse=True)
    if per[0] == 4 and all(v < 2 for v in per[1:]):
        k = ('c', len(per) - 1)        # number of single far nights
    elif per[0] == 3 and any(v >= 2 for v in per[1:]):
        k = ('d', tuple(per[:3]))
    else:
        continue
    cc[k] += 1; cf[k] += b in found
for k in sorted(cc, key=str): print(k, cf[k], '/', cc[k])
