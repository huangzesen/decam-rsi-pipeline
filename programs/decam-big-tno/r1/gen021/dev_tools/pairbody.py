import numpy as np, json, sys
T = np.load('/tmp/p5/truth.npz')
E = json.load(open('/tmp/p5/exposures.json'))
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
body, ex = T['body'], T['exp']
snr = np.load('/tmp/p5/det_snr.npy', mmap_mode='r')[T['rows']]
nb = int(body.max()) + 1
res = dict(sc=0, c1=0, c2=0, c3=0, c3w=0, c22=0, c3any=0, mix=0)
info = {}
for b in range(nb):
    m = body == b
    e = ex[m]; n = night[e]
    un = np.unique(n)
    if len(un) < 5: continue
    res['sc'] += 1
    cn = np.array([q for q in un if len(np.unique(e[n == q])) >= 2 and (et[e[n == q]].max() - et[e[n == q]].min()) * 24 > 0.0])
    k = len(cn)
    res['c1'] += k >= 1; res['c2'] += k >= 2; res['c3any'] += k >= 3
    # >=3 confirmed nights within 128 d
    w3 = any(((cn >= a) & (cn <= a + 128)).sum() >= 3 for a in cn)
    res['c3w'] += w3
    # mixed: one confirmed night + 2 other nights within 64 d window
    mix = any(((un >= a - 64) & (un <= a + 64)).sum() >= 3 for a in cn)
    res['mix'] += mix
    info[b] = (len(un), k, w3, mix)
print(res)
import pickle; pickle.dump(info, open('/tmp/w/tinfo.pkl', 'wb'))
