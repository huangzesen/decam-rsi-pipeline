import sys, json, csv
import numpy as np
sys.path.insert(0, '$RSI_SANDBOX/r1/gen113/tmp/exp')
import score_inject as si
sd, tr, ex = sys.argv[1:4]
Z = np.load(sd + '/refined_vc.npz'); R = json.load(open(sd + '/refined.json'))['candidates']
truth = json.load(open(tr)); rows = list(csv.DictReader(open(ex)))
kd = Z['rkd']; scq = Z['scq']; sm = Z['scr_m']; ss = Z['scr_s']; sx = Z['scr_x']; n = len(kd)
for i, c in enumerate(R): c['p'] = 1.0 - i / (n + 1); c['i'] = i
_, res, _, _ = si.score(truth, R, rows)
hit = np.full(n, -1)
for k, h in enumerate(res):
    if h is not None: hit[R[k]['i']] = h[0]
cls = np.array([truth['bodies'][j]['cls'] if j >= 0 else '-' for j in hit])
def zc(x):
    ref = x[kd >= 2]; m = np.median(ref); s = 1.4826 * np.median(np.abs(ref - m)) + 1e-3; return (x - m) / s
k3 = np.flatnonzero(kd == 3)
keys = {'base': zc(scq), 'sub0.25': zc(scq - 0.25 * sm), 'sub0.5': zc(scq - .5 * sm), 'sig': zc((scq - sm) / (ss + 0.3)),
        'subx0.5': zc(scq - .5 * sx), 'pos_only0.5': zc(scq - .5 * np.maximum(sm, 0))}
print('scr_m real (unique bodies, best track):')
for nm, z in keys.items():
    o = k3[np.argsort(-z[k3])]
    seen = set(); fr = []; junk = 0; uniq = 0
    for r_, i in enumerate(o[:40]):
        if hit[i] < 0: junk += 1
        elif hit[i] not in seen: seen.add(hit[i]); uniq += 1
    c25 = [r_ for r_, i in enumerate(o) if hit[i] == 25][:1]
    print(f'{nm:12s} top40 kind-3: junk {junk} unique bodies {uniq}; C25 rank {c25}')
