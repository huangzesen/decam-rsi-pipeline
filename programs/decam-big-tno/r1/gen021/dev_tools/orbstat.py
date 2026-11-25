import pickle, numpy as np, json, math, sys
sys.path.insert(0, '/app/submission')
from finder import MU, tangent
D = pickle.load(open('/tmp/o_p5a/debug.pkl', 'rb'))['links']
T = np.load('/tmp/p5/truth.npz')
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
Z = np.load('/tmp/w/cache_p5.npz'); F = Z['F']
eps = math.radians(23.43928); zecl = np.array([0.0, -math.sin(eps), math.cos(eps)])
zp = F @ zecl
def elem(x):
    g = max(x[2], 1e-4)
    P = np.array([x[0], x[1], 1.0]) / g; V = np.array([x[3], x[4], x[5]]) / g
    r = np.linalg.norm(P); v2 = V @ V
    inva = 2 / r - v2 / MU
    h = np.cross(P, V); hn = np.linalg.norm(h)
    inc = math.degrees(math.acos(np.clip(h @ zp / max(hn, 1e-12), -1, 1)))
    a = 1 / inva if inva > 1e-6 else 1e6
    e = math.sqrt(max(0.0, 1 - hn * hn * inva / MU)) if inva > 1e-6 else 1.0
    vr = (P @ V) / r; vt = hn / r; vc = math.sqrt(MU / r)
    return a, e, inc, a * (1 - e), vt / vc, vr / vc, r
rows = []
for L in D:
    if L['nights'] < 5: continue
    b = body[L['det']]; bb = b[b >= 0]
    true = len(bb) >= 0.8 * len(b) and len(bb) > 0 and np.bincount(bb).max() >= 0.8 * len(b)
    none = len(bb) == 0
    rows.append((true, none, L['arc'], L['nights'], L['seasons']) + elem(L['x']))
R = np.array(rows, float)
print('links', len(R), 'true', int(R[:, 0].sum()), 'none', int(R[:, 1].sum()))
for name, m in (('true', R[:, 0] > 0), ('chance', R[:, 1] > 0)):
    for alo, ahi in ((0, 60), (60, 250), (250, 1e5)):
        s = R[m & (R[:, 2] >= alo) & (R[:, 2] < ahi)]
        if len(s) == 0: continue
        a, e, inc, q, ut, ur = s[:, 5], s[:, 6], s[:, 7], s[:, 8], s[:, 9], s[:, 10]
        pl = (q >= 24) & (a >= 28) & (inc <= 50)
        print(f"{name:7s} arc {alo}-{ahi}: n {len(s):4d}  plausible {pl.mean():.2f} | q<20 {np.mean(q < 20):.2f} q<24 {np.mean(q<24):.2f} | i>40 {np.mean(inc > 40):.2f} i>50 {np.mean(inc > 50):.2f} | a<28 {np.mean(a < 28):.2f} | e>0.5 {np.mean(e > 0.5):.2f} | ut pct {np.round(np.percentile(ut, [5, 25, 50, 75, 95]), 2)} ur pct {np.round(np.percentile(ur, [5, 50, 95]), 2)} inc pct {np.round(np.percentile(inc, [50, 90]), 0)}")
np.save('/tmp/w/orbR.npy', R)
