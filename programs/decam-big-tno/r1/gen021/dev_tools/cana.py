"""cana.py OUT : what the carry did with the 4-night arcs of planted bodies (first carry call of the run)."""
import sys, pickle, numpy as np, json
out = sys.argv[1]
C = pickle.load(open(out + '/carry.pkl', 'rb')); dbg = C['dbg']
Z = np.load('/tmp/w/cache_p4.npz'); keep = Z['keep']; seed2 = Z['seed2']
T = np.load('/tmp/p4/truth.npz'); ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
dexp = np.load('/tmp/p4/det_exp.npy', mmap_mode='r'); snr = np.load('/tmp/p4/det_snr.npy', mmap_mode='r')
found = set(np.load(out + '/found.npy').tolist())
clean_rows = np.zeros(ntot, bool); clean_rows[keep[seed2]] = True
kept_rows = np.zeros(ntot, bool); kept_rows[keep] = True
rows = T['rows']; bi = T['body']
o = np.argsort(bi, kind='stable'); bs = bi[o]; rs = rows[o]
st = np.searchsorted(bs, np.arange(bs.max() + 1)); en = np.searchsorted(bs, np.arange(bs.max() + 1), side='right')
n4 = int(sys.argv[2]) if len(sys.argv) > 2 else 1823
tab = {}
ex = []
for k, d in enumerate(dbg[:n4]):
    det = np.asarray(d['det']); b = body[det]
    if not ((b >= 0).all() and len(np.unique(b)) == 1): continue
    j = int(b[0]); allr = rs[st[j]:en[j]]
    an = np.unique(night[dexp[det]])
    nts = night[dexp[allr]]
    far = ~np.isin(nts, an)
    tmid = mjd[dexp[det]].mean()
    near = np.abs(mjd[dexp[allr]] - tmid) < 1100 + 60
    farn = np.unique(nts[far & near]); farc = np.unique(nts[far & near & clean_rows[allr]]); fark = np.unique(nts[far & near & kept_rows[allr]])
    tot = len(np.unique(nts))
    if tot < 5: continue
    key = (min(len(farc), 3), )
    t = tab.setdefault(key, [0, 0, 0, 0, 0]); t[0] += 1; t[1] += d.get('m', 0) >= 2; t[2] += bool(d.get('acc')); t[3] += j in found
    ex.append((len(farn), len(fark), len(farc), d.get('m'), d.get('fa'), d.get('acc'), j in found, d['mu'], d.get('ncell'), d['K2'], d['nb'], float(np.median(snr[det]))))
print('true 4-night arcs of scoreable bodies, by far clean nights within reach (0,1,2,3+): [arcs, vote m>=2, passed chance test, body found at the end]')
for k in sorted(tab): print('  ', k, tab[k])
ex = np.array([[float(v) if v is not None else np.nan for v in e] for e in ex])
print('far nights (all/kept/clean), m, fa, acc, found, mu, ncell, K2, nb, medsnr  -- arcs with >=2 clean far nights that did not pass:')
q = (ex[:, 2] >= 2) & (ex[:, 5] == 0)
np.set_printoptions(linewidth=200, suppress=True, precision=4)
print(ex[q][:40])
print('fa of those with m>=2 and not acc: ', np.sort(ex[q & (ex[:, 3] >= 2)][:, 4]))
print('cross-tab: rows far nights in catalogue within reach (0,1,2,3+), cols clean far nights (0,1,2,3+)')
M = np.zeros((4, 4), int); K = np.zeros((4, 4), int)
for e in ex:
    M[int(min(e[0], 3)), int(min(e[2], 3))] += 1; K[int(min(e[1], 3)), int(min(e[2], 3))] += 1
print(M); print('rows kept far nights, cols clean far nights'); print(K)
