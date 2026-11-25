import sys, json, pickle, numpy as np
T = np.load('/tmp/p4/truth.npz'); Z = np.load('/tmp/w/cache_p4.npz'); keep = Z['keep']; snr_k = Z['snr_k']
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
NB = len(T['nights']); rows = T['rows']; bi = T['body']; ei = T['exp']
o = np.argsort(bi, kind='stable'); bs = bi[o]; es = ei[o]; rs = rows[o]
st = np.searchsorted(bs, np.arange(NB)); en = np.searchsorted(bs, np.arange(NB), side='right')
snr_row = np.zeros(ntot, np.float32); snr_row[keep] = snr_k
found = set(np.load(sys.argv[1] + '/found.npy').tolist())
C = pickle.load(open(sys.argv[1] + '/carry.pkl', 'rb')); dbg = C['dbg']
print('carry dbg records', len(dbg))
def per(b):
    nb = np.unique(night[es[st[b]:en[b]]]); g = np.flatnonzero(np.diff(nb) > 120); parts = np.split(nb, g + 1)
    return sorted([len(p) for p in parts], reverse=True)
rec = {}
for r in dbg:
    d = np.asarray(r['det']); b = body[d]
    if (b >= 0).all() and len(set(b.tolist())) == 1:
        rec.setdefault(int(b[0]), []).append(r)
out = {}
for b, rr in rec.items():
    if int(T['nights'][b]) < 5: continue
    p = per(b); k = '%d+%s' % (p[0], '2+' if any(v >= 2 for v in p[1:]) else '1s')
    n4 = [r for r in rr if len(r['det']) >= 4 and len(np.unique(night[rexp[np.asarray(r['det'])]])) >= 4]
    use = n4 if n4 else rr
    r = max(use, key=lambda r: (r.get('m', 0), -r.get('fa', 9)))
    med = float(np.median(snr_row[np.asarray(r['det'])]))
    out.setdefault((k, b in found), []).append((len(np.unique(night[rexp[np.asarray(r['det'])]])), r.get('m', -1), r.get('fa', -1), r.get('mu', -1), r.get('acc', None), r.get('cnt', -1), med, r.get('ncell', -1), p))
for k in sorted(out):
    v = out[k]
    print(k, len(v))
    if not k[1]:
        ms = np.array([x[1] for x in v]); fa = np.array([x[2] for x in v]); acc = np.array([bool(x[4]) for x in v])
        print('    seed nights:', np.bincount([x[0] for x in v]), ' new nights m at best cell 0/1/2/3+:', np.bincount(np.minimum(ms.astype(int).clip(0), 3), minlength=4), ' acc', acc.sum())
        print('    fa of those with m>=2: pct 10/50/90', np.round(np.percentile(fa[ms >= 2], [10, 50, 90]), 4) if (ms >= 2).any() else '-', ' mu pct', np.round(np.percentile([x[3] for x in v], [10, 50, 90]), 4), 'ncell med', np.median([x[7] for x in v]))
        for x in v[:8]: print('       ', x[0], 'm=%g fa=%.3g mu=%.3g acc=%s cnt=%s medsnr=%.1f ncell=%.0f' % (x[1], x[2], x[3], x[4], x[5], x[6], x[7]), x[8])
