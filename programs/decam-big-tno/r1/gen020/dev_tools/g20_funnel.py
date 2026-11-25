import sys, json, pickle, numpy as np
T = np.load('/tmp/p4/truth.npz'); Z = np.load('/tmp/w/cache_p4.npz'); keep = Z['keep']; seed = Z['seed']; snr_k = Z['snr_k']
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
NB = len(T['nights'])
rows = T['rows']; bi = T['body']; ei = T['exp']
o = np.argsort(bi, kind='stable'); bs = bi[o]; es = ei[o]; rs = rows[o]
st = np.searchsorted(bs, np.arange(NB)); en = np.searchsorted(bs, np.arange(NB), side='right')
kept_row = np.zeros(ntot, bool); kept_row[keep] = True
seed_row = np.zeros(ntot, bool); seed_row[keep[seed]] = True
bseed_row = np.zeros(ntot, bool); bseed_row[keep[seed & (snr_k >= 6.5)]] = True
def pat(b):
    nb = np.unique(night[es[st[b]:en[b]]])
    if len(nb) < 5: return None
    g = np.flatnonzero(np.diff(nb) > 120); parts = np.split(nb, g + 1)
    per = sorted([len(p) for p in parts], reverse=True)
    m = per[0]; far2 = any(v >= 2 for v in per[1:])
    k = 'a' if m >= 5 else ('b' if far2 else 'c') if m == 4 else ('d' if far2 else 'e') if m == 3 else 'f'
    return k, per, parts
found = set(np.load(sys.argv[1]).tolist())
raw = pickle.load(open('/tmp/o/base/raw5.pkl', 'rb')); g3 = pickle.load(open('/tmp/o/base/g3.pkl', 'rb'))
best_raw = {}; 
for c in raw:
    d = np.asarray(c['det']); b = body[d]
    if (b >= 0).all() and len(set(b.tolist())) == 1:
        n = len(np.unique(night[rexp[d]])); best_raw[int(b[0])] = max(best_raw.get(int(b[0]), 0), n)
has3 = {}
for c in g3:
    d = keep[np.asarray(c['kdet'])]; b = body[d]
    if (b >= 0).all() and len(set(b.tolist())) == 1:
        has3[int(b[0])] = has3.get(int(b[0]), 0) + 1
tab = {}
for b in range(NB):
    q = pat(b)
    if q is None: continue
    k, per, parts = q
    r = rs[st[b]:en[b]]; nt = night[es[st[b]:en[b]]]
    # seed nights per season part
    sn = [len(np.unique(nt[seed_row[r] & np.isin(nt, p)])) for p in parts]
    bn = [len(np.unique(nt[bseed_row[r] & np.isin(nt, p)])) for p in parts]
    kn = len(np.unique(nt[kept_row[r]]))
    t = tab.setdefault(k, dict(n=0, found=0, raw4=0, raw3=0, g3=0, noseed3=0, kept5=0, lost_raw4=0, lost_g3only=0, lost_none=0, lost_none_seed3=0, per={}))
    t['n'] += 1; f = b in found; t['found'] += f
    br = best_raw.get(b, 0); t['raw4'] += br >= 4; t['raw3'] += br == 3; t['g3'] += b in has3
    t['noseed3'] += max(sn) < 3; t['kept5'] += kn >= 5
    if not f:
        if br >= 4: t['lost_raw4'] += 1
        elif b in has3 or br == 3: t['lost_g3only'] += 1
        else:
            t['lost_none'] += 1; t['lost_none_seed3'] += max(sn) >= 3
        key = '+'.join(str(v) for v in sorted(per, reverse=True)[:4])
        if k in ('b', 'd'): t['per'][key] = t['per'].get(key, 0) + 1
for k in sorted(tab):
    t = tab[k]
    print(k, {q: v for q, v in t.items() if q != 'per'})
    if t['per']: print('    lost by nights per season:', sorted(t['per'].items(), key=lambda x: -x[1])[:12])
