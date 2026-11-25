import sys, json, pickle, numpy as np
T = np.load('/tmp/p4/truth.npz'); Z = np.load('/tmp/w/cache_p4.npz'); keep = Z['keep']; seed = Z['seed']; snr_k = Z['snr_k']
E = json.load(open('/tmp/p4/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); NB = len(T['nights']); rows = T['rows']; bi = T['body']; ei = T['exp']
o = np.argsort(bi, kind='stable'); bs = bi[o]; es = ei[o]; rs = rows[o]
st = np.searchsorted(bs, np.arange(NB)); en = np.searchsorted(bs, np.arange(NB), side='right')
seed_row = np.zeros(ntot, bool); seed_row[keep[seed]] = True
snr_row = np.zeros(ntot, np.float32); snr_row[keep] = snr_k
found = set(np.load(sys.argv[1]).tolist())
tab = {}
for b in range(NB):
    r = rs[st[b]:en[b]]; nt = night[es[st[b]:en[b]]]
    nb = np.unique(nt)
    if len(nb) < 5: continue
    g = np.flatnonzero(np.diff(nb) > 120); parts = np.split(nb, g + 1)
    per = [len(p) for p in parts]; m = max(per); far2 = sorted(per, reverse=True)[1] >= 2 if len(per) > 1 else False
    k = 'a' if m >= 5 else ('b' if far2 else 'c') if m == 4 else ('d' if far2 else 'e') if m == 3 else 'f'
    def seas(thr):
        out = []
        for p in parts:
            ok = seed_row[r] & (snr_row[r] >= thr) & np.isin(nt, p)
            out.append((len(np.unique(nt[ok])), p.mean()))
        return out
    res = {}
    for thr in (6.5, 5.5, 5.0):
        s = seas(thr)
        # two seasons with >= 2 seed nights each, within 520 d of each other (centres < 420 d apart)
        e2 = any(s[i][0] >= 2 and s[j][0] >= 2 and abs(s[i][1] - s[j][1]) < 450 for i in range(len(s)) for j in range(i + 1, len(s)))
        e2any = sum(v[0] >= 2 for v in s) >= 2
        res[thr] = (e2, e2any)
    t = tab.setdefault(k, dict(n=0, lost=0, lost_p8=0, lost_p8_55=0, lost_p8_50=0, lost_2s_any65=0, lost_2s_any50=0))
    t['n'] += 1
    if b not in found:
        t['lost'] += 1; t['lost_p8'] += res[6.5][0]; t['lost_p8_55'] += res[5.5][0]; t['lost_p8_50'] += res[5.0][0]
        t['lost_2s_any65'] += res[6.5][1]; t['lost_2s_any50'] += res[5.0][1]
for k in sorted(tab): print(k, tab[k])
