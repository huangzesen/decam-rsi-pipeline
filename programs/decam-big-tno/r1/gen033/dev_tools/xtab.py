import sys, json, numpy as np
D = sys.argv[2]; T = np.load(D + '/truth.npz'); found = set(np.load(sys.argv[1]).tolist())
E = json.load(open(D + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
snr = np.load(D + '/det_snr.npy', mmap_mode='r'); rows = T['rows']; bi = T['body']; ei = T['exp']; nn = T['nights']
o = np.argsort(bi, kind='stable'); bs = bi[o]; es = ei[o]; rs = rows[o]
NB = len(nn); st = np.searchsorted(bs, np.arange(NB)); en = np.searchsorted(bs, np.arange(NB), side='right')
tab = {}
for b in np.flatnonzero(nn >= 5):
    ni = night[es[st[b]:en[b]]]; s = np.asarray(snr[rs[st[b]:en[b]]])
    nb = np.unique(ni); g = np.flatnonzero(np.diff(nb) > 120); per = sorted([len(p) for p in np.split(nb, g + 1)], reverse=True)
    cls = 'a' if per[0] >= 5 else ('b' if per[0] == 4 and any(v >= 2 for v in per[1:]) else 'c' if per[0] == 4 else 'd' if per[0] == 3 and any(v >= 2 for v in per[1:]) else 'e' if per[0] == 3 else 'f')
    nk = min(len(nb), 8)
    nbr = len(np.unique(ni[s >= 5.0]))        # nights with a seed-level detection
    nseas = len(per)
    for k in (('n', nk if nk < 8 else '8+', cls), ('nights', nk if nk < 8 else '8+'), ('seedn<5 nights(S/N>=5)' if nbr < 5 else 'seedn>=5',), ('seasons', min(nseas, 5))):
        t = tab.setdefault(k, [0, 0]); t[0] += 1; t[1] += int(b) in found
for k in sorted(tab, key=str): print(k, f'{tab[k][1]} / {tab[k][0]}')
