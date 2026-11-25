import sys, json, numpy as np
D = sys.argv[1]
T = np.load(D + '/truth.npz'); E = json.load(open(D + '/exposures.json'))
mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
snr = np.load(D + '/det_snr.npy', mmap_mode='r')
rows, bi, ei = T['rows'], T['body'], T['exp']; NB = len(T['nights'])
sn = np.asarray(snr[rows])
season = np.round((mjd - mjd.min()) / 365.25).astype(int)
m5 = np.array([e['m5'] for e in E]); band = [e['band'] for e in E]
col = dict(g=0.65, r=0.0, i=-0.26, z=-0.44, Y=-0.48, VR=0.11)
m5r = m5 - np.array([col[b] for b in band]); deep = m5r >= np.median(m5r) - 0.3
res = {}
for b in range(NB):
    m = bi == b
    nb = np.unique(night[ei[m]])
    if len(nb) < 5: continue
    g = np.flatnonzero(np.diff(nb) > 120); per = sorted([len(p) for p in np.split(nb, g + 1)], reverse=True)
    far2 = any(v >= 2 for v in per[1:])
    k = 'a' if per[0] >= 5 else ('b' if far2 else 'c') if per[0] == 4 else ('d' if far2 else 'e') if per[0] == 3 else 'f'
    out = []
    for name, sel in (('6.5 any', sn[m] >= 6.5), ('5.5 any', sn[m] >= 5.5), ('5.0 deep', (sn[m] >= 5.0) & deep[ei[m]]), ('5.5 deep', (sn[m] >= 5.5) & deep[ei[m]]), ('4.5 any', sn[m] >= 4.5)):
        e_ = ei[m][sel]; ss = season[e_]; nn = night[e_]
        cnt = {s_: len(np.unique(nn[ss == s_])) for s_ in np.unique(ss)}
        pair = any(cnt.get(s_, 0) >= 2 and cnt.get(s_ + 1, 0) >= 2 for s_ in cnt)       # 2 + 2 in adjacent seasons
        tri = any(v >= 3 for v in cnt.values())                                          # 3 seed nights in one season
        out.append((pair, tri))
    res.setdefault(k, []).append(out)
print('class  n   | 2+2 adjacent seasons: 6.5 any / 5.5 any / 5.0 deep / 5.5 deep / 4.5 any | 3 nights in a season: same sets')
for k in sorted(res):
    a = np.array(res[k])
    print(k, f'{len(a):4d}', '  |', ' / '.join(f'{a[:, j, 0].mean():.0%}' for j in range(5)), ' |', ' / '.join(f'{a[:, j, 1].mean():.0%}' for j in range(5)))
