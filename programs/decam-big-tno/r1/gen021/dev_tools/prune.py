"""prune.py CAND.json DST : effect of dropping members in single-night seasons when >= 5 nights remain."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
dexp = np.load(sys.argv[2] + '/det_exp.npy', mmap_mode='r')
snr = np.load(sys.argv[2] + '/det_snr.npy', mmap_mode='r')
def match(d):
    b = body[d]; bb = b[b >= 0]
    if len(bb) == 0: return -1, 0
    u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
    ok = cnt.max() >= 0.8 * len(d) and len(np.unique(night[dexp[d[b == j]]])) >= 5
    return (int(j) if ok else -1), int(cnt.max())
def prune(d, mode):
    nt = night[dexp[d]]
    un = np.unique(nt)
    # seasons: split nights at gaps > 120 d
    g = np.flatnonzero(np.diff(un) > 120); parts = np.split(un, g + 1)
    weak = np.concatenate([p for p in parts if len(p) < mode] + [np.zeros(0, int)])
    strong = len(un) - len(weak)
    if len(weak) and strong >= 5:
        return d[~np.isin(nt, weak)]
    return d
for mode in (0, 2, 3):
    found = {}; lost = 0; gained = 0; nm = 0
    for c in cands[:2000]:
        d = np.array(c['det']); j0, _ = match(d)
        d2 = prune(d, mode) if mode else d
        j, _ = match(d2)
        if j >= 0: found.setdefault(j, c['p'])
        if j0 >= 0 and j < 0: lost += 1
        if j0 < 0 and j >= 0: gained += 1
    print('mode', mode, 'found', len(found), 'gained cands', gained, 'lost cands', lost)
# anatomy of wrong members: share of wrong detections by season size
tot = {}
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb) < 4: continue
    u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
    nt = night[dexp[d]]; un = np.unique(nt)
    g = np.flatnonzero(np.diff(un) > 120); parts = np.split(un, g + 1)
    size = {int(n): len(p) for p in parts for n in p}
    for k in range(len(d)):
        s = min(size[int(nt[k])], 4)
        t = tot.setdefault(s, [0, 0]); t[0] += 1; t[1] += b[k] != j
print('wrong members by nights in their season (1,2,3,4+): ', {k: (v[1], v[0], round(v[1] / v[0], 3)) for k, v in sorted(tot.items())})
