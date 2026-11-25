import sys, json, numpy as np, math
sys.path.insert(0, '/app/submission')
Z = np.load('/tmp/w/cache_p4.npz')
keep, seed, de, snr = Z['keep'], Z['seed'], Z['de'], Z['snr_k']
T = np.load('/tmp/p4/truth.npz'); bod = -np.ones(int(T['ntot']), np.int64); bod[T['rows']] = T['body']
E = json.load(open('/tmp/p4/exposures.json')); et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
tb = bod[keep]
# seasons as in finder: approximate by year blocks around anti-opposition; use gaps > 120 d between nights
un = np.unique(night); sea_of = {}; s = 0
for i, n in enumerate(un):
    if i and n - un[i - 1] > 120: s += 1
    sea_of[n] = s
found = set(np.load(sys.argv[1]).tolist()) if len(sys.argv) > 1 else set()
scored = np.flatnonzero(T['nights'] >= 5)
res = {}
def units(b, lo, maxd=60.0, clean=True):
    i = np.flatnonzero((tb == b) & (snr >= lo) & (seed if clean else True))
    out = []
    for a in range(len(i)):
        for c in range(a + 1, len(i)):
            na, nc = night[de[i[a]]], night[de[i[c]]]
            dt = abs(et[de[i[c]]] - et[de[i[a]]])
            if na != nc and sea_of[na] == sea_of[nc] and 0.6 <= dt <= maxd:
                out.append((sea_of[na], min(snr[i[a]], snr[i[c]]), max(snr[i[a]], snr[i[c]])))
    return out

keys = ['b6', 'mix5', 'mix5b', 'anyb', 'anyb55', 'd5', 'b65x5', 'mix5_lo52']
cnt = {k: 0 for k in keys}; cntl = dict(cnt)
for b in scored:
    u = units(b, 5.0)
    S = lambda f: set(s_ for s_, m, M in u if f(m, M))
    sb = S(lambda m, M: m >= 6); s5 = S(lambda m, M: True); sbf = S(lambda m, M: M >= 6); sbf55 = S(lambda m, M: M >= 6 and m >= 5.5)
    sb65 = S(lambda m, M: m >= 6.5); s52 = S(lambda m, M: m >= 5.2)
    r = dict(b6=len(sb) >= 2, mix5=len(sb) >= 1 and len(s5) >= 2, mix5b=len(sb) >= 1 and len(sbf) >= 2, anyb=len(sbf) >= 2, anyb55=len(sbf55) >= 2, d5=len(s5) >= 2,
             b65x5=len(sb65) >= 1 and len(s5) >= 2, mix5_lo52=len(sb) >= 1 and len(s52) >= 2)
    for k in cnt:
        cnt[k] += r[k]
        if b not in found: cntl[k] += r[k]
print('scored', len(scored), 'reach all', cnt); print('lost', len(scored) - len(found & set(scored.tolist())), 'reach among lost', cntl)
