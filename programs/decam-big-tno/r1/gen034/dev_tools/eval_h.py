"""eval_h.py CAND.json DST [mult]: weighted found share of the H-scored planted bodies, and a slot simulation at real-sky density."""
import sys, json, math, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
MULT = int(sys.argv[3]) if len(sys.argv) > 3 else 1
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']; mr = T['mr']; r = T['r']; NB = len(nn)
hvt = mr + 0.4 - 5 * np.log10(r * (r - 0.8))
scored = (nn >= 5) & (hvt <= 8.4); w = np.where(hvt <= 7.4, 2.0, 1.0) * scored
cb = []; cf = []
for c in cands[:2000]:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; j = -1; ok = False
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()])
        ok = cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5
    cb.append(j); cf.append(ok)
cb = np.array(cb); cf = np.array(cf)
def pold(c):
    if 'hfac' not in c: return c['p']
    return c['p'] / max(c['hfac'], 1e-9) * (1 - 0.3 * min(max(c['hv'] - 8.7, 0.0), 1.0))
P = {'as written': np.array([c['p'] for c in cands[:2000]]), 'old factor': np.array([pold(c) for c in cands[:2000]])}
fb = set(cb[cf].tolist())
print(f"bodies >=5 nights {int((nn >= 5).sum())}, scored (hv<=8.4) {int(scored.sum())} (weight {w.sum():.0f}); found any {len(fb)}, found scored {sum(scored[j] for j in fb)}, weight found {sum(w[j] for j in fb):.0f} = {sum(w[j] for j in fb) / w.sum():.3f}")
for lo, hi in ((29, 33), (33, 38), (38, 43), (43, 48), (48, 65)):
    m = scored & (r >= lo) & (r < hi); print(f"   r {lo}-{hi}: scored {int(m.sum())}, found {sum(1 for j in fb if m[j])}")
nm = cb < 0
for k, p in P.items():
    print(f"[{k}] candidates {len(p)}: no planted detection {int(nm.sum())} (p>=0.5/0.2/0.1/0.05: {[int((p[nm] >= t).sum()) for t in (0.5, 0.2, 0.1, 0.05)]}), found-scored p>=0.5/0.2/0.1/0.05: {[int(sum(1 for i in np.flatnonzero(cf) if scored[cb[i]] and p[i] >= t)) for t in (0.5, 0.2, 0.1, 0.05)]}")
rng = np.random.default_rng(1)
idx_sc = np.flatnonzero(scored)
for NS in (12, 22, 40):
    q = NS / len(idx_sc)
    out = {k: [] for k in P}
    for it in range(3000):
        sub = rng.random(NB) < q
        ns = int((sub & scored).sum())
        if ns == 0: continue
        keep = np.flatnonzero(nm | ((cb >= 0) & sub[np.maximum(cb, 0)]))
        slots = 2 * ns; wt = w[sub].sum()
        for k, p in P.items():
            pk = np.concatenate([p[keep]] + [p[np.flatnonzero(nm)]] * (MULT - 1))
            ik = np.concatenate([keep] + [np.flatnonzero(nm)] * (MULT - 1))
            o = np.argsort(-pk, kind='stable')[:slots]
            got = set(int(cb[ik[i]]) for i in o if cf[ik[i]] and cb[ik[i]] >= 0)
            out[k].append(sum(w[j] for j in got) / wt)
    print(f"slot simulation, ~{NS} scored bodies per patch, chance candidates x{MULT}: " + ', '.join(f"{k}: {np.mean(v):.4f}" for k, v in out.items()))
