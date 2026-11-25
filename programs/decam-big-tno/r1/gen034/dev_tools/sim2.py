"""sim2.py PLANTED_CAND.json DST UNPLANTED_CAND.json : slot simulation with the true candidates of a planted run and the chance
candidates of an unplanted run (its p < 0.9 ones), for variants of the H factor (recomputed offline from hv, arc, seasons)."""
import sys, json, math, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz'); un = json.load(open(sys.argv[3]))
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
nn = T['nights']; mr = T['mr']; r = T['r']; NB = len(nn)
hvt = mr + 0.4 - 5 * np.log10(r * (r - 0.8))
def base_p(c):          # p without any H factor
    if 'hfac' in c: return c['p'] / max(c['hfac'], 1e-9)
    return c['p'] / (1 - 0.3 * min(max(c['hv'] - 8.7, 0.0), 1.0))
def var(c, kind, H8=8.55, SIG=0.35, W2=0.0, H7=7.5):
    p = base_p(c)
    if kind == 'none': return p
    if kind == 'old': return p * (1 - 0.3 * min(max(c['hv'] - 8.7, 0.0), 1.0))
    sg = SIG * (1.5 if (c['arc'] < 40 and c['seasons'] <= 1) else 1.0) * math.sqrt(2)
    z8 = 0.5 * math.erfc((c['hv'] - H8) / sg); z7 = 0.5 * math.erfc((c['hv'] - H7) / sg)
    p = min(p * max(z8, 0.02), 0.999999)
    if W2 > 0:
        o = p / (1 - p) * (1 + W2 * z7); p = o / (1 + o)
    return p
un = [c for c in un if base_p(c) < 0.9]          # the sure real bodies of the unplanted run are not chance candidates
V = {'old (gen033)': lambda c: var(c, 'old'), 'H8 8.55 s.35': lambda c: var(c, 'new'), 'H8 8.55 s.35 W2 1': lambda c: var(c, 'new', W2=1.0),
     'H8 8.4 s.3': lambda c: var(c, 'new', H8=8.4, SIG=0.3), 'H8 8.8 s.4': lambda c: var(c, 'new', H8=8.8, SIG=0.4), 'H8 8.55 s.2': lambda c: var(c, 'new', SIG=0.2)}
for OFF in (8.4, 8.0, 8.8):          # hv that corresponds to catalogue H = 8 (unknown: 8.4 is my estimate)
    scored = (nn >= 5) & (hvt <= OFF); w = np.where(hvt <= OFF - 1.0, 2.0, 1.0) * scored
    cb = []; cf = []
    for c in cands[:2000]:
        d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; j = -1; ok = False
        if len(bb):
            u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()])
            ok = cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5
        cb.append(j); cf.append(ok)
    cb = np.array(cb); cf = np.array(cf); pl = cb >= 0
    print(f"--- catalogue H = 8 at hv {OFF}: scored {int(scored.sum())}, found {len(set(cb[cf & pl & scored[np.maximum(cb, 0)]].tolist()))}; chance candidates (unplanted run, p < 0.9): {len(un)}")
    for NS in (22,):
        for MULT in (1, 3, 10, 30):
            rng = np.random.default_rng(1)
            subs = [rng.random(NB) < NS / scored.sum() for _ in range(1500)]
            line = []
            for k, fn in V.items():
                pc = np.array([fn(c) for c in cands[:2000]]); pu = np.tile(np.array([fn(c) for c in un]), MULT)
                acc = []
                for sub in subs:
                    ns = int((sub & scored).sum())
                    if ns == 0: continue
                    keep = np.flatnonzero(pl & sub[np.maximum(cb, 0)])
                    allp = np.concatenate([pc[keep], pu]); o = np.argsort(-allp, kind='stable')[:2 * ns]
                    got = set(int(cb[keep[i]]) for i in o if i < len(keep) and cf[keep[i]])
                    acc.append(sum(w[j] for j in got) / w[sub].sum())
                line.append(f"{k}: {np.mean(acc):.4f}")
            print(f"  ~{NS} scored per patch, chance x{MULT}: " + ' | '.join(line))
