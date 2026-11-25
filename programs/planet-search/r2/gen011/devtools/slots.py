"""slots.py RUNDIR ... : would K of the 25 slots given to the best candidates by (score - lam * pos) find more weight?
Uses the candidate dump (all scored candidates), de-duplicated by track."""
import sys, pickle, numpy as np
res = {}
for run in sys.argv[1:]:
    D = pickle.load(open(run + "/cands.pkl", "rb")); C = [c for c in D["cands"] if c.get("score") is not None and c.get("xy") is not None]
    night, ixy, wS, isv = D["night"], D["inj_xy"], D["wS"], D["inj_sv"]; E = len(night)
    W = {}
    for b in range(ixy.shape[1]):
        s = isv[b]; ok = np.isfinite(s); nts = np.unique(night[ok])
        stk = lambda m: float(np.sum(s[m] * wS[m]) / np.sqrt(np.sum(wS[m] ** 2))) if m.any() else 0.0
        n1 = sum(1 for n in nts if (s[ok & (night == n)] >= 5).any()); n2 = sum(1 for n in nts if stk(ok & (night == n)) >= 5)
        W[b] = 1 if n1 >= 3 else 2 if n2 >= 3 else 4 if (stk(ok) >= 5 and len(nts) >= 3) else 0
    def hit(c):
        d = np.hypot(c["xy"][:, None, 0] - ixy[:, :, 0], c["xy"][:, None, 1] - ixy[:, :, 1])
        for b in range(d.shape[1]):
            if len(set(night[d[:, b] < 2.0])) >= 3: return b
        return -1
    def pick(order, n, tracks, used):
        out = []
        for i in order:
            if len(out) >= n: break
            c = C[i]; ds = [int(d) for d in c["md"] if d >= 0]
            if sum(d in used for d in ds) >= max(1, len(ds) // 2): continue
            if tracks and (np.sum(np.abs(np.array(tracks) - c["xy"][None]).max(axis=2) < 2.0, axis=1) >= max(2, E // 2)).any(): continue
            tracks.append(c["xy"]); used.update(ds); out.append(i)
        return out
    sc = np.array([c["score"] for c in C]); pos = np.array([c["pos"] for c in C]); two = np.array([bool(c.get("two")) for c in C])
    for K in (0, 3, 5, 8):
        for lam in (1.0, 0.5):
            if K == 0 and lam != 1.0: continue
            tr = []; us = set()
            a = pick(np.argsort(-sc), 25 - K, tr, us)
            alt = sc - lam * pos; alt[two] = -1e9
            b_ = pick(np.argsort(-alt), K, tr, us) if K else []
            bodies = set(hit(C[i]) for i in a + b_) - {-1}
            res.setdefault((K, lam), []).append(sum(W[b] for b in bodies))
    print(run[-14:], {k: v[-1] for k, v in res.items()})
print("TOTAL", {k: sum(v) for k, v in res.items()})
