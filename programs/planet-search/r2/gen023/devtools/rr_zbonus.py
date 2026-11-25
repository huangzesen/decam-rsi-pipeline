import sys, pickle, json, numpy as np
def load(run, injf):
    D = pickle.load(open(run + "/cands.pkl", "rb")); out = json.load(open(run + "/OUT.json"))["candidates"]
    night, ixy, wS, isv = D["night"], D["inj_xy"], D["wS"], D["inj_sv"]
    def cls(b):
        s = isv[b]; ok = np.isfinite(s); nts = np.unique(night[ok])
        stk = lambda m: float(np.sum(s[m] * wS[m]) / np.sqrt(np.sum(wS[m] ** 2))) if m.any() else 0.0
        n1 = sum(1 for n in nts if (s[ok & (night == n)] >= 5).any()); n2 = sum(1 for n in nts if stk(ok & (night == n)) >= 5); Z = stk(ok)
        return (1 if n1 >= 3 else 2 if n2 >= 3 else 4 if (Z >= 5 and len(nts) >= 3) else 0)
    W = [cls(b) for b in range(ixy.shape[1])]
    rows = []
    for c in out:
        xy = np.array(c["_xy"]); d = np.hypot(xy[:, None, 0] - ixy[:, :, 0], xy[:, None, 1] - ixy[:, :, 1])
        hit = None
        for b in range(d.shape[1]):
            if len(set(night[d[:, b] < 2.0])) >= 3: hit = b; break
        rows.append((c["p"], c["score"], c["Z"], hit, c.get("two", 0), c.get("alt_of") is not None, c["snr"]))
    return rows, W, sum(W)
def ev(rows, W, key):
    o = sorted(range(len(rows)), key=lambda q: -key(rows[q]))[:25]
    got = set(rows[q][3] for q in o if rows[q][3] is not None)
    return sum(W[b] for b in got)
runs = [a.split(",") for a in sys.argv[1:]]
data = [load(*r) for r in runs]
print("total weight", [d[2] for d in data], "by p:", [ev(d[0], d[1], lambda r: r[0]) for d in data], "by score:", [ev(d[0], d[1], lambda r: r[1]) for d in data])
for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5):
    for z0 in (7.0, 8.0, 9.0):
        for cap in (4.0, 8.0, 20.0):
            print(a, z0, cap, [ev(d[0], d[1], lambda r: r[1] + a * min(max(r[2] - z0, 0.0), cap)) for d in data])
