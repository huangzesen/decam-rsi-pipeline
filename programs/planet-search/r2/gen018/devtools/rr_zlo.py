import sys, pickle, json, numpy as np
def load(run, injf):
    D = pickle.load(open(run + "/cands.pkl", "rb")); out = json.load(open(run + "/OUT.json"))["candidates"]
    night, ixy, wS, isv = D["night"], D["inj_xy"], np.asarray(D["wS"], float), D["inj_sv"]
    def stk(s, m):
        return float(np.sum(s[m] * wS[m]) / np.sqrt(np.sum(wS[m] ** 2))) if m.any() else 0.0
    def cls(b):
        s = isv[b]; ok = np.isfinite(s); nts = np.unique(night[ok])
        n1 = sum(1 for n in nts if (s[ok & (night == n)] >= 5).any()); n2 = sum(1 for n in nts if stk(s, ok & (night == n)) >= 5); Z = stk(s, ok)
        return (1 if n1 >= 3 else 2 if n2 >= 3 else 4 if (Z >= 5 and len(nts) >= 3) else 0)
    W = [cls(b) for b in range(ixy.shape[1])]
    rows = []
    for c in out:
        xy = np.array(c["_xy"]); d = np.hypot(xy[:, None, 0] - ixy[:, :, 0], xy[:, None, 1] - ixy[:, :, 1])
        hit = None
        for b in range(d.shape[1]):
            if len(set(night[d[:, b] < 2.0])) >= 3: hit = b; break
        s = np.array([np.nan if v is None else v for v in c["snr"]], float); ok = np.isfinite(s)
        Z = stk(s, ok)
        zlo = min([stk(s, ok & (night != n)) for n in np.unique(night[ok])] + [Z])
        sc = c["score"] - (0.6 * min(max(c["Z"] - 8.5, 0.0), 8.0) if "/zb_" in run else 0.0)
        rows.append((c["p"], sc, c["Z"], hit, zlo))
    return rows, W, sum(W)
def ev(rows, W, key):
    o = sorted(range(len(rows)), key=lambda q: -key(rows[q]))[:25]
    got = set(rows[q][3] for q in o if rows[q][3] is not None)
    return sum(W[b] for b in got)
runs = [a.split(",") for a in sys.argv[1:]]
data = [load(*r) for r in runs]
print("total weight", [d[2] for d in data], "by score:", [ev(d[0], d[1], lambda r: r[1]) for d in data])
print("Z bonus 0.6/8.5/8:", [ev(d[0], d[1], lambda r: r[1] + 0.6 * min(max(r[2] - 8.5, 0.0), 8.0)) for d in data])
for a in (0.3, 0.6, 1.0, 1.5, 2.5):
    for z0 in (4.0, 5.0, 6.0, 7.0):
        for cap in (3.0, 6.0, 20.0):
            v = [ev(d[0], d[1], lambda r: r[1] + a * min(max(r[4] - z0, 0.0), cap)) for d in data]
            print("Zlo", a, z0, cap, v, sum(v))
