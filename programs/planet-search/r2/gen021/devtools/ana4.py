"""ana4.py DUMPDIR INJ.json [SVDUMPDIR]: per injected body: grader-like weight class from true S/N, found?, rank, n non-inj above."""
import sys, pickle, json, numpy as np
D = pickle.load(open(sys.argv[1] + "/cands.pkl", "rb")); bodies = json.load(open(sys.argv[2]))
D2 = pickle.load(open(sys.argv[3] + "/cands.pkl", "rb")) if len(sys.argv) > 3 else D
night, ixy = D["night"], D["inj_xy"]; E = len(night); cs = D["cands"]; wS = D["wS"]
isv = D2["inj_sv"]
cs = sorted([c for c in cs if c.get("score") is not None and np.isfinite(c["score"]) and c["score"] > -1e8], key=lambda c: -c["score"])
tracks = []; out = []
for c in cs:
    xy = c["xy"]
    if tracks:
        dd = np.abs(np.array(tracks) - xy[None]).max(axis=2)
        if (np.sum(dd < 2.0, axis=1) >= max(2, E // 2)).any(): continue
    tracks.append(xy); out.append(c)
    if len(out) >= 400: break
found = {}; nonj = []
for rk, c in enumerate(out):
    xy = c["xy"]
    d = np.hypot(xy[:, None, 0] - ixy[:, :, 0], xy[:, None, 1] - ixy[:, :, 1])
    hit = None
    for b in range(d.shape[1]):
        if len(set(night[d[:, b] < 2.0])) >= 3: hit = b; break
    if hit is None: nonj.append(rk)
    elif hit not in found: found[hit] = (rk, len(nonj), c["score"], bool(c.get("two")))
print("non-inj scores at non-inj rank 1,5,10,25,50,100,200:", [round(out[nonj[k]]["score"], 1) for k in (0, 4, 9, 24, 49, 99, 199) if k < len(nonj)])
tot = {1: 0, 2: 0, 4: 0}; got = {1: 0, 2: 0, 4: 0}; got25 = {1: 0, 2: 0, 4: 0}
for b in range(isv.shape[0]):
    s = isv[b]; ok = np.isfinite(s)
    nts = np.unique(night[ok])
    n1 = sum(1 for n in nts if (s[ok & (night == n)] >= 5).any())
    def stk(m):
        return float(np.sum(s[m] * wS[m]) / np.sqrt(np.sum(wS[m] ** 2))) if m.any() else 0.0
    ns = [stk(ok & (night == n)) for n in nts]
    n2 = sum(1 for v in ns if v >= 5)
    Z = stk(ok)
    w = 1 if n1 >= 3 else 2 if n2 >= 3 else 4 if (Z >= 5 and len(nts) >= 3) else 0
    f = found.get(b)
    if w:
        tot[w] += 1
        if f: got[w] += 1
        if f and f[1] < 25: got25[w] += 1
    print(f"body {b:2d} mag {bodies[b]['mag']:.2f} r {bodies[b]['r']:.0f} w{w} Z {Z:5.1f} nightS/N {np.round(ns, 1).tolist()} sv {np.round(s, 1).tolist()}", "FOUND rank %d nonj_above %d score %.1f%s" % (f[0], f[1], f[2], " two" if f[3] else "") if f else "-")
print("class totals", tot, "found", got, "found with <25 non-inj above", got25)
print("weighted: total", sum(k * v for k, v in tot.items()), "found", sum(k * v for k, v in got.items()), "top", sum(k * v for k, v in got25.items()))
