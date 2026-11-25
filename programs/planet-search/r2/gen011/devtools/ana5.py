"""ana5.py RUNDIR INJ.json [SVRUNDIR] : the final OUT.json (needs FINDER_DUMPC: _xy) against the injected bodies, grader's rule."""
import sys, pickle, json, numpy as np
D = pickle.load(open(sys.argv[1] + "/cands.pkl", "rb")); bodies = json.load(open(sys.argv[2]))
D2 = pickle.load(open(sys.argv[3] + "/cands.pkl", "rb")) if len(sys.argv) > 3 else D
out = json.load(open(sys.argv[1] + "/OUT.json"))["candidates"]
out = sorted(out, key=lambda c: -c["p"])
night, ixy, wS = D["night"], D["inj_xy"], D["wS"]; isv = D2.get("inj_sv")
found = {}; nonj = 0; rows = []
for rk, c in enumerate(out):
    xy = np.array(c["_xy"]); d = np.hypot(xy[:, None, 0] - ixy[:, :, 0], xy[:, None, 1] - ixy[:, :, 1])
    hit = None
    for b in range(d.shape[1]):
        if len(set(night[d[:, b] < 2.0])) >= 3: hit = b; break
    near = int(np.argmin(np.sort(d, axis=0)[1]))
    rows.append((rk, c["p"], c["score"], c.get("two", 0), c.get("alt_of"), hit, near, np.round(d[:, near], 1).tolist(), c["snr"]))
    if hit is None:
        if c.get("alt_of") is None: nonj += 1
    elif hit not in found: found[hit] = (rk, nonj, c["score"], c.get("two", 0), c.get("alt_of") is not None)
def cls(b):
    if isv is None: return 0, 0.0
    s = isv[b]; ok = np.isfinite(s); nts = np.unique(night[ok])
    stk = lambda m: float(np.sum(s[m] * wS[m]) / np.sqrt(np.sum(wS[m] ** 2))) if m.any() else 0.0
    n1 = sum(1 for n in nts if (s[ok & (night == n)] >= 5).any()); n2 = sum(1 for n in nts if stk(ok & (night == n)) >= 5); Z = stk(ok)
    return (1 if n1 >= 3 else 2 if n2 >= 3 else 4 if (Z >= 5 and len(nts) >= 3) else 0), Z
tot = {0: 0, 1: 0, 2: 0, 4: 0}; got = dict(tot); g25 = dict(tot)
for b in range(ixy.shape[1]):
    w, Z = cls(b); tot[w] += 1
    f = found.get(b)
    if f: got[w] += 1; g25[w] += f[0] < 25
    if w or f:
        print(f"body {b:2d} mag {bodies[b]['mag']:.2f} r {bodies[b]['r']:.0f} w{w} Z {Z:5.1f} sv {np.round(isv[b], 1).tolist() if isv is not None else ''}", ("FOUND rank %d (%d non-inj above) score %.1f two=%s alt=%s" % f) if f else "-")
print("classes", tot, "found", got, "in top 25", g25, "| weighted total", sum(k * v for k, v in tot.items()), "found", sum(k * v for k, v in got.items()), "top25", sum(k * v for k, v in g25.items()))
print("top 30 of OUT.json:")
for r in rows[:30]:
    print("  ", r[0], f"p={r[1]:.3f} s={r[2]:.1f}", "two%d" % r[3] if r[3] else "    ", "ALT" if r[4] is not None else "   ", "HIT b%d" % r[5] if r[5] is not None else "      ", "near b%d" % r[6], r[7], r[8])
print("two-night candidates in OUT:", sum(1 for r in rows if r[3] and r[4] is None), "alternates:", sum(1 for r in rows if r[4] is not None))
