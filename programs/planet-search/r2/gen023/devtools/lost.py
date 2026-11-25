"""lost.py RUNDIR ... : every findable injected body that is not in the top 25: was it linked, and what did it score?"""
import sys, re, json, pickle, numpy as np
tot = {}
for run in sys.argv[1:]:
    D = pickle.load(open(run + "/cands.pkl", "rb")); out = sorted(json.load(open(run + "/OUT.json"))["candidates"], key=lambda c: -c["p"])
    night, ixy, wS, isv = D["night"], D["inj_xy"], D["wS"], D["inj_sv"]
    s25 = out[24]["score"]; rank = {}
    for rk, c in enumerate(out):
        xy = np.array(c["_xy"]); d = np.hypot(xy[:, None, 0] - ixy[:, :, 0], xy[:, None, 1] - ixy[:, :, 1])
        for b in range(d.shape[1]):
            if len(set(night[d[:, b] < 2.0])) >= 3 and b not in rank: rank[b] = rk
    ref = {}
    for l in open(run + "/log"):
        m = re.search(r"inj (\d+) refined: bright ([-\d.]+) pos ([-\d.]+) nights (\d+) fitted (\w+) sv (\[.*?\]) off (\[.*?\])", l)
        if m:
            b = int(m.group(1)); off = np.array(eval(m.group(7))); good = len(set(night[off < 2.0])) >= 3
            e = (float(m.group(2)) + float(m.group(3)), float(m.group(2)), float(m.group(3)), int(m.group(4)), good)
            if good and (b not in ref or e[0] > ref[b][0]): ref[b] = e
    print(run[-14:], "score of rank 25: %.1f" % s25)
    for b in range(ixy.shape[1]):
        s = isv[b]; ok = np.isfinite(s); nts = np.unique(night[ok])
        stk = lambda m: float(np.sum(s[m] * wS[m]) / np.sqrt(np.sum(wS[m] ** 2))) if m.any() else 0.0
        n1 = sum(1 for n in nts if (s[ok & (night == n)] >= 5).any()); n2 = sum(1 for n in nts if stk(ok & (night == n)) >= 5); Z = stk(ok)
        w = 1 if n1 >= 3 else 2 if n2 >= 3 else 4 if (Z >= 5 and len(nts) >= 3) else 0
        if not w: continue
        if rank.get(b, 999) < 25: k = "top25"
        elif Z < 8: k = "Z<8"
        elif b not in ref: k = "never linked right (Z>=8)"
        elif ref[b][3] < 3: k = "linked, nights<3"
        elif ref[b][1] >= s25 + 0.5: k = "linked, bright>=s25 but prior sinks it"
        else: k = "linked, bright<s25"
        tot.setdefault(k, [0, 0]); tot[k][0] += 1; tot[k][1] += w
        if k not in ("top25", "Z<8"): print("   body", b, "w", w, "Z %.1f" % Z, k, ref.get(b), np.round(s, 1).tolist())
print({k: tuple(v) for k, v in tot.items()})
