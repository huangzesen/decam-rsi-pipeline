"""ev15.py RUNDIR ... : injected bodies in the final list of a FINDER_INJECT + FINDER_DUMPC run: weight in the top 25,
rank / score / deep tag of every findable body, how many of the top 25 are deep-pair candidates (gen015)."""
import sys, json, pickle, numpy as np
for run in sys.argv[1:]:
    D = pickle.load(open(run + "/cands.pkl", "rb")); out = sorted(json.load(open(run + "/OUT.json"))["candidates"], key=lambda c: -c["p"])
    night, ixy, wS, isv = D["night"], D["inj_xy"], D["wS"], D["inj_sv"]
    rank = {}
    for rk, c in enumerate(out):
        xy = np.array(c["_xy"]); d = np.hypot(xy[:, None, 0] - ixy[:, :, 0], xy[:, None, 1] - ixy[:, :, 1])
        for b in range(d.shape[1]):
            if len(set(night[d[:, b] < 2.0])) >= 3 and b not in rank: rank[b] = rk
    # best scored candidate of every body among all scored candidates (also those cut from the list)
    best = {}
    for c in D["cands"]:
        if c.get("score") is None or c.get("xy") is None: continue
        d = np.hypot(c["xy"][:, None, 0] - ixy[:, :, 0], c["xy"][:, None, 1] - ixy[:, :, 1])
        for b in np.nonzero((d < 2.0).sum(0) >= 3)[0]:
            if len(set(night[d[:, b] < 2.0])) >= 3 and (b not in best or c["score"] > best[b]): best[b] = c["score"]
    tot = 0; got = 0; rows = []
    for b in range(ixy.shape[1]):
        s = isv[b]; ok = np.isfinite(s); nts = np.unique(night[ok])
        stk = lambda m: float(np.sum(s[m] * wS[m]) / np.sqrt(np.sum(wS[m] ** 2))) if m.any() else 0.0
        n1 = sum(1 for n in nts if (s[ok & (night == n)] >= 5).any()); n2 = sum(1 for n in nts if stk(ok & (night == n)) >= 5); Z = stk(ok)
        w = 1 if n1 >= 3 else 2 if n2 >= 3 else 4 if (Z >= 5 and len(nts) >= 3) else 0
        if not w: continue
        tot += w; got += w if rank.get(b, 999) < 25 else 0
        rows.append((Z, b, w, rank.get(b), best.get(b), s))
    print(run, "weight in top 25: %d of %d" % (got, tot), "| score of rank 25: %.1f" % out[min(24, len(out) - 1)]["score"],
          "| deep in top 25:", sum(1 for c in out[:25] if c.get("deep")), "| injected in top 25:", sum(1 for b in rank if rank[b] < 25))
    for Z, b, w, rk, bs, s in sorted(rows, key=lambda r: -r[0]):
        if Z >= 6.5:
            c = out[rk] if rk is not None else None
            print("   body %2d w %d Z %5.1f rank %s %s best scored %s  snr %s" % (b, w, Z, rk, ("score %.1f%s" % (c["score"], " DEEP" if c.get("deep") else "")) if c else "", None if bs is None else round(bs, 1), np.round(s, 1).tolist()))
