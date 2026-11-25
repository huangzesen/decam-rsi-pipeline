"""ev21.py RUNDIR ... : A/B of the recovered-map term (gen021) inside one FINDER_INJECT + FINDER_DUMPC run: the candidate
list ranked by score (with the term) and by score - rec (without; candidates that exist only through a recovered
night left out), injected weight in the top 25 of each, and every injected body's rank in both."""
import sys, pickle, numpy as np
for run in sys.argv[1:]:
    D = pickle.load(open(run + "/cands.pkl", "rb"))
    night, ixy, wS, isv = D["night"], D["inj_xy"], D["wS"], D["inj_sv"]
    E = len(night)
    cs = [c for c in D["cands"] if c.get("score") is not None and c.get("xy") is not None]
    for c in cs:
        d = np.hypot(c["xy"][:, None, 0] - ixy[:, :, 0], c["xy"][:, None, 1] - ixy[:, :, 1])
        c["b"] = [b for b in np.nonzero((d < 2.0).sum(0) >= 3)[0] if len(set(night[d[:, b] < 2.0])) >= 3]
    wt = {}
    for b in range(ixy.shape[1]):
        s = isv[b]; ok = np.isfinite(s); nts = np.unique(night[ok])
        stk = lambda m: float(np.sum(s[m] * wS[m]) / np.sqrt(np.sum(wS[m] ** 2))) if m.any() else 0.0
        n1 = sum(1 for n in nts if (s[ok & (night == n)] >= 5).any()); n2 = sum(1 for n in nts if stk(ok & (night == n)) >= 5); Z = stk(ok)
        w = 1 if n1 >= 3 else 2 if n2 >= 3 else 4 if (Z >= 5 and len(nts) >= 3) else 0
        if w: wt[b] = (w, Z, int((~ok).sum()))
    res = {}
    for name in ("with", "without"):
        key = (lambda c: c["score"]) if name == "with" else (lambda c: c["score"] - (c.get("rec") or 0.0))
        pool = cs if name == "with" else [c for c in cs if not c.get("recup")]
        tracks = []; out = []
        for c in sorted(pool, key=lambda c: -key(c)):
            if tracks:
                dd = np.abs(np.array(tracks) - c["xy"][None]).max(axis=2)
                if (np.sum(dd < 2.0, axis=1) >= max(2, E // 2)).any(): continue
            tracks.append(c["xy"]); out.append(c)
            if len(out) >= 200: break
        rank = {}
        for rk, c in enumerate(out):
            for b in c["b"]:
                rank.setdefault(b, (rk, key(c)))
        got = sum(wt[b][0] for b in wt if rank.get(b, (999,))[0] < 25)
        ch = [key(c) for c in out if not c["b"]]
        res[name] = rank
        print(f"{run} {name:8s}: weight in top 25 {got} of {sum(w[0] for w in wt.values())}, injected in top 25 {sum(1 for b in rank if rank[b][0] < 25)}, top-100 {sum(1 for b in rank if rank[b][0] < 100)}; chance scores (1st, 5th, 20th, 40th non-injected): {[round(ch[i], 1) for i in (0, 4, 19, 39) if i < len(ch)]}")
    for b in sorted(wt, key=lambda b: -wt[b][1]):
        a_, n_ = res["with"].get(b), res["without"].get(b)
        if a_ or n_:
            print("   body %2d w %d Z %5.1f masked exps %d | with: %s | without: %s" % (b, wt[b][0], wt[b][1], wt[b][2], a_ and (a_[0], round(a_[1], 1)), n_ and (n_[0], round(n_[1], 1))))
    rc = np.array([c.get("rec") or 0.0 for c in cs]); nb = np.array([bool(c["b"]) for c in cs]); nr = np.array([c.get("nrec") or 0 for c in cs])
    for lab, m in (("injected", nb), ("others", ~nb)):
        if m.any(): print(f"   {lab}: {m.sum()} scored candidates, {np.mean(nr[m] > 0):.2f} have a recovered exposure, mean term of those {rc[m & (nr > 0)].mean() if (m & (nr > 0)).any() else 0:.2f}, upgraded to 3 nights {sum(1 for c, q in zip(cs, m) if q and c.get('recup'))}")
