"""rr_data.py OUT.npz RUNDIR:INJ.json ... : features + labels of the final lists of injection runs (re-ranker training)."""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rerank, finder
X = []; Y = []; G = []; RK = []; BODY = []; BW = {}
for gi, arg in enumerate(sys.argv[2:]):
    run, injf = arg.split(":")
    if not os.path.exists(run + "/cands.pkl"): print("missing", run); continue
    D = pickle.load(open(run + "/cands.pkl", "rb")); out = json.load(open(run + "/OUT.json"))["candidates"]
    out = sorted(out, key=lambda c: -c["p"])
    night, ixy, wS, t, isv = D["night"], D["inj_xy"], D["wS"], D["t"], D["inj_sv"]
    refs = rerank.scen_refs([c["score"] for c in out if c.get("alt_of") is None], [c["Z"] for c in out if c.get("alt_of") is None])
    for b in range(ixy.shape[1]):
        s = isv[b]; ok = np.isfinite(s); nts = np.unique(night[ok])
        stk = lambda m: float(np.sum(s[m] * wS[m]) / np.sqrt(np.sum(wS[m] ** 2))) if m.any() else 0.0
        n1 = sum(1 for n in nts if (s[ok & (night == n)] >= 5).any()); n2 = sum(1 for n in nts if stk(ok & (night == n)) >= 5)
        BW[(gi, b)] = (1 if n1 >= 3 else 2 if n2 >= 3 else 4 if (stk(ok) >= 5 and len(nts) >= 3) else 0, stk(ok))
    for rk, c in enumerate(out):
        xy = np.array(c["_xy"]); d = np.hypot(xy[:, None, 0] - ixy[:, :, 0], xy[:, None, 1] - ixy[:, :, 1])
        hit = -1
        for b in range(d.shape[1]):
            if len(set(night[d[:, b] < 2.0])) >= 3: hit = b; break
        X.append(rerank.features(c, wS, night, t, refs, finder.orbit_prior)); Y.append(hit >= 0); G.append(gi); RK.append(rk); BODY.append(hit)
        # alt / promoted flags
    print(run, "cands", len(out), "hits", sum(1 for g, y in zip(G, Y) if g == gi and y), "refs", np.round(refs, 1), "E", len(night))
pickle.dump({"X": np.array(X), "Y": np.array(Y), "G": np.array(G), "RK": np.array(RK), "BODY": np.array(BODY), "BW": BW, "runs": sys.argv[2:],
             "ALT": None}, open(sys.argv[1], "wb"))
