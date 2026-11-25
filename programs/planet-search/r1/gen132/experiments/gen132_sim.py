import json, sys, math
import numpy as np
W = {"A": 1, "B": 2, "C": 4}
def fac(f, a):
    if f is None: return 1.0
    lo, hi, fl, st, pr = a
    z = f["zlbno"]
    m = fl + (1 - fl) * min(max((z - lo) / (hi - lo), 0), 1)
    if f["zoff"] > max(3.0, z): m *= st
    return m
def prom(p, f, a):
    pr = a[4]
    if pr and f is not None and f["zlbno"] >= pr and f["npos"] >= 3 and f["zoff"] < max(3.0, 0.5 * f["zlbno"]):
        return max(p, 0.5)
    return p
def rb(ps):  # rank_balance as in the wrapper (Q0 .7, R 25)
    o = np.argsort(-ps, kind="stable"); out = ps.copy()
    for r, i in enumerate(o): out[i] = 1 - (1 - ps[i]) * (1 - 0.7 * math.exp(-r / 25))
    return out
def run(a, names, verbose=False):
    allr = []
    tot = 0
    for n in names:
        tab = json.load(open(f"feats_{n}_tab.json"))
        # undo rank_balance approximately: use p as given (already rank balanced); re-apply after adjust
        ps = np.array([t["p"] for t in tab])
        fs = [t if "zon" in t else None for t in tab]
        q = np.array([prom(p * fac(f, a), f, a) for p, f in zip(ps, fs)])
        q = rb(q) if a != BASE else ps
        o = np.argsort(-q, kind="stable")
        seen = set(); sc = 0
        for r, i in enumerate(o[:25]):
            lab = tab[i]["lab"]
            if lab in W: sc += W[lab]
        tot += sc
        for i in range(len(tab)): allr.append((q[i], tab[i]["lab"]))
        if verbose: print(n, sc, [ (r, tab[i]["lab"]) for r, i in enumerate(o[:40]) if tab[i]["lab"] in W])
    allr.sort(key=lambda x: -x[0])
    g = sum(W.get(l, 0) for _, l in allr[:25 * len(names)])
    return tot, g
BASE = (0, 1, 1, 1, 0)
names = sys.argv[1:] or ["injCol", "injC", "injD"]
print("base", run(BASE, names, True))
for a in [(1.5, 4.5, 0.5, 0.3, 0), (2.5, 5.5, 0.3, 0.3, 0), (1.5, 4.5, 0.5, 1, 0), (1.5, 4.5, 0.5, 0.3, 8), (1.5, 4.5, 0.5, 0.3, 10), (2.0, 5.0, 0.4, 0.3, 8), (1, 4, 0.6, 0.5, 0), (2.5, 5.5, 0.3, 0.3, 8)]:
    print(a, run(a, names, True))
