import sys, json, csv, math, os
import numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen081/workspace/submission")
import finder as F
scen = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
nt = {}
for r in rows:
    t_ = float(r["mjd_mid"]); nt.setdefault(math.floor(t_ - .5), (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
T = np.array([v[0] for v in nt.values()]); O = np.array([v[1] for v in nt.values()])
for suf in (".linker.json", ""):
    a = json.load(open(sys.argv[2] + suf))["candidates"]; b = json.load(open(sys.argv[3] + suf))["candidates"]
    ua, ub = F.tracks(a, T, O), F.tracks(b, T, O)
    cm = math.cos(2 * F.ARC)
    def found(u, U):
        return ((np.einsum("knd,nd->kn", U, u) > cm).sum(1) >= 3).any()
    for name, X, UX, UY in (("only-in-A", a, ua, ub), ("only-in-B", b, ub, ua)):
        miss = [(r, round(X[r]["p"], 3)) for r in range(len(X)) if not found(UX[r], UY)]
        print(suf or "final", name, "top-50 ranks/p not in other:", [m for m in miss if m[0] < 50])
print("---")
fa = json.load(open(sys.argv[2]))["candidates"]
for nm in (sys.argv[2], sys.argv[3]):
    for suf in (".linker.json", ".stack.json"):
        X = json.load(open(nm + suf))["candidates"]; UX = F.tracks(X, T, O)
        u = F.tracks(fa[:4], T, O)
        print(nm + suf, [[ (r, round(X[r]["p"],3)) for r in range(len(X)) if (np.einsum("nd,nd->n", UX[r], u[k]) > cm).sum() >= 3][:2] for k in range(4)])
