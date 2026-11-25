import sys, os, json, csv, math, importlib
import numpy as np
sub = sys.argv[1]; scen = sys.argv[2]; outp = sys.argv[3]
sys.path.insert(0, sub)
F = importlib.import_module("finder")
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
nt = {}
for r in rows:
    t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
    nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
night_t = np.array([v[0] for v in nt.values()]); night_o = np.array([v[1] for v in nt.values()])
A = F.load(outp + ".linker.json"); B = F.load(outp + ".stack.json")
print("stk in linker out:", sum(c.get("kind") == "stk" for c in A), "top-25 p of stk:", [round(c["p"],3) for c in A if c.get("kind")=="stk"][:10])
for ad in (0, 1):
    F.ADOPT = ad
    cs = F.rank_balance(F.merge([dict(c) for c in A], [dict(c) for c in B], night_t, night_o))[:200]
    print("ADOPT", ad, "states changed in top 50:", sum(1 for c in cs[:50] if "_adopt_p" in c))
