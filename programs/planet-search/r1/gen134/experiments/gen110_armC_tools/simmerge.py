"""simmerge.py WRAPPER_DIR SCEN LINKER.json STACK.json OUT.json : rerun the wrapper's merge+write offline."""
import sys, os, csv, math, json, importlib.util
import numpy as np
wd, scen, fl, fs, fo = sys.argv[1:6]
sys.path.insert(0, os.path.join(wd, "linker"))
spec = importlib.util.spec_from_file_location("wrap", os.path.join(wd, "finder.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
nt = {}
for r in rows:
    t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
    nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
night_t = np.array([v[0] for v in nt.values()]); night_o = np.array([v[1] for v in nt.values()])
A = W.load(fl); S = W.load(fs)
m = W.merge(A, S, night_t, night_o)
W.write(fo, m)
# report where promoted stack candidates land
ms = sorted(W.rank_balance(m), key=lambda c: -c["p"])
for k, c in enumerate(ms):
    if c.get("k3pro") or ("zc" in c and k < 40):
        print("rank", k, "p %.3f" % c["p"], "src", c.get("src"), "zc", round(c.get("zc", -1), 2), "k3pro", c.get("k3pro", 0))
print("p at rank 25: %.3f" % ms[24]["p"] if len(ms) > 24 else "")
