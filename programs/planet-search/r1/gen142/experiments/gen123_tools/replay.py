# usage: replay.py SUBDIR SCEN BASE_OUT_JSON NEW_OUT_JSON   (env vars pass to the wrapper module)
import sys, os, csv, math, json, importlib.util
import numpy as np
sub, scen, base, new = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("wf", os.path.join(sub, "finder.py"))
sys.path.insert(0, os.path.join(sub, "linker"))
wf = importlib.util.module_from_spec(spec); spec.loader.exec_module(wf)
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
nt = {}
for r in rows:
    t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
    nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
night_t = np.array([v[0] for v in nt.values()]); night_o = np.array([v[1] for v in nt.values()])
if hasattr(wf, "set_expt"): wf.set_expt(rows)
A = wf.load(base + ".linker.json"); S = wf.load(base + ".stack.json")
wf.write(new, wf.merge(A, S, night_t, night_o))
