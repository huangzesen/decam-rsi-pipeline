# usage: replay.py SUBDIR SCEN RUNJSON OUTJSON  (env TNO_* applies)
import sys, os, csv, math, json
import numpy as np
sys.path.insert(0, sys.argv[1]); import finder as F
scen, run, out = sys.argv[2], sys.argv[3], sys.argv[4]
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
nt = {}
for r in rows:
    t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
    nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
night_t = np.array([v[0] for v in nt.values()]); night_o = np.array([v[1] for v in nt.values()])
F.set_expt(rows)
A = F.load(run + ".linker.json"); S = F.load(run + ".stack.json")
F.write(out, F.merge(A, S, night_t, night_o))
