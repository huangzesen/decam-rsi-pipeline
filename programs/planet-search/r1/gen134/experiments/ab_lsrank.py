import os, sys, csv, math, json, subprocess
flag, out = sys.argv[1], sys.argv[2]
os.environ["TNO_LSRANK"] = flag
sys.path.insert(0, "$RSI_SANDBOX/r1/gen078/workspace/submission")
import numpy as np, finder
T = "$RSI_SANDBOX/r1/gen078/tmp"; scen = T + "/inj7"
rows = list(csv.DictReader(open(scen + "/exposures.csv")))
nt = {}
for r in rows:
    t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
    nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
night_t = np.array([v[0] for v in nt.values()]); night_o = np.array([v[1] for v in nt.values()])
finder.write(out, finder.merge(finder.load(T + "/base.json.linker.json"), finder.load(T + "/base.json.stack.json"), night_t, night_o))
