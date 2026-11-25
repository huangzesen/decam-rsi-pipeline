"""Offline: re-run the wrapper merge on kept linker/stack outputs with a given env, write OUT json."""
import sys, os, json, csv, math, importlib
sub, linj, stj, scen, out = [os.path.abspath(x) for x in sys.argv[1:6]]
sys.path.insert(0, sub)
import numpy as np
os.chdir(sub)
spec = importlib.util.spec_from_file_location("wrap", os.path.join(sub, "finder.py"))
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
nt = {}
for r in rows:
    t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
    nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
night_t = np.array([v[0] for v in nt.values()]); night_o = np.array([v[1] for v in nt.values()])
A, S = w.load(linj), w.load(stj)
fin = w.write(out, w.merge(A, S, night_t, night_o))
