import csv, sys, os, json, numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(src, "exposures.csv"))))
rows.sort(key=lambda r: float(r["mjd_mid"]))
t0 = 60619.05
# 4 nights (day 0, 2, 5, 9), exposures 1.0 h apart: 3, 3, 3, 4
plan = [(0, 3), (2, 3), (5, 3), (9, 4)]
times = [t0 + d + j * (1.0 / 24) for d, n in plan for j in range(n)]
# interleave bands across nights: take rows in order i -> spread
idx = list(range(len(rows)))
for r, tm in zip(rows, times):
    dt = tm - float(r["mjd_mid"])
    r["mjd_mid"] = repr(tm); r["mjd_start"] = repr(float(r["mjd_start"]) + dt)
    T = Time(tm, format="mjd", scale="utc")
    p = (get_body_barycentric("earth", T) - get_body_barycentric("sun", T)).xyz.to_value("au")
    r["obs_x"], r["obs_y"], r["obs_z"] = (repr(float(v)) for v in p)
with open(os.path.join(dst, "exposures.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
m = json.load(open(os.path.join(src, "meta.json"))); m.update(first_mjd=times[0], last_mjd=times[-1], n_nights=4, sid="sF1")
json.dump(m, open(os.path.join(dst, "meta.json"), "w"))
