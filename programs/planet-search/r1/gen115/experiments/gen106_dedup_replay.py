"""Post-hoc: apply sky dedup at TOL arcsec (>=3 nights) to a finished OUT.json, write OUT_dd.json."""
import sys, json, math, csv, os
import numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen106/workspace/submission/linker")
import orbit
ARC = math.pi / 180 / 3600
out, scen, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
nt = {}
for r in rows:
    t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
    nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
T = np.array([v[0] for v in nt.values()]); O = np.array([v[1] for v in nt.values()])
cs = sorted(json.load(open(out))["candidates"], key=lambda c: -c["p"])
U = np.array([orbit.predict_dirs(np.array(c["state"]), c["epoch_mjd"], T, O) for c in cs])
cosm = math.cos(tol * ARC); keep = []
for j in range(len(cs)):
    if keep and ((np.einsum("knd,nd->kn", U[keep], U[j]) > cosm).sum(1) >= 3).any():
        continue
    keep.append(j)
ps = [c["p"] for c in cs]
new = [dict(cs[k], p=ps[i]) for i, k in enumerate(keep)]
print(f"{len(cs)} -> {len(keep)}; removed ranks {[j for j in range(len(cs)) if j not in keep][:30]}")
json.dump({"candidates": new}, open(out.replace(".json", f"_dd{tol:g}.json"), "w"))
