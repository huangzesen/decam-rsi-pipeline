"""mkinj.py SCEN OUT.json N MAG_LO MAG_HI SEED : random TNO-like bodies through a practice field (development only).

Same generator as the RSI program r2/gen006 devtools/mkinj.py (copied, so both programs get the same kind of bodies):
distance mostly 41 +- 6 au (30-60) with a 15% tail to 80 au, prograde near-circular motion with a cold (4 deg) and hot
(16 deg) inclination component, (v_t/v_c)^2 0.75-1.3, radial speed +-0.25 v_c, uniform magnitude."""
import sys, csv, json, math, os
import numpy as np
GM = 2.959122082855911e-4; EPS = math.radians(23.43928)
scen, out, N, mlo, mhi, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
rng = np.random.default_rng(seed)
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
mjd = np.array([float(r["mjd_mid"]) for r in rows])
k = int(np.argmin(np.abs(mjd - 0.5 * (mjd.min() + mjd.max()))))
O = np.array([float(rows[k][c]) for c in ("obs_x", "obs_y", "obs_z")])
ra0 = math.radians(float(np.median([float(r["ra_center"]) for r in rows]))); dec0 = math.radians(float(np.median([float(r["dec_center"]) for r in rows])))
kecl = np.array([0.0, -math.sin(EPS), math.cos(EPS)])
bodies = []
for b in range(N):
    dra = rng.uniform(-1.0, 1.0); ddec = rng.uniform(-0.95, 0.95)
    ra = ra0 + math.radians(dra) / math.cos(dec0); dec = dec0 + math.radians(ddec)
    u = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    r = float(np.clip(rng.normal(41, 6), 30, 60)) if rng.random() < 0.85 else rng.uniform(30, 80)
    bq = O @ u; rho = -bq + math.sqrt(bq * bq - O @ O + r * r)
    P = O + rho * u; ph = P / r
    e1 = np.cross(kecl, ph); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ph, e1)
    beta = math.radians(rng.normal(0, 4) if rng.random() < 0.4 else rng.normal(0, 16))
    w = rng.uniform(0.75, 1.3); vc = math.sqrt(GM / r)
    f = rng.uniform(-0.25, 0.25)
    V = vc * math.sqrt(w) * (math.cos(beta) * e1 + math.sin(beta) * e2) + f * vc * ph
    if V @ V / 2 - GM / r >= 0:
        continue
    bodies.append({"epoch_mjd": float(mjd[k]), "state": [float(x) for x in np.concatenate([P, V])], "mag": float(rng.uniform(mlo, mhi)),
                   "r": r, "w": w, "beta": math.degrees(beta), "f": f})
json.dump(bodies, open(out, "w"))
print(len(bodies), "bodies")
