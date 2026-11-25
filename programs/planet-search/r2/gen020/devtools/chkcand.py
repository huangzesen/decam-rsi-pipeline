import sys, os, csv, math, json, pickle, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r2/gen012/workspace/submission")
import finder as F
scen = sys.argv[1]; out = sys.argv[2]; cache = sys.argv[3]; ranks = [int(x) for x in sys.argv[4].split(",")]
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
mjd = np.array([float(r["mjd_mid"]) for r in rows])
rac = np.array([math.radians(float(r["ra_center"])) for r in rows]); dec = np.array([math.radians(float(r["dec_center"])) for r in rows])
cvec = np.stack([np.cos(dec) * np.cos(rac), np.cos(dec) * np.sin(rac), np.sin(dec)], 1).mean(0); cvec /= np.linalg.norm(cvec)
ra0 = math.atan2(cvec[1], cvec[0]); ez = cvec; ex = np.array([-math.sin(ra0), math.cos(ra0), 0.0]); ey = np.cross(ez, ex)
R = np.stack([ex, ey, ez]); tref = 0.5 * (mjd.min() + mjd.max())
O = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows]) @ R.T
dets, maps = pickle.load(open(cache, "rb"))
c = sorted(json.load(open(out))["candidates"], key=lambda c: -c["p"])
for rk in ranks:
    st = np.array(c[rk]["state"])
    xy = F.predict_xy(R @ st[:3], R @ st[3:], c[rk]["epoch_mjd"] - tref, mjd - tref, O)
    print("rank", rk, "score", c[rk]["score"], "r", np.linalg.norm(st[:3]).round(1))
    for e in sorted(dets):
        d = dets[e]; dd = np.hypot(d[:, 0] - xy[e, 0], d[:, 1] - xy[e, 1]); j = np.argmin(dd)
        y0, x0, arr = maps[e]; iy = int(round(xy[e, 1] / F.PS)) - y0; ix = int(round(xy[e, 0] / F.PS)) - x0
        mv = arr[iy - 2:iy + 3, ix - 2:ix + 3].max() / 10 if 2 <= iy < arr.shape[0] - 2 and 2 <= ix < arr.shape[1] - 2 else None
        print("   exp", e, "xy %.1f %.1f" % tuple(xy[e]), "nearest det %.2f\" snr %.1f" % (dd[j], d[j, 2]), "map max5x5", mv)
