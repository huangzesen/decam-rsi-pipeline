"""Gen 8c dev test: inject a faint moving body into real S/N maps relabelled as one long night; check nightstack
recovers it.  usage: python test_nightstack.py DET.npy SCENARIO_DIR [amp]"""
import sys, os, math, csv, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from orbit import ARC, radec_to_unit, tangent_frame
from detect import gnomonic, poly2
from forced import Forced
import nightstack, hough, link

res = np.load(sys.argv[1], allow_pickle=True).item()
rows = list(csv.DictReader(open(os.path.join(sys.argv[2], "exposures.csv"))))
meta = json.load(open(os.path.join(sys.argv[2], "meta.json")))
amp = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
B = link.build(rows, res)
et, eobs = B["et"].copy(), B["eobs"]
done = B["done"]
mexp = [e for e in done if res[e].get("maps")][:12]
ne = len(mexp)
t0 = et[mexp[0]]
for k, e in enumerate(mexp):
    et[e] = t0 + k * 0.25 / 24
enight = np.floor(B["et"] - 0.5).astype(int)
enight[mexp] = 777
vobs = link.observer_velocity(B["et"], eobs)
F = tangent_frame(math.radians(meta["ra"]), math.radians(meta["dec"]))
eref = mexp[ne // 2]
vof = F @ vobs[mexp].mean(0)
V, dv = hough.rate_grid(vof, 2.75)
rng = np.random.default_rng(1)
truth = []
for b in range(20):
    v = V[rng.integers(len(V))] + rng.uniform(-0.5, 0.5, 2) * dv
    p0 = rng.uniform(-0.6, 0.6, 2) * 3600     # arcsec from centre
    truth.append((p0, v))
    for e in mexp:
        dt = (et[e] - et[eref]) * 24
        p = (p0 + v * dt) * ARC
        f = np.array([p[0], p[1], 1.0]); uu = f @ F; uu /= np.linalg.norm(uu)
        ra, de = math.atan2(uu[1], uu[0]), math.asin(uu[2])
        for g, m in res[e]["maps"]:
            xi, eta = gnomonic(np.array([ra]), np.array([de]), g["rc"], g["dc"])
            A = poly2(xi, eta)
            x = float(A @ g["cx"]) / 2; y = float(A @ g["cy"]) / 2
            if 3 <= x < m.shape[1] - 4 and 3 <= y < m.shape[0] - 4:
                yy, xx = np.mgrid[-3:4, -3:4]
                ix, iy = int(x), int(y)
                prof = np.exp(-((xx + ix - x) ** 2 + (yy + iy - y) ** 2) / (2 * 1.4 ** 2))
                sl = m[iy - 3:iy + 4, ix - 3:ix + 4]
                m[iy - 3:iy + 4, ix - 3:ix + 4] = (sl.astype(np.float32) + amp * prof).astype(m.dtype)
FPt = Forced(res)
for p0, v in truth[:3]:
    vals = []
    for e in mexp:
        dt = (et[e] - et[eref]) * 24
        p = (p0 + v * dt) * ARC
        f = np.array([p[0], p[1], 1.0]); uu = f @ F; uu /= np.linalg.norm(uu)
        vals.append(float(FPt.sample(e, uu[None])[0]))
    print("forced at truth", np.round(vals, 1))
if os.environ.get("ONLYF"):
    sys.exit()
D, E = B["D"], B["E"]
u0 = radec_to_unit(np.radians(D[:, 0]), np.radians(D[:, 1]))
VR, VE = nightstack.virtual_dets(Forced(res), F, et, enight, done, vobs, u0, enight[E], 120.0, ckey=1)
uv = radec_to_unit(np.radians(VR[:, 0]), np.radians(VR[:, 1])) @ F.T
xy = uv[:, :2] / uv[:, 2:3] / ARC
nfound = 0
for p0, v in truth:
    d = np.hypot(*(xy - p0).T)
    k = np.argmin(d)
    ok = d[k] < 2.0
    nfound += ok
    print(f"body at {p0.round(0)} rate {v.round(2)}: nearest virtual {d[k]:.2f}\" S/N {VR[k, 2]:.1f}")
print(f"recovered {nfound}/{len(truth)}; {len(VR)} virtual detections total")
