"""For each B/C body: nearest candidate in a candidates json (n nights within 2/10/20", rank, kind)."""
import json, csv, sys, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen104/workspace/submission/linker")
import orbit
ARC = np.pi / 180 / 3600
S = "$RSI_SANDBOX/r1/gen104/tmp/inj7c"
T = json.load(open(S + "/truth.json")); rows = list(csv.DictReader(open(S + "/exposures.csv")))
t = np.array([float(r["mjd_mid"]) for r in rows]); obs = np.array([[float(r[k]) for k in ("obs_x", "obs_y", "obs_z")] for r in rows])
night = np.floor(t + 0.3).astype(int)
C = json.load(open(sys.argv[1]))["candidates"]
C = sorted(C, key=lambda c: -c["p"])
ep = np.array([c["epoch_mjd"] for c in C])
U = []
for c in C:
    U.append(orbit.predict_dirs(c["state"], c["epoch_mjd"], t, obs))
U = np.array(U)
which = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(15, 40)
for j in which:
    b = T["bodies"][j]; ex = np.array(b["exps"])
    u = orbit.predict_dirs(b["state"], b["epoch_mjd"], t, obs)
    sep = np.arccos(np.clip((U[:, ex] * u[ex][None]).sum(-1), -1, 1)) / ARC
    med = np.median(sep, 1)
    k = int(np.argmin(med))
    nn = lambda tol, kk: len(set(night[ex[sep[kk] < tol]].tolist()))
    best = {tol: max(range(len(C)), key=lambda kk: (nn(tol, kk), -med[kk])) for tol in (2, 10, 20)}
    print(j, b["cls"], "closest med %.1f\" rank %d kind %s n2/n10/n20 %d/%d/%d" % (med[k], k, C[k].get("kind"), nn(2, k), nn(10, k), nn(20, k)),
          "| max nights<2\": %d (rank %d) <10\": %d <20\": %d" % (nn(2, best[2]), best[2], nn(10, best[10]), nn(20, best[20])))
