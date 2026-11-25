"""score_inject.py SCEN BODIES.json NAME=CANDS.json [NAME=CANDS.json ...] : injection recovery per program.

A body is recovered by a program when one of the program's 25 highest-p candidates puts it within 2 arcsec of the body
on >= 3 different nights (two-body motion and light time from both states, as the grader does). Bodies counted: those
inside 1.0 deg of the field centre on >= 3 nights. Also reports recovery anywhere in the (<= 200) output."""
import sys, csv, json, math, os
import numpy as np
GM = 2.959122082855911e-4; C_AU = 173.1446326846693; ASEC = 206264.806


def propagate(P0, V0, dt, nstep=8):
    K = len(dt); P = np.repeat(P0[None], K, 0).astype(float); V = np.repeat(V0[None], K, 0).astype(float)
    h = (np.asarray(dt, float) / nstep)[:, None]
    acc = lambda p: -GM * p / (np.sum(p * p, 1, keepdims=True) ** 1.5)
    for _ in range(nstep):
        k1p, k1v = V, acc(P); k2p, k2v = V + 0.5 * h * k1v, acc(P + 0.5 * h * k1p)
        k3p, k3v = V + 0.5 * h * k2v, acc(P + 0.5 * h * k2p); k4p, k4v = V + h * k3v, acc(P + h * k3p)
        P = P + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p); V = V + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return P


def observe(st, t0, t, O):
    st = np.asarray(st, float); lt = np.linalg.norm(st[:3] - O, axis=1) / C_AU
    for _ in range(3):
        D = propagate(st[:3], st[3:], t - lt - t0) - O; lt = np.linalg.norm(D, axis=1) / C_AU
    return D / np.linalg.norm(D, axis=1, keepdims=True)


scen, bfile = sys.argv[1], sys.argv[2]
progs = [x.split("=", 1) for x in sys.argv[3:]]
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
mjd = np.array([float(r["mjd_mid"]) for r in rows]); night = np.floor(mjd + 0.3).astype(int)
O = np.array([[float(r[c]) for c in ("obs_x", "obs_y", "obs_z")] for r in rows])
cen = np.array([[math.cos(math.radians(float(r["dec_center"]))) * math.cos(math.radians(float(r["ra_center"]))),
                 math.cos(math.radians(float(r["dec_center"]))) * math.sin(math.radians(float(r["ra_center"]))),
                 math.sin(math.radians(float(r["dec_center"])))] for r in rows])
bodies = json.load(open(bfile))
B = [observe(b["state"], b["epoch_mjd"], mjd, O) for b in bodies]
inside = [len(set(night[np.sum(u * cen, 1) > math.cos(math.radians(1.0))].tolist())) >= 3 for u in B]
res = {}
for name, cf in progs:
    c = json.load(open(cf))["candidates"] if os.path.exists(cf) else []
    c = sorted(c, key=lambda x: -x["p"])
    pred = [observe(x["state"], x["epoch_mjd"], mjd, O) for x in c]
    rec25, rec200, rank = [], [], []
    for u in B:
        best = None
        for k, q in enumerate(pred):
            sep = np.arccos(np.clip(np.sum(u * q, 1), -1, 1)) * ASEC
            nn = len(set(night[sep <= 2.0].tolist()))
            if nn >= 3:
                best = k; break
        rank.append(best)
        rec25.append(best is not None and best < 25); rec200.append(best is not None)
    res[name] = dict(rec25=rec25, rec200=rec200, rank=rank, n=len(c))
mags = np.array([b["mag"] for b in bodies])
edges = np.arange(np.floor(mags.min() * 2) / 2, mags.max() + 0.5, 0.5)
print(f"{os.path.basename(os.path.normpath(scen))}: {len(bodies)} bodies, {sum(inside)} inside the field on >= 3 nights")
hdr = "mag bin      n  " + "  ".join(f"{n:>14s}" for n, _ in progs)
print(hdr)
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = [i for i in range(len(bodies)) if inside[i] and lo <= mags[i] < hi]
    if not sel:
        continue
    print(f"{lo:5.1f}-{hi:4.1f} {len(sel):4d}  " + "  ".join(
        f"{sum(res[n]['rec25'][i] for i in sel):5d} ({sum(res[n]['rec200'][i] for i in sel):3d} any)" for n, _ in progs))
sel = [i for i in range(len(bodies)) if inside[i]]
print(f"all       {len(sel):4d}  " + "  ".join(f"{sum(res[n]['rec25'][i] for i in sel):5d} ({sum(res[n]['rec200'][i] for i in sel):3d} any)" for n, _ in progs))
json.dump({"bodies": [dict(mag=float(mags[i]), inside=bool(inside[i]), **{n: dict(rank=res[n]["rank"][i]) for n, _ in progs}) for i in range(len(bodies))]},
          open(os.path.splitext(bfile)[0] + ".score.json", "w"), indent=0)
