"""gen_train.py SCEN OUT.pkl : stamps of detections >= 4.5 sigma with self-made labels (dev only).
label 1 = static (a >= 3 sigma detection within 0.6" on another night), 0 = nothing within 2" in any other exposure
although >= 2 other exposures cover the place, -1 = neither."""
import sys, os, csv, math, numpy as np, torch, pickle
sys.path.insert(0, "$RSI_SANDBOX/r2/gen007/workspace/submission"); import finder
from scipy.spatial import cKDTree
scen = sys.argv[1]; out = sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
mjd = np.array([float(r["mjd_mid"]) for r in rows]); night = np.round(mjd - 0.2).astype(int)
rac = np.radians([float(r["ra_center"]) for r in rows]); dec = np.radians([float(r["dec_center"]) for r in rows])
cv = np.stack([np.cos(dec) * np.cos(rac), np.cos(dec) * np.sin(rac), np.sin(dec)], 1).mean(0); cv /= np.linalg.norm(cv)
ra0 = math.atan2(cv[1], cv[0]); ex = np.array([-math.sin(ra0), math.cos(ra0), 0.0]); R = np.stack([ex, np.cross(cv, ex), cv])
hd = list(range(3, 61, 5))
full = {}; hi = {}
for e, r in enumerate(rows):
    ip = os.path.join(scen, r["image"]); mp = os.path.join(scen, r["mask"])
    cats = []; hc = []; sts = []
    for c in finder.load_chunk((ip, mp, hd, 3.0, 1.2, R, None)):
        cat, piece = finder.gpu_ccd(torch, "mps", c, 3.0, stamp_thr=4.5)
        if cat is None: continue
        sel, st = c["_stamps"]
        cats.append(cat); hc.append(cat[sel]); sts.append(st)
    full[e] = np.concatenate(cats); hi[e] = (np.concatenate(hc), np.concatenate(sts))
    print(e, r["band"], r["exptime"], len(full[e]), len(hi[e][0]), flush=True)
trees = {e: cKDTree(full[e][:, :2]) for e in full}
S = []; L = []; SN = []; EX = []
rng = np.random.default_rng(0)
for e in full:
    cat, st = hi[e]; n = len(cat)
    stat = np.zeros(n, bool); near = np.zeros(n, bool); cov = np.zeros(n, int)
    for o in full:
        if o == e: continue
        d, _ = trees[o].query(cat[:, :2], distance_upper_bound=20.0)
        cov += np.isfinite(d); near |= d < 2.0
        if night[o] != night[e]: stat |= d < 0.6
    lab = np.where(stat, 1, np.where(~near & (cov >= 2) & (cat[:, 2] >= 6.0), 0, -1))
    lab[(cat[:, 2] > 60)] = -1
    idx = np.concatenate([rng.permutation(np.nonzero(lab == 1)[0])[:4000], np.nonzero(lab == 0)[0]])
    S.append(st[idx]); L.append(lab[idx]); SN.append(cat[idx, 2]); EX.append(np.full(len(idx), e))
    print(e, "pos", (lab == 1).sum(), "neg", (lab == 0).sum(), flush=True)
pickle.dump({"st": np.concatenate(S), "lab": np.concatenate(L), "snr": np.concatenate(SN), "exp": np.concatenate(EX)}, open(out, "wb"))
