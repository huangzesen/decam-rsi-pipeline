import sys, os, csv, math, numpy as np, torch, pickle
sys.path.insert(0, "$RSI_SANDBOX/r2/gen007/workspace/submission"); import finder
from astropy.io import fits
from scipy.spatial import cKDTree
scen = sys.argv[1]; ei = [int(x) for x in sys.argv[2].split(",")]; out = sys.argv[3]
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
rac = np.radians([float(r["ra_center"]) for r in rows]); dec = np.radians([float(r["dec_center"]) for r in rows])
cv = np.stack([np.cos(dec) * np.cos(rac), np.cos(dec) * np.sin(rac), np.sin(dec)], 1).mean(0); cv /= np.linalg.norm(cv)
ra0 = math.atan2(cv[1], cv[0]); ex = np.array([-math.sin(ra0), math.cos(ra0), 0.0]); R = np.stack([ex, np.cross(cv, ex), cv])
res = {}
for e in ei:
    r = rows[e]; ip = os.path.join(scen, r["image"]); mp = os.path.join(scen, r["mask"])
    hd = list(range(1, 61))[10:22]
    cats = []; sts = []
    for c in finder.load_chunk((ip, mp, hd, 3.0, 1.2, R, None)):
        cat, piece = finder.gpu_ccd(torch, "mps", c, 3.0, stamp_thr=4.5)
        if cat is None: continue
        sel, st = c["_stamps"]
        cats.append(cat[sel]); sts.append(st)
    res[e] = (np.concatenate(cats), np.concatenate(sts))
    print(e, r["band"], r["exptime"], len(res[e][0]))
pickle.dump(res, open(out, "wb"))
