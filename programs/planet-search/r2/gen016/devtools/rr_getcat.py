import sys, os, csv, math, pickle, time, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r2/gen012/workspace/submission")
import finder as F
import torch
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
def main():
    scen = sys.argv[1]; exps = [int(x) for x in sys.argv[2].split(",")]; out = sys.argv[3]
    rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
    rac = np.array([math.radians(float(r["ra_center"])) for r in rows]); dec = np.array([math.radians(float(r["dec_center"])) for r in rows])
    cvec = np.stack([np.cos(dec) * np.cos(rac), np.cos(dec) * np.sin(rac), np.sin(dec)], 1).mean(0); cvec /= np.linalg.norm(cvec)
    ra0 = math.atan2(cvec[1], cvec[0]); ez = cvec; ex = np.array([-math.sin(ra0), math.cos(ra0), 0.0]); ey = np.cross(ez, ex)
    R = np.stack([ex, ey, ez])
    dev = "mps"
    res = {}
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i in exps:
            t = time.time()
            r = rows[i]; ip = os.path.join(scen, r["image"]); mp = os.path.join(scen, r["mask"])
            with fits.open(ip) as h:
                hd = [k for k in range(len(h)) if h[k].header.get("ZNAXIS", 0) == 2 or h[k].header.get("NAXIS", 0) == 2]
            tasks = [(ip, mp, hd[c::8], 3.0, float(r["seeing_arcsec"]), R, None) for c in range(8) if hd[c::8]]
            L = []
            for chunk in pool.map(F.load_chunk, tasks):
                for c in chunk:
                    cat, piece = F.gpu_ccd(torch, dev, c, 3.0, cls=None)
                    ny, nx = c["d"].shape
                    cc = {k: v for k, v in c.items() if k != "d"}; cc["shape"] = (ny, nx)
                    L.append((cc, cat))
            res[i] = L
            print(i, len(L), time.time() - t, flush=True)
    pickle.dump(res, open(out, "wb"))
if __name__ == "__main__":
    main()
