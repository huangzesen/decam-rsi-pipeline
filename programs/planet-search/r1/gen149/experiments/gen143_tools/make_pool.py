"""Sample normalised 21x21 background stamps (real DECam data) from dev scenarios -> npz (stamps, fwhm, scen id)."""
import sys, os, csv, math, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen143/workspace/submission"); sys.path.insert(0, "$RSI_SANDBOX/r1/gen143/workspace/submission/stackd")
import vetcnn, fastfits
from astropy.io import fits
D = "$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev"
scen = sys.argv[1]; out = sys.argv[2]; nper = int(sys.argv[3]) if len(sys.argv) > 3 else 400
rng = np.random.default_rng(abs(hash(scen)) % 2**32)
rows = list(csv.DictReader(open(os.path.join(D, scen, "exposures.csv"))))
ST, FW, EX = [], [], []
def rd(h):
    try:
        a = fastfits.read_tiled(h)
        if a is not None: return a
    except Exception: pass
    return None
for ie, r in enumerate(rows):
    hi = fits.open(os.path.join(D, scen, r["image"])); hw = fits.open(os.path.join(D, scen, r["weight"])); hm = fits.open(os.path.join(D, scen, r["mask"]))
    hb = [fits.open(os.path.join(D, scen, r[k]), disable_image_compression=True) for k in ("image", "weight", "mask")]
    ks = rng.choice(np.arange(1, len(hi)), size=min(10, len(hi) - 1), replace=False)
    for k in ks:
        try:
            im = rd(hb[0][k]); wt = rd(hb[1][k]); mk = rd(hb[2][k])
            if im is None: im = np.asarray(hi[k].data, np.float32)
            if wt is None: wt = np.asarray(hw[k].data, np.float32)
            if mk is None: mk = np.asarray(hm[k].data, np.float32)
            fw = float(hi[k].header.get("FWHM", 4.0) or 4.0)
            if not (0.5 < fw < 20): fw = 4.0
        except Exception as ex:
            print("skip", ex); continue
        ny, nx = im.shape
        RC = max(13, int(math.ceil(2.6 * fw)))
        n = 0
        for _ in range(nper * 2):
            x0 = int(rng.integers(RC, nx - RC - 1)); y0 = int(rng.integers(RC, ny - RC - 1))
            z = vetcnn.norm_stamp(im[y0-RC:y0+RC+1, x0-RC:x0+RC+1], wt[y0-RC:y0+RC+1, x0-RC:x0+RC+1], mk[y0-RC:y0+RC+1, x0-RC:x0+RC+1], fw, RC, RC, half=10)
            if z is None: continue
            ST.append(z.astype(np.float16)); FW.append(fw); EX.append(ie); n += 1
            if n >= nper: break
    print(scen, ie, len(ST), flush=True)
np.savez(out, st=np.array(ST), fw=np.array(FW, np.float32), ex=np.array(EX))
