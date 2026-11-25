import os, csv, numpy as np, warnings
warnings.simplefilter("ignore")
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from scipy.spatial import cKDTree
S = "$RSI_SANDBOX/r1/gen104/tmp/inj7c"
rows = list(csv.DictReader(open(S + "/exposures.csv")))
def stars(ei, ccd):
    h = fits.open(os.path.join(S, rows[ei]["image"]))
    for k in range(1, len(h)):
        if h[k].header.get("EXTNAME") == ccd:
            img = h[k].data.astype(np.float32); w = WCS(h[k].header)
            bg = np.median(img); sd = 1.4826 * np.median(np.abs(img - bg))
            sm = ndimage.gaussian_filter(img - bg, 1.5)
            mx = ndimage.maximum_filter(sm, 9)
            yy, xx = np.nonzero((sm == mx) & (sm > 30 * sd / 3))
            o = np.argsort(-sm[yy, xx])[:300]
            ra, de = w.all_pix2world(xx[o], yy[o], 0)
            return np.c_[ra * np.cos(np.radians(de)), de] * 3600
for ccd in ("N4", "S10", "N20"):
    ref = stars(3, ccd)
    for ei in (0, 1, 2, 5, 8):
        p = stars(ei, ccd)
        d, _ = cKDTree(ref).query(p)
        print(ccd, ei, rows[ei]["magzero"] or "EMPTY", "n", len(p), "median NN sep arcsec %.2f, frac<1\" %.2f" % (np.median(d), (d < 1).mean()), flush=True)
