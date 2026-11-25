import sys, os, csv, numpy as np, warnings
warnings.filterwarnings("ignore")
from astropy.io import fits
from astropy.wcs import WCS
import scipy.ndimage as ndi
scen = sys.argv[1]; ccds = [int(x) for x in sys.argv[2].split(",")]
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
def stars(path, h):
    with fits.open(path) as f:
        d = f[h].data.astype(np.float32); hd = f[h].header
    d = d - np.median(d); sig = 1.4826 * np.median(np.abs(d))
    sm = ndi.uniform_filter(d, 7) * 49          # 7x7 aperture flux
    mx = ndi.maximum_filter(sm, 15)
    y, x = np.nonzero((sm == mx) & (sm > 60 * sig * 7) & (d < 30000))
    w = WCS(hd)
    ra, dec = w.all_pix2world(x, y, 0)
    return ra, dec, sm[y, x], hd.get("EXTNAME")
res = {}
for r in rows:
    res[r["exp_id"]] = [stars(os.path.join(scen, r["image"]), h) for h in ccds]
ref = {}
for r in rows:
    b = r["band"]
    if b not in ref: ref[b] = r
    r0 = ref[b]; offs = []
    for (ra, dec, f, _), (ra0, dec0, f0, _) in zip(res[r["exp_id"]], res[r0["exp_id"]]):
        for i in range(len(ra)):
            dd = np.hypot((ra0 - ra[i]) * np.cos(np.radians(dec[i])), dec0 - dec[i]) * 3600
            j = np.argmin(dd) if len(dd) else None
            if j is not None and dd[j] < 1.0: offs.append(2.5 * np.log10(f[i] / f0[j]))
    mz = lambda q: float(q["magzero"]) if q["magzero"] else float("nan")
    print(r["exp_id"], b, r["exptime"], "magzero", r["magzero"], "| header diff vs ref %.2f" % (mz(r) - mz(r0)), "| star flux diff %.2f (n=%d)" % (np.median(offs) if offs else np.nan, len(offs)), "seeing", r["seeing_arcsec"])
