import sys, os, csv, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen104/tmp/armC/sub/stackd")
import torch; torch.set_num_threads(2)
import mapper; mapper._TORCH = (torch, "cpu")
from astropy.io import fits
S = "$RSI_SANDBOX/r1/gen104/tmp/inj7c"
rows = list(csv.DictReader(open(S + "/exposures.csv")))
for ei in [0, 1, 2, 3]:
    hi = fits.open(os.path.join(S, rows[ei]["image"])); hm = fits.open(os.path.join(S, rows[ei]["mask"]))
    out = []
    for k in (5, 20, 35, 50):
        img = hi[k].data.astype(np.float32); msk = hm[k].data
        pm = mapper.snr_pooled(img, msk, float(hi[k].header.get("FWHM") or 4), pool=4)
        if pm is None: out.append(None); continue
        pm = pm.astype(np.float32); f = np.isfinite(pm)
        out.append((round(float((pm[f] >= 5).mean()), 4), round(float((pm[f] >= 4).mean()), 4), round(float(np.median(pm[f])), 2)))
    print(ei, rows[ei]["magzero"] or "EMPTY", rows[ei]["band"], out, flush=True)
