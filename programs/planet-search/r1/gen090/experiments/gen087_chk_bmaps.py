import sys, json, csv, math, os
import numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen087/tmp/armB/stackd")
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import mapper
mapper._TORCH = (torch, "cpu")
from astropy.io import fits
S = "$RSI_SANDBOX/r1/gen087/tmp/inj7/"
tr = json.load(open(S + "truth.json")); rows = list(csv.DictReader(open(S + "exposures.csv")))
cls = [b["cls"] for b in tr["bodies"]]
inj = [x for x in tr["injections"] if cls[x["body"]] == sys.argv[1]]
rng = np.random.default_rng(0); sel = [inj[i] for i in rng.choice(len(inj), int(sys.argv[2]), replace=False)]
for x in sel:
  try:
      r = rows[x["exp"]]
      hi = fits.open(S + r["image"], memmap=False)
      hm = fits.open(S + r["mask"], memmap=False); hw = fits.open(S + r["weight"], memmap=False)
      img = hi[x["ccd"]].data.astype(np.float32); msk = hm[x["ccd"]].data; wt = hw[x["ccd"]].data.astype(np.float32)
      fwhm = float(hi[x["ccd"]].header.get("FWHM") or 4.0)
      pm = mapper.snr_pooled(img, msk, fwhm)
      ix, iy = int(round(x["x"])), int(round(x["y"]))
      pv = float(np.nanmax(pm[max(0, iy // 4 - 1):iy // 4 + 2, max(0, ix // 4 - 1):ix // 4 + 2]))
      p1 = float(pm[iy // 4, ix // 4])
      s = x["fwhm"] / 2.355; k = int(3 * s) + 1
      yy, xx = np.mgrid[-k:k + 1, -k:k + 1]; P = np.exp(-(xx ** 2 + yy ** 2) / (2 * s * s)); P /= P.sum()
      cut = img[iy - k:iy + k + 1, ix - k:ix + k + 1].astype(float); w = wt[iy - k:iy + k + 1, ix - k:ix + k + 1].astype(float)
      bg = np.median(img[max(0, iy - 60):iy + 60, max(0, ix - 60):ix + 60])
      est = ((cut - bg) * P * w).sum() / math.sqrt((P * P * w).sum())
      sd_img = 1.4826 * np.median(np.abs(img[::7, ::7] - np.median(img[::7, ::7])))
      print(f"{sys.argv[1]} body {x['body']} exp {x['exp']} truth {x['snr']:.2f} fwhm {x['fwhm']:.2f}/{fwhm:.2f} "
            f"msk {int(msk[iy, ix])} pm {p1:.2f} pm3x3 {pv:.2f} wopt {est:.2f} 1/sqrt(wmed) {1/math.sqrt(x['wmed']):.2f} pixsd {sd_img:.2f}", flush=True)

  except Exception as e: print('skip', e)
