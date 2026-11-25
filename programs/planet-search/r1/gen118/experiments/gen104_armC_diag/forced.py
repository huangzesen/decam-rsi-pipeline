"""Forced matched-filter S/N at injected positions (truth used for measurement only)."""
import json, csv, sys, os, math, collections
import numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen104/tmp/armC/sub/stackd")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch
torch.set_num_threads(2)
import mapper
mapper._TORCH = (torch, "cpu")
from astropy.io import fits
S = "$RSI_SANDBOX/r1/gen104/tmp/inj7c"
truth = json.load(open(S + "/truth.json"))
rows = list(csv.DictReader(open(S + "/exposures.csv")))
inj = truth["injections"]
byexp = collections.defaultdict(list)
for r in inj:
    byexp[r["exp"]].append(r)
z = {}  # (body, exp) -> z
for ei, lst in sorted(byexp.items()):
    hi = fits.open(os.path.join(S, rows[ei]["image"]), memmap=False)
    hm = fits.open(os.path.join(S, rows[ei]["mask"]), memmap=False)
    names = {hi[k].header.get("EXTNAME"): k for k in range(1, len(hi))}
    mn = {hm[k].header.get("EXTNAME"): k for k in range(1, len(hm))}
    byccd = collections.defaultdict(list)
    for r in lst:
        byccd[r["ccd"]].append(r)
    for ccd, rl in byccd.items():
        k = names[ccd]
        img = hi[k].data.astype(np.float32)
        msk = hm[mn.get(ccd, k)].data
        fwhm = float(hi[k].header.get("FWHM") or 4.0)
        pm = mapper.snr_pooled(img, msk, fwhm, pool=1)
        if pm is None:
            continue
        pm = pm.astype(np.float32)
        for r in rl:
            ix, iy = int(round(r["x"])), int(round(r["y"]))
            v = pm[iy, ix]
            vm = np.nanmax(pm[iy - 1:iy + 2, ix - 1:ix + 2])
            z[(r["body"], ei)] = (float(v), float(vm))
    print("exp", ei, "done", flush=True)
t = np.array([float(r["mjd_mid"]) for r in rows])
night = np.floor(t + 0.3).astype(int)
out = []
for j, b in enumerate(truth["bodies"]):
    ex = [e for e in range(len(rows)) if (j, e) in z]
    zz = {e: z[(j, e)][0] for e in ex if np.isfinite(z[(j, e)][0])}
    nights = sorted({night[e] for e in zz})
    nz = {n: sum(zz[e] for e in zz if night[e] == n) / math.sqrt(sum(1 for e in zz if night[e] == n)) for n in nights}
    allz = sum(zz.values()) / math.sqrt(max(len(zz), 1))
    best3 = max([sum(nz[n] for n in nights[i:i + 3]) / math.sqrt(3) for i in range(len(nights) - 2)] or [float('nan')])
    rec = dict(j=j, cls=b["cls"], nom=b["snr"], exp_z=[round(zz[e], 1) for e in sorted(zz)],
               night_z=[round(nz[n], 1) for n in nights], n_exp5=len({night[e] for e in zz if zz[e] >= 5}),
               n_night5=sum(1 for n in nights if nz[n] >= 5), n_night46=sum(1 for n in nights if nz[n] >= 4.6),
               all_z=round(allz, 2), best3=round(best3, 2))
    out.append(rec)
    print(rec)
json.dump(out, open("$RSI_SANDBOX/r1/gen104/tmp/armC/diag/forced.json", "w"))
