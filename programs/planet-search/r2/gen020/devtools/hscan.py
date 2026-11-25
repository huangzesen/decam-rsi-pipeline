import sys, os, csv, glob, numpy as np
from astropy.io import fits
D = sys.argv[1]
for scen in sorted(glob.glob(D + "/s*")):
    rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
    for r in rows:
        h = fits.open(os.path.join(scen, r["image"]))
        p = h[0].header
        fw = []; ext = []; skys = []; nbad = 0
        for k in range(1, len(h)):
            x = h[k].header
            f = x.get("FWHM"); 
            try: fw.append(float(f))
            except Exception: nbad += 1
            ext.append((x.get("EXTNAME"), x.get("CCDNUM")))
        d = dict(ext)
        print(os.path.basename(scen), r["exp_id"], r["band"], "nhdu", len(h) - 1, "WCSCAL", p.get("WCSCAL"), "PHOTCAL?", p.get("PHOTFLAG"), "MAGZERO", p.get("MAGZERO"), "csv", r["magzero"],
              "FWHM med %.2f px = %.2f\" csv %s" % (np.median(fw) if fw else -1, (np.median(fw) if fw else -1) * 0.263, r["seeing_arcsec"]), "nofw", nbad, "S29->", d.get("S29"), "N31->", d.get("N31"),
              "PROCTYPE", p.get("PROCTYPE"), "PLVER", p.get("PLVER"), "PROPID", p.get("PROPID"), "OBJECT", str(p.get("OBJECT"))[:25])
