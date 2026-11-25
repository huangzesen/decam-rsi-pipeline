"""Truth for each scenario: the known TNOs in its exposures, where they are, and how hard they are to find.

For every scenario in ROOT/private/scenarios/ (all exposures, held-back ones included):
  1. candidate bodies: every TNO of the JPL SBDB snapshot whose two-body position (osculating elements, as in
     linker/footprint) falls within --radius deg of an exposure centre;
  2. JPL Horizons astrometric RA/Dec (ICRF), V, r, delta and RA/Dec 3-sigma at every exposure mid-time, observer W84
     (DECam), one query per body (cached in ROOT/private/horizons/);
  3. the CCD and pixel under each position (TPV WCS of the InstCal CCD headers) and forced aperture photometry there:
     radius 0.7 FWHM, sky = median of a 3-5 FWHM annulus, variance from the weight map (inverse variance), flagged when
     a mask pixel or zero weight falls in the aperture;
  4. a tier from the visible (not held-back) exposures, for bodies whose positions are precise (3-sigma <= --max-sigma
     arcsec on those exposures): 1 = >= 3 nights with a single exposure at S/N >= 5; 2 = >= 3 nights whose
     inverse-variance night stack reaches S/N >= 5; 3 = the stack of all visible exposures reaches S/N >= --snr3;
     0 = not findable in these data. Imprecise bodies get tier None (neither counted nor penalised).
Output ROOT/private/truth/<sid>.json. A finder never sees this tree.

Usage: ~/anaconda3/bin/python -I make_truth.py ROOT SBDB.json [--sid scen00 ...]
"""
import argparse, glob, json, os, sys, time, urllib.parse, urllib.request, warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "linker"))
from orbits import elements_to_rv, ecl_to_eq, GM
warnings.simplefilter("ignore", FITSFixedWarning)
warnings.simplefilter("ignore", RuntimeWarning)
PIX = 0.263


def horizons(pdes, jds, cache):
    f = os.path.join(cache, pdes.replace(" ", "_") + "_" + str(abs(hash(tuple(round(j, 6) for j in jds))) % 10**12) + ".json")
    if os.path.exists(f):
        return json.load(open(f))
    cmd = f"{pdes};" if pdes.isdigit() else f"DES={pdes};"
    out = []
    for k in range(0, len(jds), 400):
        q = {"format": "json", "COMMAND": f"'{cmd}'", "OBJ_DATA": "NO", "MAKE_EPHEM": "YES", "EPHEM_TYPE": "OBSERVER",
             "CENTER": "'W84'", "TLIST": " ".join(f"{j:.6f}" for j in jds[k:k + 400]), "TLIST_TYPE": "JD",
             "TIME_TYPE": "UT", "QUANTITIES": "'1,9,19,20,36'", "ANG_FORMAT": "DEG", "EXTRA_PREC": "YES", "CSV_FORMAT": "YES"}
        url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q)
        for t in range(5):
            try:
                res = json.loads(urllib.request.urlopen(url, timeout=180).read()).get("result", "")
                break
            except Exception as e:
                print("  horizons retry", pdes, t, e, flush=True); time.sleep(5 * (t + 1))
        else:
            return None
        i, j = res.find("$$SOE"), res.find("$$EOE")
        if i < 0:
            return None
        for line in res[i + 5:j].strip().splitlines():
            c = [x.strip() for x in line.split(",")]
            fl = lambda s: float(s) if s not in ("", "n.a.") else None
            out.append(dict(ra=fl(c[3]), dec=fl(c[4]), V=fl(c[5]), r=fl(c[7]), delta=fl(c[9]),
                            s_ra=fl(c[11]), s_dec=fl(c[12])))
    if len(out) != len(jds):
        return None
    json.dump(out, open(f, "w"))
    time.sleep(0.3)
    return out


class Exposure:
    """CCD WCSs of one InstCal exposure, data read on demand."""
    def __init__(self, img, wt, msk):
        self.paths = (img, wt, msk)
        self.h = fits.open(img, memmap=False)
        self.ccds = []
        for k, x in enumerate(self.h):
            if k == 0 or "CTYPE1" not in x.header:
                continue
            self.ccds.append((k, WCS(x.header), x.header.get("NAXIS1", 2046), x.header.get("NAXIS2", 4094),
                              x.header.get("FWHM") or 4.0))
        self.cache = {}

    def locate(self, ra, dec):
        for k, w, nx, ny, fwhm in self.ccds:
            x, y = w.all_world2pix(ra, dec, 0)
            if 0 <= x < nx and 0 <= y < ny:
                return k, float(x), float(y), float(fwhm)
        return None

    def data(self, k):
        if k not in self.cache:
            with fits.open(self.paths[1], memmap=False) as hw, fits.open(self.paths[2], memmap=False) as hm:
                self.cache[k] = (self.h[k].data.astype(np.float32), hw[k].data.astype(np.float32), hm[k].data)
        return self.cache[k]

    def close(self):
        self.h.close(); self.cache.clear()


def phot(img, wt, msk, x, y, fwhm):
    ra_, r1, r2 = max(1.5, 0.7 * fwhm), 3 * fwhm, 5 * fwhm
    x0, x1 = int(max(0, np.floor(x - r2))), int(min(img.shape[1], np.ceil(x + r2) + 1))
    y0, y1 = int(max(0, np.floor(y - r2))), int(min(img.shape[0], np.ceil(y + r2) + 1))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    I, W, M = img[y0:y1, x0:x1], wt[y0:y1, x0:x1], msk[y0:y1, x0:x1]
    ap, an = d <= ra_, (d >= r1) & (d <= r2) & (M == 0) & (W > 0)
    if ap.sum() == 0 or an.sum() < 20:
        return None, None, "edge"
    flag = "bad" if ((M[ap] != 0) | (W[ap] <= 0)).any() else ""
    sky = np.median(I[an])
    f = float(np.sum(I[ap] - sky))
    var = float(np.sum(1.0 / np.where(W[ap] > 0, W[ap], np.inf))) + ap.sum() ** 2 * np.pi / 2 * np.var(I[an]) / an.sum()
    return f, float(np.sqrt(var)) if var > 0 else None, flag


def stack(f, e):
    f, e = np.array(f, float), np.array(e, float)
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    if not ok.any():
        return 0.0
    w = 1 / e[ok] ** 2
    return float(np.sum(f[ok] * w) / np.sqrt(np.sum(w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("sbdb")
    ap.add_argument("--sid", nargs="*"); ap.add_argument("--radius", type=float, default=1.2)
    ap.add_argument("--max-sigma", type=float, default=1.5); ap.add_argument("--snr3", type=float, default=5.0)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    pdir, cache = os.path.join(a.root, "private"), os.path.join(a.root, "private", "horizons")
    os.makedirs(cache, exist_ok=True); os.makedirs(os.path.join(pdir, "truth"), exist_ok=True)
    S = json.load(open(a.sbdb))
    rows = [dict(zip(S["fields"], r)) for r in S["data"]]
    rows = [r for r in rows if all(r[k] not in (None, "") for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H"))
            and float(r["e"]) < 1 and float(r["a"]) > 0]
    el = {k: np.array([float(r[k]) for r in rows]) for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H")}
    t0, nmot = el["epoch"] - 2400000.5, np.sqrt(GM / el["a"] ** 3)
    cosr = np.cos(np.radians(a.radius))
    files = sorted(glob.glob(os.path.join(pdir, "scenarios", "*.json")))
    for fn in files:
        P = json.load(open(fn))
        sid = P["sid"]
        if a.sid and sid not in a.sid:
            continue
        out = os.path.join(pdir, "truth", f"{sid}.json")
        if os.path.exists(out) and not a.redo:
            continue
        E = P["all_exposures"]
        held = set(P["held_back"])
        sdir = os.path.join(a.root, f"scen_{P['split']}", sid)
        data_dir = os.path.join(a.root, "data", str(P["window"]))
        mids = np.array([e["mjd_mid"] for e in E])
        ue = np.array([[np.cos(np.radians(e["dec_center"])) * np.cos(np.radians(e["ra_center"])),
                        np.cos(np.radians(e["dec_center"])) * np.sin(np.radians(e["ra_center"])),
                        np.sin(np.radians(e["dec_center"]))] for e in E])
        obs = np.array([[e["obs_x"], e["obs_y"], e["obs_z"]] for e in E])
        cand = np.zeros(len(rows), bool)
        for k in range(len(E)):
            r, _ = elements_to_rv(el["a"], el["e"], np.radians(el["i"]), np.radians(el["om"]), np.radians(el["w"]),
                                  np.radians(el["ma"]) + nmot * (mids[k] - t0))
            g = ecl_to_eq(r) - obs[k]
            cand |= (g / np.linalg.norm(g, axis=1)[:, None]) @ ue[k] > cosr
        idx = np.flatnonzero(cand)
        print(f"{sid}: {len(E)} exposures, {idx.size} candidate bodies", flush=True)
        exps = {}
        bodies = []
        for b in idx:
            pdes = rows[b]["pdes"]
            eph = horizons(pdes, [m + 2400000.5 for m in mids], cache)
            if eph is None:
                print("  no ephemeris", pdes, flush=True)
                continue
            recs = []
            for k, (e, q) in enumerate(zip(E, eph)):
                rec = dict(exp=e["image"], night=int(np.floor(e["mjd_mid"] - 0.5)), held=e["image"] in held, **q)
                if k not in exps:
                    exps[k] = Exposure(os.path.join(data_dir, e["image"]), os.path.join(data_dir, e["weight"]),
                                       os.path.join(data_dir, e["mask"]))
                loc = exps[k].locate(q["ra"], q["dec"]) if q["ra"] is not None else None
                if loc:
                    hdu, x, y, fwhm = loc
                    f, err, flag = phot(*exps[k].data(hdu), x, y, fwhm)
                    rec.update(hdu=hdu, x=round(x, 2), y=round(y, 2), flux=f, err=err, flag=flag,
                               snr=(f / err) if (f is not None and err) else None)
                recs.append(rec)
            if not any("hdu" in r for r in recs):
                continue
            vis = [r for r in recs if not r["held"] and "hdu" in r and r["flag"] == "" and r["err"]]
            sig = max([max(r["s_ra"] or 99, r["s_dec"] or 99) for r in recs if not r["held"]] or [99])
            tier = None
            if sig <= a.max_sigma:
                nights = {}
                for r in vis:
                    nights.setdefault(r["night"], []).append(r)
                n1 = sum(any(r["snr"] >= 5 for r in rs) for rs in nights.values())
                n2 = sum(stack([r["flux"] for r in rs], [r["err"] for r in rs]) >= 5 for rs in nights.values())
                tot = stack([r["flux"] for r in vis], [r["err"] for r in vis]) if vis else 0.0
                tier = 1 if n1 >= 3 else 2 if n2 >= 3 else 3 if (tot >= a.snr3 and len(nights) >= 3) else 0
            bodies.append(dict(pdes=pdes, H=float(el["H"][b]), first_obs=rows[b]["first_obs"], max_sigma=sig,
                               tier=tier, recs=recs))
            print(f"  {pdes:14s} H {el['H'][b]:5.2f} 3sig {sig:7.2f}\" tier {tier}", flush=True)
        for x in exps.values():
            x.close()
        json.dump(dict(sid=sid, split=P["split"], held_back=sorted(held), bodies=bodies), open(out, "w"))
        n = [b["tier"] for b in bodies]
        print(f"{sid} done: tiers 1/2/3/0/imprecise = {n.count(1)}/{n.count(2)}/{n.count(3)}/{n.count(0)}/{n.count(None)}", flush=True)


if __name__ == "__main__":
    main()
