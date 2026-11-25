"""Turn downloaded field-windows into scenarios for the RSI finder task.

For each window whose files are all present (image, weight, mask of every exposure), read the InstCal headers and write
  <ROOT>/scen_dev/<sid>/ or <ROOT>/scen_sealed/<sid>/   what a finder script sees:
      exposures/          hard links to the InstCal files (same APFS volume, no extra space)
      exposures.csv       one row per exposure: exp_id, image, weight, mask, mjd_start, exptime, mjd_mid, band, airmass,
                          magzero, seeing_arcsec (median CCD FWHM x 0.263), ra_center, dec_center, obs_x, obs_y, obs_z
                          (observer = CTIO, heliocentric ICRS equatorial, au, at mjd_mid)
      meta.json           sid, centre, first/last night, number of exposures and nights
  <ROOT>/private/scenarios/<sid>.json   window key, split, held-back exposures (never shown to a finder)
Held back: one middle night in windows with >= --holdback-min nights. Split: --dev windows go to scen_dev (a finder may
be run there during development; no truth is ever given), the rest to scen_sealed. The split is a fixed hash of the
window key, so reruns are stable and new windows join without moving old ones.

Usage: ~/anaconda3/bin/python -I make_scenarios.py ROOT FILES.json [FILES2.json ...] [--dev-frac 0.12]
"""
import argparse, csv, hashlib, json, os, sys
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_body_barycentric, solar_system_ephemeris
import astropy.units as u

PIX = 0.263
SITE = EarthLocation(lat=-30.16606 * u.deg, lon=-70.81489 * u.deg, height=2215 * u.m)
COLS = ["exp_id", "image", "weight", "mask", "mjd_start", "exptime", "mjd_mid", "band", "airmass", "magzero",
        "seeing_arcsec", "ra_center", "dec_center", "obs_x", "obs_y", "obs_z"]


def observer(mjd):
    t = Time(mjd, format="mjd", scale="utc")
    with solar_system_ephemeris.set("builtin"):
        e = get_body_barycentric("earth", t).xyz.to(u.au).value
        s = get_body_barycentric("sun", t).xyz.to(u.au).value
    g = SITE.get_gcrs_posvel(t)[0].xyz.to(u.au).value
    return e - s + g


def frac(key):
    return int(hashlib.sha1(str(key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("files", nargs="+")
    ap.add_argument("--data", default=None, help="download dir (default ROOT/data)")
    ap.add_argument("--dev-frac", type=float, default=0.12)
    ap.add_argument("--holdback-min", type=int, default=6)
    a = ap.parse_args()
    data = a.data or os.path.join(a.root, "data")
    os.makedirs(os.path.join(a.root, "private", "scenarios"), exist_ok=True)
    W = [w for f in a.files for w in json.load(open(f))["windows"] if w["n_exp"]]
    made = skipped = 0
    for w in W:
        key = w["window"]
        sid = "s" + hashlib.sha1(str(key).encode()).hexdigest()[:8]
        split = "dev" if frac(key) < a.dev_frac else "sealed"
        out = os.path.join(a.root, f"scen_{split}", sid)
        if os.path.exists(os.path.join(out, "meta.json")):
            continue
        groups = {}
        for f in w["files"]:
            p = os.path.join(data, str(key), f["name"])
            if not (os.path.exists(p) and os.path.getsize(p) == f["size"]):
                groups = None
                break
            base, c = f["name"][:17], f["name"][20]
            groups.setdefault(base, {})[{"i": "image", "w": "weight", "d": "mask"}[c]] = f["name"]
        if not groups:
            skipped += 1
            continue
        rows = []
        for base, g in sorted(groups.items()):
            with fits.open(os.path.join(data, str(key), g["image"]), memmap=False) as h:
                p = h[0].header
                fw = [x.header.get("FWHM") for x in h[1:] if x.header.get("FWHM")]
                mjd0, t = float(p["MJD-OBS"]), float(p["EXPTIME"])
                mid = mjd0 + t / 2 / 86400
                o = observer(mid)
                ra = (p.get("RA") or "0:0:0").split(":"); de = (p.get("DEC") or "0:0:0").split(":")
                rad = 15 * (float(ra[0]) + float(ra[1]) / 60 + float(ra[2]) / 3600)
                sg = -1 if de[0].strip().startswith("-") else 1
                ded = sg * (abs(float(de[0])) + float(de[1]) / 60 + float(de[2]) / 3600)
                rows.append(dict(exp_id=int(p.get("EXPNUM", 0)), image=g["image"], weight=g["weight"], mask=g["mask"],
                                 mjd_start=round(mjd0, 8), exptime=t, mjd_mid=round(mid, 8),
                                 band=str(p.get("FILTER", "")).split(" ")[0], airmass=p.get("AIRMASS"),
                                 magzero=p.get("MAGZERO"), seeing_arcsec=round(float(np.median(fw)) * PIX, 3) if fw else None,
                                 ra_center=round(rad, 6), dec_center=round(ded, 6),
                                 obs_x=round(float(o[0]), 9), obs_y=round(float(o[1]), 9), obs_z=round(float(o[2]), 9)))
        nights = sorted({int(np.floor(r["mjd_mid"] - 0.5)) for r in rows})   # local night (UTC-4/-3 at CTIO)
        held = []
        if len(nights) >= a.holdback_min:
            hn = nights[len(nights) // 2]
            held = [r["image"] for r in rows if int(np.floor(r["mjd_mid"] - 0.5)) == hn]
        vis = [r for r in rows if r["image"] not in held]
        os.makedirs(os.path.join(out, "exposures"), exist_ok=True)
        for r in vis:
            for k in ("image", "weight", "mask"):
                dst = os.path.join(out, "exposures", r[k])
                if not os.path.exists(dst):
                    os.link(os.path.join(data, str(key), r[k]), dst)
        with open(os.path.join(out, "exposures.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, COLS); wr.writeheader()
            for r in vis:
                wr.writerow({k: (f"exposures/{r[k]}" if k in ("image", "weight", "mask") else r[k]) for k in COLS})
        vn = sorted({int(np.floor(r["mjd_mid"] - 0.5)) for r in vis})
        meta = dict(sid=sid, ra=w["ra"], dec=w["dec"], first_mjd=min(r["mjd_mid"] for r in vis),
                    last_mjd=max(r["mjd_mid"] for r in vis), n_exposures=len(vis), n_nights=len(vn),
                    bands="".join(sorted({r["band"] for r in vis})))
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        json.dump(dict(sid=sid, window=key, split=split, held_back=held, all_exposures=rows,
                       n_nights_total=len(nights)), open(os.path.join(a.root, "private", "scenarios", f"{sid}.json"), "w"), indent=0)
        made += 1
        print(f"{sid} {split:6s} window {key}: {len(vis)} exposures on {len(vn)} nights (held back {len(held)})", flush=True)
    print(f"made {made}, waiting for downloads {skipped}")


if __name__ == "__main__":
    main()
