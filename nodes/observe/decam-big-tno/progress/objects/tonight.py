#!/usr/bin/env python3
"""Which promising bodies (progress/objects/objects.json) can a given telescope observe on a given night, brighter than a
magnitude limit. 2026-11-17. Default site G37 (Lowell Discovery Telescope, Happy Jack AZ). Per body: orbit from the refit or
extended arc (runs/<run>/arc/<id>[.refit].json), place and 1-sigma from the arc covariance, V = H + 5 log10(r Delta)
(no phase term; H as the lead check measured it, so about +-0.3 mag), the time it is above ALT_MIN deg in astronomical
darkness, its rate and the Moon. Private: carries positions of unfiled bodies; never on a shared page.
Usage: ~/anaconda3/bin/python -I progress/objects/tonight.py [--date 2026-11-17] [--vmax 23] [--alt 30]"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun, get_body, get_body_barycentric
from astropy.time import Time
HERE = "$ALE_HOME/rsi/decam-big-tno"
sys.path.insert(0, os.path.join(HERE, "harness"))
import lead_check as LC, observer_batch as QB     # noqa: E402
from lead_check import O, unit_to_radec          # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--date", default="2026-11-17"); ap.add_argument("--vmax", type=float, default=23.0)
ap.add_argument("--alt", type=float, default=30.0); a = ap.parse_args()
LDT = EarthLocation.from_geodetic(-111.4223 * u.deg, 34.7444 * u.deg, 2361 * u.m)
O.SITES.setdefault("G37", LDT)
# the local night: 17:00 MST on --date to 07:00 MST next day, every 10 min (MST = UTC-7)
t_utc = Time(a.date + "T00:00:00") + (24 * u.hour) + np.arange(0, 14 * 60 + 1, 10) * u.min
frame = AltAz(obstime=t_utc, location=LDT)
sun_alt = get_sun(t_utc).transform_to(frame).alt.deg; dark = sun_alt < -18
moon = get_body("moon", t_utc, LDT); moon_alt = moon.transform_to(frame).alt.deg
sun_c = get_sun(t_utc[len(t_utc) // 2]); m_c = moon[len(t_utc) // 2]
illum = (1 - np.cos(np.radians(sun_c.separation(m_c).deg))) / 2      # rough illuminated fraction from the elongation
mst = lambda t: (t - 7 * u.hour).strftime("%H:%M")
D = json.load(open(os.path.join(HERE, "progress/objects/objects.json")))["decam_big_tno"]
mjd = t_utc.utc.mjd; rows = []; skipped = []
for x in D:
    f = [p for p in (f"{HERE}/runs/{x['run']}/arc/{x['id']}.refit.json", f"{HERE}/runs/{x['run']}/arc/{x['id']}.json") if os.path.exists(p)]
    if not f: skipped.append(x.get("label") or x["id"]); continue
    if x.get("H") is None: skipped.append((x.get("label") or x["id"]) + " (no H)"); continue
    A = json.load(open(f[0])); p = np.array(A["p"], float); cov = np.array(A["cov"], float); t0 = float(A["epoch_tdb"])
    if not (0 < (A.get("a") or 0) and (A.get("e") or 0) < 1): skipped.append((x.get("label") or x["id"]) + " (unbound)"); continue
    R, tt = O.observer_bary(mjd, "G37")
    uu, s1 = O.predict_sigma(p, cov, t0, tt, R)
    ra, de = unit_to_radec(uu); ra = np.degrees(ra) % 360; de = np.degrees(de)
    pos = O.propagate(p[:3], p[3:], tt - t0); sun = get_body_barycentric("sun", Time(tt, format="mjd", scale="tdb")).xyz.to("au").value.T
    rh = np.linalg.norm(pos - sun, axis=1); dl = np.linalg.norm(pos - R, axis=1)
    V = x["H"] + 5 * np.log10(rh * dl)
    c = SkyCoord(ra * u.deg, de * u.deg); alt = c.transform_to(frame).alt.deg
    ok = dark & (alt >= a.alt)
    i0 = len(t_utc) // 2
    rate = c[i0].separation(c[i0 + 6]).arcsec          # arcsec per hour (6 steps of 10 min)
    row = dict(label=x.get("label") or "", id=x["id"], run=x["run"], stage=x["stage"], rubin=bool(x.get("rubin")), H=x["H"],
               r=round(float(rh[i0]), 1), V=round(float(V[i0]), 2), sigma=round(float(s1[i0]), 1),
               ra=round(float(ra[i0]), 5), dec=round(float(de[i0]), 5), ra_hms=QB.ra_hms(ra[i0]), dec_dms=QB.dec_dms(de[i0]),
               rate=round(float(rate), 2), alt_max=round(float(alt[dark].max()), 0) if dark.any() else None,
               hours=round(ok.sum() / 6, 1), start=mst(t_utc[ok][0]) if ok.any() else None, end=mst(t_utc[ok][-1]) if ok.any() else None,
               moon_sep=round(float(c[i0].separation(m_c).deg), 0), nights=x.get("nights"), arc_years=x.get("arc_years"),
               archives=x.get("archives"))
    rows.append(row)
good = sorted([r for r in rows if r["V"] < a.vmax and r["hours"] >= 1], key=lambda r: r["V"])
out = dict(date=a.date, site="G37 LDT", vmax=a.vmax, alt_min=a.alt, dark_mst=[mst(t_utc[dark][0]), mst(t_utc[dark][-1])],
           moon_illum=round(float(illum), 2), moon_up_dark_hours=round(float((dark & (moon_alt > 0)).sum() / 6), 1),
           n_checked=len(rows), skipped=skipped, observable=good,
           brighter_but_not_up=[(r["label"] or r["id"], r["V"], r["alt_max"]) for r in rows if r["V"] < a.vmax and r["hours"] < 1])
json.dump(out, open(os.path.join(HERE, f"progress/objects/tonight_{a.date}_G37.json"), "w"), indent=1)
print(f"{a.date} LDT: dark {out['dark_mst'][0]}-{out['dark_mst'][1]} MST, Moon {out['moon_illum']:.0%} lit, up {out['moon_up_dark_hours']} dark h; "
      f"{len(rows)} bodies with orbits, {len(good)} with V < {a.vmax} and >= 1 h above {a.alt:.0f} deg")
for r in good:
    print(f"{r['label'] or r['id'][-40:]:42s} V {r['V']:5.2f} H {r['H']:.2f} r {r['r']:5.1f} | {r['ra_hms']} {r['dec_dms']} 1sig {r['sigma']:6.1f}\" "
          f"rate {r['rate']:.1f}\"/h | up {r['start']}-{r['end']} max {r['alt_max']:.0f} | moon {r['moon_sep']:.0f} | {r['stage']}{' | RUBIN-ONLY' if r['rubin'] else ''}")
print("brighter than the limit but not up >= 1 h:", out["brighter_but_not_up"])
print("skipped:", skipped)
