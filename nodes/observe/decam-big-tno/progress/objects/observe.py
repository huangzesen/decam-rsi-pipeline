#!/usr/bin/env python3
"""Observability of every promising body over the next 14 nights, for the "Observe" tab of the Promising Bodies page
(2026-11-19).

Bodies: every decam-big-tno body of objects.json with an arc file (runs/<run>/arc/<id>.json: barycentric state p, cov,
epoch_tdb), plus the LDT 5 % search's large leads (runs/sldt5/followup_ids.txt, H <= 6.6, not yet tested in other images).
For each site and night: the Sun's and Moon's altitude, astronomical dusk and dawn, and each body's altitude and azimuth
every STEP minutes from sunset to sunrise (its place at local midnight; a distant body moves a few arcseconds an hour);
hours above airmass 2 in astronomical darkness, best altitude and its time, Moon separation, predicted V
(H + 5 log10(r Delta) + 0.15 x phase angle) and the place's 1-sigma (arcsec) from the orbit's covariance.
Right ascension and declination are not written (the page stays without sky positions of unfiled bodies; altitude and
azimuth are rounded to whole degrees).
Usage: ~/anaconda3/bin/python -I progress/objects/observe.py  -> progress/objects/observe.json
  --sites G37 --private --out FILE: one site, and each body's place at local midnight (RA/Dec, rate) for the private LDT
  Targets page (runs/ldt_20261119/; 2026-11-19).
Orbits: the refit with other archives' positions (arc/<id>.refit.json, ldt_20261117/refit.py) when it exists."""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_body, get_body_barycentric

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "harness"))
import arc_extend as AX
sys.path.insert(0, os.path.join(HERE, "..", "..", "planet-search", "linker"))
import orbit_fit as O

STEP = 20                       # minutes
NIGHTS = 14
SITES = [  # code, name, telescope note, lon, lat, height m, UTC offset h (standard or summer time in force in Nov-Dec 2026)
    ("G37", "Lowell Discovery Telescope", "4.3 m, Happy Jack, Arizona", -111.4223, 34.7444, 2360, -7, "MST"),
    ("568", "Maunakea", "Subaru, CFHT, Keck", -155.4717, 19.8260, 4139, -10, "HST"),
    ("807", "Cerro Tololo", "Blanco / DECam", -70.8065, -30.1691, 2207, -3, "CLST"),
    ("675", "Palomar", "Hale 5 m, ZTF", -116.8650, 33.3563, 1712, -8, "PST"),
]


def size(H):
    return round(1329 / 0.1 ** 0.5 * 10 ** (-H / 5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sites", default=""); ap.add_argument("--private", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "progress", "objects", "observe.json")); args = ap.parse_args()
    sites = [x for x in SITES if not args.sites or x[0] in args.sites.split(",")]
    d = json.load(open(os.path.join(HERE, "progress", "objects", "objects.json")))
    bodies = []
    for b in d["decam_big_tno"]:
        f = os.path.join(HERE, "runs", b["run"], "arc", b["id"] + ".json")
        if os.path.exists(f[:-5] + ".refit.json"):
            f = f[:-5] + ".refit.json"
        if os.path.exists(f) and b.get("H") is not None:
            name = b["label"] or b["id"].replace("dbt-", "").rsplit("-p", 1)[0]
            bodies.append(dict(key=b["id"], name=name, run=b["run"], exp=b["experiment"], stage=b["stage"], H=b["H"], arcf=f))
    # 2026-11-23: bodies proven to be catalogued MPC bodies need no observing (TRKSUB01: C###, C### were still in the plan)
    try:
        reg = json.load(open(os.path.join(HERE, "candidates", "registry.json")))["ids"]
        known = {c["cid"] for c in json.load(open(os.path.join(HERE, "candidates", "index.json")))["cards"]
                 if c.get("stage") == "known MPC body" or not c.get("active")}
        bodies = [x for x in bodies if reg.get(x["key"]) not in known]
    except Exception as e:
        print("known-body filter skipped:", e)
    have = {x["key"] for x in bodies}
    fl = os.path.join(HERE, "runs", "sldt5", "followup_ids.txt")
    lc = {x["id"]: x for x in json.load(open(os.path.join(HERE, "runs", "sldt5", "lead_check", "ldt5-gen044.json")))}
    for i in (open(fl).read().split() if os.path.exists(fl) else []):
        f = os.path.join(HERE, "runs", "sldt5", "arc", i + ".json")
        if i not in have and os.path.exists(f) and lc.get(i, {}).get("H") is not None:
            bodies.append(dict(key=i, name=i.replace("dbt-", "").rsplit("-p", 1)[0], run="sldt5",
                               exp="fourth 5 % search (LDT-weighted, conveyor)", stage="large, not yet tested in other images",
                               H=lc[i]["H"], arcf=f))
    print(len(bodies), "bodies with orbits")
    # nights: the local evening of today .. +13 at each site
    today = Time(time.strftime("%Y-%m-%d"), scale="utc")
    out = dict(asof=time.strftime("%Y-%m-%d %H:%M %Z"), step_min=STEP, sites=[], bodies=[])
    # body places at 0h UTC of each date (and +1 d), interpolated linearly to local midnight
    mjd0 = today.mjd + np.arange(NIGHTS + 2)
    sun_b = get_body_barycentric("sun", Time(mjd0, format="mjd", scale="utc")).xyz.to(u.au).value.T
    Rg, _ = O.observer_bary(mjd0, "W84")
    place = {}
    for b in bodies:
        A = json.load(open(b["arcf"]))
        if "p" not in A:                     # a refit with no fit (2026-11-23: crashed every refresh since 11-22 22:13): base arc
            base = b["arcf"].replace(".refit.json", ".json").replace(".nolone.json", ".json").replace(".drop.json", ".json")
            A = json.load(open(base)) if base != b["arcf"] and os.path.exists(base) else {}
            if "p" not in A or b.get("H") is None:
                continue
        p, cov, t0 = np.array(A["p"]), np.array(A["cov"]), A["epoch_tdb"]
        ra, dec, s1 = AX.predict(p, cov, t0, mjd0)
        tt = Time(mjd0, format="mjd", scale="utc").tdb.mjd
        xyz = O.propagate(p[:3], p[3:], tt - t0)
        r = np.linalg.norm(xyz - sun_b, axis=1); dl = np.linalg.norm(xyz - Rg, axis=1)
        cosa = np.einsum("ij,ij->i", xyz - sun_b, xyz - Rg) / (r * dl); alpha = np.degrees(np.arccos(np.clip(cosa, -1, 1)))
        V = b["H"] + 5 * np.log10(r * dl) + 0.15 * alpha
        place[b["key"]] = (np.unwrap(np.radians(ra)), dec, s1, V, r)
        out["bodies"].append(dict(key=b["key"], name=b["name"], run=b["run"], exp=b["exp"], stage=b["stage"], H=b["H"],
                                  D=size(b["H"]), r=round(float(r[0]), 1), a=A.get("a"), e=A.get("e"), i=A.get("i"),
                                  arc=A.get("arc_years"), nights=A.get("nights"),
                                  V=[round(float(v), 2) for v in V[:NIGHTS]], sig=[round(float(s), 1) for s in s1[:NIGHTS]]))
    keys = [b["key"] for b in bodies]
    for code, name, tel, lon, lat, h, utc, tzn in sites:
        loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, h * u.m)
        S = dict(code=code, name=name, tel=tel, utc=utc, tz=tzn, lat=lat, lon=lon, nights=[])
        for k in range(NIGHTS):
            noon = today.mjd + k + (12 - utc) / 24          # local noon of night k, in UTC MJD
            grid = noon + np.arange(0, 24 * 60 + 1, 5) / 1440
            T = Time(grid, format="mjd", scale="utc")
            sun = get_body("sun", T, loc).transform_to(AltAz(obstime=T, location=loc)).alt.deg
            def cross(level, rising):
                idx = np.where((sun[:-1] > level) & (sun[1:] <= level))[0] if not rising else np.where((sun[:-1] <= level) & (sun[1:] > level))[0]
                if not len(idx):
                    return None
                j = idx[0]; f = (sun[j] - level) / (sun[j] - sun[j + 1])
                return float(grid[j] + f * (grid[j + 1] - grid[j]))
            ss, sr = cross(-0.833, False), cross(-0.833, True)
            n12, d12 = cross(-12, False), cross(-12, True)
            a18, d18 = cross(-18, False), cross(-18, True)
            t = np.arange(ss, sr + 1e-9, STEP / 1440); Tt = Time(t, format="mjd", scale="utc"); fr = AltAz(obstime=Tt, location=loc)
            sun_alt = get_body("sun", Tt, loc).transform_to(fr).alt.deg
            mo = get_body("moon", Tt, loc); moa = mo.transform_to(fr)
            mid = (ss + sr) / 2; Tm = Time(mid, format="mjd", scale="utc")
            sm = get_body("sun", Tm, loc); mm = get_body("moon", Tm, loc)
            elong = sm.separation(mm).rad; illum = (1 - np.cos(elong)) / 2
            waxing = ((mm.ra - sm.ra).wrap_at(360 * u.deg).deg % 360) < 180
            # bodies at local midnight
            w = (mid - mjd0[0]); k0 = int(np.floor(w)); f = w - k0
            ra = np.array([place[x][0][k0] * (1 - f) + place[x][0][k0 + 1] * f for x in keys])
            de = np.array([place[x][1][k0] * (1 - f) + place[x][1][k0 + 1] * f for x in keys])
            c = SkyCoord(ra=np.degrees(ra) % 360 * u.deg, dec=de * u.deg)
            aa = c[:, None].transform_to(AltAz(obstime=Tt[None, :], location=loc))
            alt, az = aa.alt.deg, aa.az.deg
            msep = c.separation(get_body("moon", Tm, loc)).deg
            dark = sun_alt <= -18
            B = {}
            for j, x in enumerate(keys):
                good = dark & (alt[j] >= 30)
                dh = good.sum() * STEP / 60
                if (alt[j][dark] > 0).any() if dark.any() else False:
                    jj = np.argmax(np.where(dark, alt[j], -99))
                    B[x] = dict(alt=[int(round(v)) for v in alt[j]], az=[int(round(v)) for v in az[j]], dh=round(dh, 2),
                                best=int(round(alt[j][jj])), tbest=round((t[jj] - ss) * 1440), msep=int(round(msep[j])))
                    if args.private:      # place at local midnight and its motion (arcsec per hour, from the day's change)
                        k1 = min(k0 + 1, len(mjd0) - 1)
                        r1, d1 = place[x][0][k1], place[x][1][k1]; r0_, d0_ = place[x][0][k0], place[x][1][k0]
                        dra = (r1 - r0_) * np.cos(np.radians(de[j])); ddec = np.radians(d1 - d0_)
                        B[x].update(ra=round(float(np.degrees(ra[j]) % 360), 5), dec=round(float(de[j]), 5),
                                    rate=round(float(np.degrees(np.hypot(dra, ddec)) * 3600 / 24), 2))
            S["nights"].append(dict(date=Time(noon, format="mjd").strftime("%Y-%m-%d"), ss=round(ss, 6), sr=round(sr, 6),
                                    n12=n12, d12=d12, a18=a18, d18=d18, n=len(t),
                                    moon=dict(alt=[int(round(v)) for v in moa.alt.deg], az=[int(round(v)) for v in moa.az.deg],
                                              illum=round(float(illum), 3), waxing=bool(waxing)),
                                    sun=[int(round(v)) for v in sun_alt], bodies=B))
            print(code, S["nights"][-1]["date"], len(B), "bodies up in the dark", flush=True)
        out["sites"].append(S)
    json.dump(out, open(args.out, "w"), separators=(",", ":"))
    print("wrote", args.out, os.path.getsize(args.out) // 1024, "kB")


if __name__ == "__main__":
    main()
