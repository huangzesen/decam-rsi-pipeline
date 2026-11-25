"""Fetch non-stationary single-exposure detections from the NOIRLab Source Catalog DR2 (Astro Data Lab, anonymous).

Usage: python -I nsc_fetch.py OUTDIR --ra 27.0 --dec 27.15 --radius 3.0 --mjd0 56800 --mjd1 58500 [--ndet 2]
Writes OUTDIR/exposures.csv (DECam exposures whose centre is within `radius` deg) and OUTDIR/meas/<exposure>.csv:
every measurement of that exposure whose NSC object has ndet <= `ndet` (detected at most that many times in all of
NSC, so stationary sources are dropped; a moving body seen twice within minutes is one object with ndet 2).
Queries go per exposure (the meas table is fast by exposure and times out on cone searches). Existing files are kept.
"""
import argparse, csv, io, os, time, urllib.parse, urllib.request

URL = "https://datalab.noirlab.edu/query/query"
TOKEN = "anonymous.0.0.anon_access"


def query(sql, tries=4):
    q = urllib.parse.urlencode(dict(sql=sql, ofmt="csv", async_="false")).replace("async_=", "async=")
    for k in range(tries):
        try:
            req = urllib.request.Request(f"{URL}?{q}", headers={"X-DL-AuthToken": TOKEN})
            txt = urllib.request.urlopen(req, timeout=300).read().decode()
            if txt.startswith("Error"):
                raise RuntimeError(txt.strip())
            return txt
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  retry {k + 1}: {e}", flush=True)
            time.sleep(10 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--dec", type=float, required=True)
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--mjd0", type=float, required=True)
    ap.add_argument("--mjd1", type=float, required=True)
    ap.add_argument("--ndet", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "meas"), exist_ok=True)
    ex = query(f"SELECT exposure, instrument, mjd, ra, dec, filter, exptime, depth95, fwhm, nmeas FROM nsc_dr2.exposure "
               f"WHERE q3c_radial_query(ra, dec, {a.ra}, {a.dec}, {a.radius}) AND mjd BETWEEN {a.mjd0} AND {a.mjd1} "
               f"AND instrument = 'c4d' ORDER BY mjd")
    open(os.path.join(a.out, "exposures.csv"), "w").write(ex)
    rows = list(csv.DictReader(io.StringIO(ex)))
    print(f"{len(rows)} exposures", flush=True)
    n = 0
    for k, r in enumerate(rows):
        f = os.path.join(a.out, "meas", r["exposure"] + ".csv")
        if os.path.exists(f):
            continue
        t0 = time.time()
        txt = query(f"SELECT m.measid, m.objectid, m.mjd, m.ra, m.dec, m.raerr, m.decerr, m.mag_auto, m.magerr_auto, "
                    f"m.filter, m.fwhm, m.flags, m.class_star, o.ndet FROM nsc_dr2.meas m "
                    f"JOIN nsc_dr2.object o ON o.id = m.objectid "
                    f"WHERE m.exposure = '{r['exposure']}' AND o.ndet <= {a.ndet}")
        open(f + ".part", "w").write(txt)
        os.replace(f + ".part", f)
        m = txt.count("\n") - 1
        n += m
        print(f"{k + 1}/{len(rows)} {r['exposure']} {m} rows {time.time() - t0:.1f}s", flush=True)
    print(f"done; {n} new rows")


if __name__ == "__main__":
    main()
