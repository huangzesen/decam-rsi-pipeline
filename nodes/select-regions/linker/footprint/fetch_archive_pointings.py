"""Every DECam science exposure's pointing from the NOIRLab Astro Data Archive (raw, obs_type object), one JSON per
year: centre RA/Dec, calendar date, exposure time, filter. Existing years are kept.

Usage: python -I footprint/fetch_archive_pointings.py OUTDIR [--years 2012 2026]
"""
import argparse, json, os, time, urllib.request

API = "https://astroarchive.noirlab.edu/api"
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--years", type=int, nargs=2, default=(2012, 2026))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for y in range(a.years[0], a.years[1] + 1):
    f = os.path.join(a.out, f"decam_raw_{y}.json")
    if os.path.exists(f):
        continue
    body = {"outfields": ["ra_center", "dec_center", "caldat", "exposure", "ifilter"],
            "search": [["instrument", "decam"], ["obs_type", "object"], ["proc_type", "raw"], ["caldat", f"{y}-01-01", f"{y}-12-31"]]}
    t = time.time()
    req = urllib.request.Request(f"{API}/adv_search/find/?limit=300000", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    rows = [[x["ra_center"], x["dec_center"], x["caldat"], x["exposure"], (x["ifilter"] or "").split(" ")[0]] for x in r[1:]]
    json.dump(dict(fields=["ra", "dec", "caldat", "exptime", "filter"], rows=rows), open(f + ".part", "w"))
    os.replace(f + ".part", f)
    print(y, len(rows), f"{time.time() - t:.0f}s", flush=True)
