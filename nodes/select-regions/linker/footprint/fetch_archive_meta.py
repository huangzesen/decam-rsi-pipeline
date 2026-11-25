"""Every DECam science exposure with the metadata needed to group the archive: pointing, night, exact start time,
exposure time, filter, proposal (the observing program), exposure number, object name, release date and the file stamp
(c4d_YYMMDD_HHMMSS, shared by the Community Pipeline products). NOIRLab Astro Data Archive, raw, obs_type object, one
JSON per year. Existing years are kept. The older `fetch_archive_pointings.py` output (5 fields) is left as it is.

Usage: python -I footprint/fetch_archive_meta.py OUTDIR [--years 2012 2026]
"""
import argparse, json, os, re, time, urllib.request

API = "https://astroarchive.noirlab.edu/api"
OUT = ["ra_center", "dec_center", "caldat", "exposure", "ifilter", "proposal", "dateobs_min", "EXPNUM", "OBJECT",
       "release_date", "archive_filename"]
FIELDS = ["ra", "dec", "caldat", "exptime", "filter", "proposal", "dateobs", "expnum", "object", "release", "stamp"]
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--years", type=int, nargs=2, default=(2012, 2026))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for y in range(a.years[0], a.years[1] + 1):
    f = os.path.join(a.out, f"decam_meta_{y}.json")
    if os.path.exists(f):
        continue
    rows = []
    for h0, h1 in (("01-01", "06-30"), ("07-01", "12-31")):
        body = {"outfields": OUT, "search": [["instrument", "decam"], ["obs_type", "object"], ["proc_type", "raw"],
                                             ["caldat", f"{y}-{h0}", f"{y}-{h1}"]]}
        t = time.time()
        for attempt in range(4):
            try:
                req = urllib.request.Request(f"{API}/adv_search/find/?limit=300000", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
                break
            except Exception as e:
                print(y, h0, "retry", attempt, repr(e)[:120], flush=True)
                time.sleep(30)
        else:
            raise SystemExit(f"{y} {h0}: failed")
        for x in r[1:]:
            m = re.search(r"(c4d_\d{6}_\d{6})", x.get("archive_filename") or "")
            rows.append([x["ra_center"], x["dec_center"], x["caldat"], x["exposure"], (x["ifilter"] or "").split(" ")[0],
                         x.get("proposal"), x.get("dateobs_min"), x.get("EXPNUM"), x.get("OBJECT"), x.get("release_date"),
                         m.group(1) if m else None])
        print(y, h0, len(r) - 1, f"{time.time() - t:.0f}s", flush=True)
    json.dump(dict(fields=FIELDS, rows=rows), open(f + ".part", "w"))
    os.replace(f + ".part", f)
