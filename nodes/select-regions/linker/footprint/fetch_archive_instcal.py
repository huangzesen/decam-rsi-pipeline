"""File list of DECam's Community Pipeline InstCal products (image ooi, weight map oow, data-quality mask ood):
name, md5, size, release date and exposure number of every file, from the NOIRLab Astro Data Archive. One query per month, one JSON per
year; existing years are kept. Metadata only: no image is downloaded. The md5 is the download key
(https://astroarchive.noirlab.edu/api/retrieve/<md5>/).

Used by ../../groups/kit/make_groups.py to give each group its file list.

Usage: python -I footprint/fetch_archive_instcal.py OUTDIR [--years 2014 2015 2016 ...]
"""
import argparse, calendar, json, os, re, time, urllib.request

API = "https://astroarchive.noirlab.edu/api"
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--years", type=int, nargs="+", default=list(range(2012, 2027)))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
pat = re.compile(r"c4d_\d{6}_\d{6}_oo[iwd]_[^_]+_.+\.fits\.fz$")
for y in a.years:
    f = os.path.join(a.out, f"instcal_{y}.json")
    if os.path.exists(f):
        continue
    rows = []
    for m in range(1, 13):
        if (y, m) < (2012, 9) or (y, m) > tuple(time.gmtime()[:2]):
            continue
        body = {"outfields": ["md5sum", "archive_filename", "filesize", "release_date", "EXPNUM"],
                "search": [["instrument", "decam"], ["proc_type", "instcal"],
                           ["caldat", f"{y}-{m:02d}-01", f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]}"]]}
        t = time.time()
        for attempt in range(5):
            try:
                req = urllib.request.Request(f"{API}/adv_search/find/?limit=400000", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
                break
            except Exception as e:
                print(y, m, "retry", attempt, repr(e)[:120], flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            raise SystemExit(f"{y}-{m:02d}: failed")
        n = 0
        for x in r[1:]:
            name = (x.get("archive_filename") or "").split("/")[-1]
            if pat.fullmatch(name):
                rows.append([name, x["md5sum"], x["filesize"], (x.get("release_date") or "")[:10], x.get("EXPNUM")])
                n += 1
        print(f"{y}-{m:02d}: {len(r) - 1} rows, {n} kept, {time.time() - t:.0f} s", flush=True)
        if len(r) - 1 >= 400000:
            raise SystemExit(f"{y}-{m:02d}: hit the row limit; split the month")
    json.dump(dict(fields=["name", "md5", "size", "release", "expnum"], rows=rows), open(f + ".part", "w"))
    os.replace(f + ".part", f)
    print(f"{y}: {len(rows)} files", flush=True)
