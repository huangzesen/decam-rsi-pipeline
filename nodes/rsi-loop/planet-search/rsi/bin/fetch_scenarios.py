"""Download the DECam InstCal files (image, weight map, data-quality mask) of the field-windows picked by
select_scenarios.py from the NOIRLab Astro Data Archive.

  list SELECT.json FILES.json   one archive query per picked window (cell bounds, its dates); broad bands, >= 30 s,
                                released; one processing version per exposure (newest community-pipeline vN, else
                                the first special version); writes the file list with md5 and size
  get  FILES.json DATADIR       --workers parallel streams; each file goes to <window>/<name>.part, its md5 is checked
                                while streaming, then it is renamed; existing complete files are skipped, so a rerun
                                resumes. Progress lines go to stdout.

Usage: python -I fetch_scenarios.py list rsi/scenarios_select.json rsi/files.json
       python -I fetch_scenarios.py get rsi/files.json rsi/data [--workers 12]
"""
import argparse, hashlib, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

API = "https://astroarchive.noirlab.edu/api"
BANDS = {"g", "r", "i", "z", "Y", "VR"}


def caldat(m):
    return (date(1970, 1, 1) + timedelta(days=int(m) - 40587)).isoformat()


def find(body, limit=20000):
    req = urllib.request.Request(f"{API}/adv_search/find/?limit={limit}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for k in range(5):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=600).read())[1:]
        except Exception as e:
            print("query retry", k, e, flush=True); time.sleep(10 * (k + 1))
    raise RuntimeError("archive query failed")


def do_list(a):
    S = json.load(open(a.select))
    pick = set(S["picked"])
    wins = [w for w in S["candidates"] if w["key"] in pick]
    today = date.today().isoformat()
    out, tot = [], 0
    for n, w in enumerate(wins):
        r = find({"outfields": ["md5sum", "archive_filename", "prod_type", "filesize", "caldat", "exposure", "ifilter",
                                "ra_center", "dec_center", "release_date"],
                  "search": [["instrument", "decam"], ["proc_type", "instcal"], ["caldat", caldat(w["mjd0"]), caldat(w["mjd1"])],
                             ["ra_center", w["ra_lo"], w["ra_hi"]], ["dec_center", w["dec_lo"], w["dec_hi"]]]})
        exps = {}
        for x in r:
            band = (x["ifilter"] or "").split(" ")[0]
            if band not in BANDS or not (x["exposure"] or 0) >= 30 or (x["release_date"] or "9999") > today:
                continue
            fn = x["archive_filename"].split("/")[-1]
            m = re.match(r"(c4d_\d{6}_\d{6})_oo([iwd])_[^_]+_(.+)\.fits\.fz$", fn)
            if not m:
                continue
            exps.setdefault(m.group(1), {}).setdefault(m.group(3), {})[m.group(2)] = (x["md5sum"], x["filesize"], fn)
        files = []
        for base, vers in sorted(exps.items()):
            full = {v: d for v, d in vers.items() if set(d) == {"i", "w", "d"}}
            if not full:
                continue
            cp = sorted((v for v in full if re.fullmatch(r"v\d+", v)), key=lambda v: int(v[1:]))
            v = cp[-1] if cp else sorted(full)[0]
            for c in "iwd":
                md5, size, fn = full[v][c]
                files.append(dict(window=w["key"], name=fn, md5=md5, size=size))
        gb = sum(f["size"] for f in files) / 1e9
        tot += gb
        out.append(dict(window=w["key"], ra=w["ra"], dec=w["dec"], mjd0=w["mjd0"], mjd1=w["mjd1"], known=w["known"],
                        stack_only=w["stack_only"], n_exp=len(files) // 3, gb=round(gb, 2), files=files))
        print(f"{n + 1}/{len(wins)} window {w['key']}: {len(files) // 3} exposures ({w['n_exp']} raw), {gb:.1f} GB; total {tot:.0f} GB", flush=True)
    json.dump(dict(select=a.select, made=time.strftime("%Y-%m-%d %H:%M"), windows=out), open(a.files, "w"), indent=0)


def fetch(f, root):
    d = os.path.join(root, str(f["window"]))
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, f["name"])
    if os.path.exists(dst) and os.path.getsize(dst) == f["size"]:
        return f, "skip", 0.0
    part = dst + ".part"
    for k in range(4):
        try:
            h = hashlib.md5()
            t = time.time()
            with urllib.request.urlopen(f"{API}/retrieve/{f['md5']}/", timeout=600) as r, open(part, "wb") as o:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    h.update(b); o.write(b)
            if h.hexdigest() != f["md5"]:
                raise IOError("md5 mismatch")
            os.replace(part, dst)
            return f, "ok", time.time() - t
        except Exception as e:
            print("retry", f["name"], k, e, flush=True); time.sleep(15 * (k + 1))
    return f, "FAILED", 0.0


def do_get(a):
    W = json.load(open(a.files))["windows"]
    files = [f for w in W for f in w["files"]]
    total = sum(f["size"] for f in files)
    done, t0, nfail = 0, time.time(), 0
    print(f"{len(files)} files, {total / 1e9:.0f} GB, {a.workers} streams", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for k, fut in enumerate(as_completed([ex.submit(fetch, f, a.data) for f in files])):
            f, st, dt = fut.result()
            done += f["size"]
            nfail += st == "FAILED"
            if st != "skip" or k % 200 == 0:
                el = time.time() - t0
                print(f"{time.strftime('%H:%M:%S')} {k + 1}/{len(files)} {st} {f['name']} | {done / 1e9:.1f}/{total / 1e9:.0f} GB, "
                      f"{done / 1e6 / max(el, 1):.0f} MB/s avg, failed {nfail}", flush=True)
    print("DONE", f"failed {nfail}", flush=True)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("list"); p.add_argument("select"); p.add_argument("files")
p = sub.add_parser("get"); p.add_argument("files"); p.add_argument("data"); p.add_argument("--workers", type=int, default=12)
a = ap.parse_args()
do_list(a) if a.cmd == "list" else do_get(a)
