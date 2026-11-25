"""Rubin's actual visits from its public nightly scheduler reports (schedview nightsum, LSSTCam, s3df.slac.stanford.edu).
Each report embeds Bokeh tables of that night's visits; every table with fieldRA is read and the visits are merged by
observationId. One JSON per night: observationId, observationStartMJD, fieldRA, fieldDec, band, science_program,
observation_reason. The HTML (~10 MB each) is parsed in a temporary file and not kept. Existing nights are kept.

Usage: python -I footprint/fetch_rubin_visits.py REPORT_LIST.json OUTDIR TMPDIR [--workers 3]
"""
import argparse, json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor

B = "https://s3df.slac.stanford.edu/data/rubin/sim-data/schedview/reports/nightsum/lsstcam/"
COLS = ["observationId", "observationStartMJD", "fieldRA", "fieldDec", "band", "science_program", "observation_reason"]


def tables(s):
    dec = json.JSONDecoder()
    out = []
    def walk(o):
        if isinstance(o, dict):
            if o.get("type") == "map" and isinstance(o.get("entries"), list):
                d = {e[0]: e[1] for e in o["entries"] if isinstance(e, list) and len(e) == 2 and isinstance(e[0], str)}
                if "fieldRA" in d and isinstance(d["fieldRA"], list):
                    out.append(d)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for m in re.finditer(r"const docs_json = ", s):
        walk(dec.raw_decode(s, m.end())[0])
    return out


def one(rel, out, tmp):
    night = re.search(r"(\d{4}-\d{2}-\d{2})", rel).group(1)
    f = os.path.join(out, f"visits_{night}.json")
    if os.path.exists(f):
        return night, None
    h = os.path.join(tmp, f"{night}.html")
    r = subprocess.run(["curl", "-s", "-f", "-m", "600", "-o", h, B + rel])
    if r.returncode:
        return night, f"curl {r.returncode}"
    size = os.path.getsize(h)
    vis = {}
    for d in tables(open(h, errors="replace").read()):
        n = len(d["fieldRA"])
        for i in range(n):
            key = d.get("observationId", [None] * n)[i] or d.get("observationStartMJD")[i]
            vis[str(key)] = [d[c][i] if c in d and isinstance(d[c], list) and len(d[c]) == n else None for c in COLS]
    os.remove(h)
    json.dump(dict(night=night, report_bytes=size, fields=COLS, rows=list(vis.values())), open(f + ".part", "w"))
    os.replace(f + ".part", f)
    return night, f"{len(vis)} visits, {size / 1e6:.1f} MB report"


ap = argparse.ArgumentParser()
ap.add_argument("reports")
ap.add_argument("out")
ap.add_argument("tmp")
ap.add_argument("--workers", type=int, default=3)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
os.makedirs(a.tmp, exist_ok=True)
rels = json.load(open(a.reports))
with ThreadPoolExecutor(a.workers) as ex:
    for night, msg in ex.map(lambda r: one(r, a.out, a.tmp), rels):
        if msg:
            print(night, msg, flush=True)
print("done")
