#!/usr/bin/env python3
"""Static parts of "DECam Data Movement" (2026-11-17): the DECam Slice
Atlas (every planned slice of the archive, from slices/atlas/build_atlas.py's data) and the measured numbers of the first 5 %
dashboard (archive5/dash), plus later measurements. Writes out/atlas.json {slices, slice_classes, measured}; upload it once.
Usage, from planet-search/: python3 -I fill/dash/build_static.py"""
import collections, json, os, re
D = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(D))
CAD = ["singles", "revisits", "rolling", "dense", "deep"]; SKY = ["ecliptic", "high latitude", "crowded"]
html = open(os.path.join(P, "slices/atlas/out/decam-slice-atlas.html")).read()
A, _ = json.JSONDecoder().raw_decode(html, html.index('const D = {"asof"') + len("const D = "))
groups, slices, w = A["groups"], A["slices"], A["w"]
rows, k = [], 0
for s in slices:
    gi, part, nw, nexp, gb, cost, first, last, bands, known = s
    ras, decs = [], []
    for j in range(nw):
        ra, dec = w[4 * (k + j)], w[4 * (k + j) + 1]; ras.append(ra); decs.append(dec)
    k += nw
    g = groups[gi]
    # circular mean of RA
    import math
    x = sum(math.cos(math.radians(r)) for r in ras); y = sum(math.sin(math.radians(r)) for r in ras)
    rows.append(dict(id=f"{g[0]}-{part:03d}", group=g[0], cadence=CAD[g[1]], sky=SKY[g[2]], program=g[3], windows=nw, exposures=nexp,
                     gb=gb, a100h=cost, first=str(first)[:10], last=str(last)[:10], bands=bands, known=known if known >= 0 else None,
                     ra=round(math.degrees(math.atan2(y, x)) % 360, 2), dec=round(sum(decs) / len(decs), 2)))
assert k * 4 == len(w), (k, len(w))
cls = collections.defaultdict(lambda: dict(slices=0, windows=0, exposures=0, tb=0.0, a100h=0.0))
for r in rows:
    c = cls[(r["cadence"], r["sky"])]; c["slices"] += 1; c["windows"] += r["windows"]; c["exposures"] += r["exposures"]; c["tb"] += r["gb"] / 1000; c["a100h"] += r["a100h"]
slice_classes = [dict(cls=f"{a} | {b}", cadence=a, sky=b, **{kk: (round(v, 1) if isinstance(v, float) else v) for kk, v in c.items()}) for (a, b), c in sorted(cls.items(), key=lambda kv: -kv[1]["exposures"])]
measured = json.load(open(os.path.join(P, "archive5/dash/out/progress.json")))["measured"]
measured += [
    dict(metric="Download, 4 free NERSC transfer jobs together", value="about 200 MB/s (9.6 TB in 13.5 h)", source="archive10 a10n jobs, 2026-11-14/15"),
    dict(metric="Download, one NERSC compute node", value="149 MB/s at 32 streams, no ceiling seen", source="job <slurm-job-id>, 2026-11-10"),
    dict(metric="Globus NERSC to Vista, one task", value="about 400 MB/s", source="task 965fe752, 2026-11-17"),
    dict(metric="Globus NERSC to Vista, three tasks at once", value="about 600-640 MB/s", source="a10n tasks, 2026-11-17"),
    dict(metric="Damaged downloads from NOIRLab", value="about 1-4 % of files fail md5 and are retried", source="fetch logs, 2026-11-12 to 11-15"),
]
# regions of the three 5 % searches (the second 5 % = the 25 archive10 regions of steps A and B)
regions = []
s10 = set(json.load(open(os.path.join(P, "archive10/stepA/regions_exposures.json")))) | set(json.load(open(os.path.join(P, "archive10/stepB/regions_exposures.json"))))
for folder, name, tag, keep in (("archive5", "first 5 %", "5", None), ("archive10", "second 5 %", "10", s10), ("archive15", "third 5 %", "15", None)):
    for r in json.load(open(os.path.join(P, folder, "regions.json")))["regions"]:
        if keep is not None and r["id"] not in keep: continue
        regions.append(dict(region=f"{tag}-{r['id']}", search=name, label=r["label"], ra=r["ra"], dec=r["dec"], deg2=r["deg2"], exposures=r["exposures"],
                            tb=r["tb"], years=f"{r['first_year']}-{r['last_year']}", seasons=r["seasons"]))
json.dump(dict(slices=rows, slice_classes=slice_classes, measured=measured, regions=regions), open(os.path.join(D, "out", "atlas.json"), "w"), separators=(",", ":"))
print(len(rows), "slices,", len(slice_classes), "classes,", len(measured), "measured,", len(regions), "regions;", os.path.getsize(os.path.join(D, "out", "atlas.json")) // 1024, "KB")
