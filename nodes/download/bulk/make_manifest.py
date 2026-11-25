"""Manifest of every usable, public DECam exposure's InstCal files (image ooi, weight oow, mask ood), split between
two TACC machines for a bulk download to scratch (2026-11-11).

Usable = g r i z Y VR, >= 30 s, with coordinates, released by --asof: the rule of linker/footprint/groups.py. Files =
the newest complete Community Pipeline version of the exposure (groups/kit/make_groups.py load_instcal), all three
released by --asof. Order = the order the images are needed: the five pilot groups, then the other groups in the plan's
stage order (DECAM_PLAN.md section 4), then the exposures in no 3-night window, by night. A group's windows stay on
one machine; groups and nights go to whichever machine has fewer bytes so far, the pilots to Lonestar6.

Output in OUT/: ls6.tsv, vista.tsv (md5, size, name, tag; tag = group id or "x" + night), summary.json.
Metadata only: nothing is downloaded.

Usage: ~/anaconda3/bin/python -I bulk/make_manifest.py linker/data_real/noirlab_decam_meta linker/data_real/noirlab_decam_instcal \
           linker/footprint OUT [--asof 2026-11-11]
"""
import argparse, datetime, glob, json, os, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("meta"); ap.add_argument("instcal"); ap.add_argument("footprint"); ap.add_argument("out")
ap.add_argument("--asof", default=str(datetime.date.today()))
a = ap.parse_args()
BANDS = ("g", "r", "i", "z", "Y", "VR"); D0 = datetime.date(2012, 9, 1)
PILOTS = ["g099", "g121", "g162", "g176", "g177"]
STAGE = {"revisits | ecliptic": 0, "revisits | high latitude": 1, "rolling | high latitude": 1, "rolling | ecliptic": 1,
         "singles | high latitude": 2, "singles | ecliptic": 2, "dense | ecliptic": 3, "dense | high latitude": 3}

# 1. usable public exposures: stamp -> night
e2s = {}
for y in (2012, 2013):
    f = os.path.join(a.instcal, f"instcal_{y}.json")
    if os.path.exists(f):
        for row in json.load(open(f))["rows"]:
            if len(row) > 4 and row[4] is not None:
                e2s[int(row[4])] = row[0][:17]
usable = {}; n_all = n_held = n_nostamp = 0
for f in sorted(glob.glob(os.path.join(a.meta, "decam_meta_*.json"))):
    for r in json.load(open(f))["rows"]:
        n_all += 1
        if r[0] is None or r[1] is None or not r[2] or r[4] not in BANDS or r[3] is None or float(r[3]) < 30:
            continue
        if (datetime.date.fromisoformat(r[2][:10]) - D0).days < 0:
            continue
        if (r[9] or "9999")[:10] > a.asof:
            n_held += 1; continue
        st = r[10] or e2s.get(r[7])
        if st is None:
            n_nostamp += 1; continue
        usable[st] = r[2][:10]
print(f"{n_all} raw rows, {len(usable)} usable public with a stamp, {n_held} not yet public, {n_nostamp} without a stamp", flush=True)

# 2. their files: newest complete Community Pipeline version, all three released
vers = {}
for f in sorted(glob.glob(os.path.join(a.instcal, "instcal_*.json"))):
    for name, md5, size, rel, *_ in json.load(open(f))["rows"]:
        m = re.match(r"(c4d_\d{6}_\d{6})_oo([iwd])_[^_]+_(.+)\.fits\.fz$", name)
        if m and m.group(1) in usable:
            vers.setdefault(m.group(1), {}).setdefault(m.group(3), {})[m.group(2)] = (name, md5, size, rel or "")
files = {}; n_closed = 0
for st, vs in vers.items():
    full = {v: x for v, x in vs.items() if set(x) == {"i", "w", "d"}}
    if not full:
        continue
    cp = sorted((v for v in full if re.fullmatch(r"v\d+", v)), key=lambda v: int(v[1:]))
    pick = full[cp[-1] if cp else sorted(full)[0]]
    if any(x[3][:10] > a.asof or not x[1] or not x[2] for x in pick.values()):
        n_closed += 1; continue
    files[st] = [pick[k][:3] for k in "iwd"]
print(f"{len(files)} exposures with a complete public InstCal set; {len(usable) - len(files) - n_closed} without one, "
      f"{n_closed} with a set not yet public", flush=True)

# 3. which group each exposure is in
W = json.load(open(os.path.join(a.footprint, "groups_windows.json")))
WE = json.load(open(os.path.join(a.footprint, "groups_exposures.json")))["windows"]
gfam = {}; gexp = defaultdict(list)
for w in W:
    g = w["id"]
    gfam[g] = w["family"]
    for row in WE[str(w["key"])]:
        if row[0] in files:
            gexp[g].append(row[0])
in_window = {st for v in gexp.values() for st in v}
size = {st: sum(x[2] for x in fs) for st, fs in files.items()}

def gorder(g):
    return (0 if g in PILOTS else 1, STAGE.get(gfam[g], 4), g)
units = [(g, gexp[g]) for g in sorted(gexp, key=gorder)]
nights = defaultdict(list)
for st in files:
    if st not in in_window:
        nights[usable[st]].append(st)
units += [("x" + n.replace("-", ""), sorted(nights[n])) for n in sorted(nights)]

# 4. split by bytes, in order
site = {"ls6": [], "vista": []}; tot = {"ls6": 0, "vista": 0}; nexp = {"ls6": 0, "vista": 0}; where = {}
for tag, sts in units:
    b = sum(size[s] for s in sts)
    s = "ls6" if tag in PILOTS else min(tot, key=tot.get)
    site[s].append((tag, sts)); tot[s] += b; nexp[s] += len(sts); where[tag] = s
os.makedirs(a.out, exist_ok=True)
for s, us in site.items():
    with open(os.path.join(a.out, s + ".tsv"), "w") as o:
        for tag, sts in us:
            for st in sts:
                for name, md5, sz in files[st]:
                    o.write(f"{md5}\t{sz}\t{name}\t{tag}\n")
by_year = defaultdict(lambda: [0, 0])
for st in files:
    y = usable[st][:4]; by_year[y][0] += 1; by_year[y][1] += size[st]
S = dict(made=str(datetime.date.today()), asof=a.asof, raw_rows=n_all, usable_public=len(usable), not_yet_public=n_held,
         exposures=len(files), files=3 * len(files), bytes=sum(size.values()),
         in_windows=dict(exposures=len(in_window), bytes=sum(size[s] for s in in_window), groups=len(gexp)),
         outside_windows=dict(exposures=len(files) - len(in_window), bytes=sum(size[s] for s in files if s not in in_window), nights=len(nights)),
         sites={s: dict(exposures=nexp[s], files=3 * nexp[s], bytes=tot[s], groups=sum(1 for t, _ in site[s] if not t.startswith("x")),
                        nights=sum(1 for t, _ in site[s] if t.startswith("x"))) for s in site},
         group_site={t: s for t, s in where.items() if not t.startswith("x")},
         by_year={y: dict(exposures=v[0], bytes=v[1]) for y, v in sorted(by_year.items())},
         median_exposure_bytes=sorted(size.values())[len(size) // 2])
json.dump(S, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
TB = 1e12
print(f"total {S['exposures']} exposures, {S['files']} files, {S['bytes'] / TB:.1f} TB "
      f"(in windows {S['in_windows']['exposures']} / {S['in_windows']['bytes'] / TB:.1f} TB, "
      f"outside {S['outside_windows']['exposures']} / {S['outside_windows']['bytes'] / TB:.1f} TB)")
for s in site:
    print(f"  {s}: {nexp[s]} exposures, {tot[s] / TB:.1f} TB, {S['sites'][s]['groups']} groups, {S['sites'][s]['nights']} nights")
