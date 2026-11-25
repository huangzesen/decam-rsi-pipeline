#!/usr/bin/env python3
"""The DECam blank: every usable public exposure's InstCal files (bulk/ manifests, 2026-11-11 rule) that NERSC does not
hold, in download order. 2026-11-17.

Held = in the Legacy Surveys mirror at NERSC (linker/data_real/nersc_decam_mirror/mirror_exposures.tsv, any version),
or in the second 5 %'s downloads (fill/a10n_all_stems.txt, now also on Vista via Globus), or in the first 5 % (archive5,
already searched). Order: the third 5 %'s missing exposures (archive15/missing.txt) first, then the bulk manifests'
own order (pilot groups, plan stages, then exposures in no window by night), the two machine lists interleaved.
Output fill/blank.tsv (md5, size, name, tag; same format as bulk/, so bulk/fetch.py reads it) and fill/blank.json.
Usage: python3 -I fill/make_blank.py"""
import json, os, re
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(H, "linker/data_real/decam_bulk")
stamp = lambda n: re.match(r"(c4d_\d{6}_\d{6})", n).group(1)
held = set()
for line in open(os.path.join(H, "linker/data_real/nersc_decam_mirror/mirror_exposures.tsv")):
    if line.startswith("c4d_"): held.add(line.split("\t", 1)[0])
n_mirror = len(held)
a10n = {l.strip() for l in open(os.path.join(H, "fill/a10n_all_stems.txt")) if l.strip()}   # every exposure with an image in
# NERSC archive10/a10n_0..3/img (listed 2026-11-17 by find; 17,404 stems, some incomplete sets), now copied to Vista too
a5 = {s for v in json.load(open(os.path.join(H, "archive5/regions_exposures.json"))).values() for s in v}
held |= a10n | a5
first = [l.strip() for l in open(os.path.join(H, "archive15/missing.txt")) if l.strip()]
rows = {}; order = []
L = [open(os.path.join(B, f)).read().splitlines() for f in ("ls6.tsv", "vista.tsv")]
for i in range(max(map(len, L))):                     # interleave the two lists, keeping each one's order
    for part in L:
        if i < len(part):
            md5, size, name, tag = part[i].split("\t"); s = stamp(name)
            if s not in rows: rows[s] = []; order.append(s)
            rows[s].append((md5, size, name, tag))
missing_first = [s for s in first if s in rows]
seen = set(missing_first)
out = missing_first + [s for s in order if s not in held and s not in seen]
nb = 0; nf = 0
with open(os.path.join(H, "fill/blank.tsv.new"), "w") as f:
    for s in out:
        for md5, size, name, tag in rows[s]:
            f.write(f"{md5}\t{size}\t{name}\t{'a15' if s in seen else tag}\n"); nb += int(size); nf += 1
os.replace(os.path.join(H, "fill/blank.tsv.new"), os.path.join(H, "fill/blank.tsv"))
a15b = sum(int(r[1]) for s in missing_first for r in rows[s])
summ = dict(manifest_exposures=len(rows), held_mirror=n_mirror, held_a10n=len(a10n), held_archive5=len(a5),
            blank_exposures=len(out), blank_files=nf, blank_tb=round(nb / 1e12, 2),
            first_archive15_missing=len(missing_first), first_tb=round(a15b / 1e12, 2),
            archive15_missing_not_in_manifest=len(first) - len(missing_first))
json.dump(summ, open(os.path.join(H, "fill/blank.json"), "w"), indent=1); print(summ)
