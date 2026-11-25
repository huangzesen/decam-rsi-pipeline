#!/usr/bin/env python3
"""NERSC side of fill/release.py: delete the regular files in image folders whose Globus copy to Vista has SUCCEEDED.
Reads folder paths (absolute) on stdin; accepts only folders inside $NERSC_SCRATCH/archive5/n<k>/img or
$NERSC_SCRATCH/archive10/a10n_<k>/img. Files listed in archive15/a10n_keep.txt stay (the third 5 % links to them).
Prints: folders, files removed, TB removed, files kept."""
import os, re, sys
OK = re.compile(r"^$NERSC_SCRATCH/(archive5/n\d|archive10/a10n_\d)/img(/\d{6})?$")
keep = set(os.path.realpath(l.strip()) for l in open("$NERSC_SCRATCH/archive15/a10n_keep.txt") if l.strip())
nd = n = b = k = 0
for line in sys.stdin:
    d = os.path.realpath(line.strip())
    if not OK.match(d) or not os.path.isdir(d): continue
    nd += 1
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p) or not os.path.isfile(p): continue
            if p in keep: k += 1; continue
            b += os.path.getsize(p); os.remove(p); n += 1
print(nd, n, round(b / 1e12, 3), k)
