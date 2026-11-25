#!/usr/bin/env python3
"""NERSC side: after a Globus copy to Vista has SUCCEEDED (checksums verified), delete the copied images under
DIR/img/ (regular files only), keeping every path listed in KEEP (e.g. archive15/a10n_keep.txt: files the third 5 % links to).
Usage: python3 -I clear_img.py DIR [KEEP] [--yes]   (without --yes: count only). Prints files and bytes removed and kept."""
import os, sys
d = os.path.realpath(sys.argv[1]); yes = "--yes" in sys.argv
keep = set()
if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
    keep = set(os.path.realpath(l.strip()) for l in open(sys.argv[2]) if l.strip())
assert d.startswith("$NERSC_SCRATCH/archive") and os.path.isdir(os.path.join(d, "img")), d
n = b = nk = bk = 0
for root, _, files in os.walk(os.path.join(d, "img")):
    for f in files:
        p = os.path.join(root, f)
        if os.path.islink(p) or not os.path.isfile(p): continue
        s = os.path.getsize(p)
        if p in keep: nk += 1; bk += s; continue
        if yes: os.remove(p)
        n += 1; b += s
print(f"{d}: {'removed' if yes else 'would remove'} {n} files {b/1e12:.2f} TB; kept {nk} files {bk/1e12:.2f} TB")
