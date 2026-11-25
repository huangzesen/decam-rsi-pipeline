"""Download manifest for a list of slices: md5, size, name, slice id per line, in the order given (the format
fetch.py reads). A slice's files come from its group: groups/<gid>/windows.json (which window is in which slice) and
groups/<gid>/files.json (the three InstCal files of each exposure). Metadata only.

Usage: python3 -I bulk/slice_manifest.py OUT.tsv g099-017 g121-009 ...
"""
import json, os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out, ids = sys.argv[1], sys.argv[2:]
G = {}; tot = 0; nexp = 0
with open(out, "w") as o:
    for sid in ids:
        gid = sid.split("-")[0]
        if gid not in G:
            G[gid] = (json.load(open(os.path.join(root, "groups", gid, "windows.json")))["windows"],
                      json.load(open(os.path.join(root, "groups", gid, "files.json"))))
        W, F = G[gid]; b = 0; n = 0
        for w in W:
            if w.get("slice") != sid:
                continue
            for e in w["exposures"]:
                f = F.get(e["stamp"])
                if not f:
                    continue
                n += 1
                for p in "iwd":
                    name, md5, size = f[p][:3]
                    o.write(f"{md5}\t{size}\t{name}\t{sid}\n"); b += size
        print(f"{sid}: {n} exposures, {b / 1e9:.1f} GB"); tot += b; nexp += n
print(f"total {len(ids)} slices, {nexp} exposures, {3 * nexp} files, {tot / 1e12:.3f} TB -> {out}")
