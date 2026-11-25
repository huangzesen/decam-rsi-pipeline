#!/usr/bin/env python3
"""Cut a region and its labels to a part of the sky, so that a test of the program runs in minutes on CPU.

    python3 /app/tools/subset.py SID WORK --radius 1.5                      a disc of 1.5 deg at the region centre
    python3 /app/tools/subset.py SID WORK --radius 1.5 --centre RA DEC      a disc somewhere else (degrees)
    python3 /app/tools/subset.py SID WORK --near-bodies 0.5                 everything within 0.5 deg of a labelled detection
                                                                           of a findable body (every findable body stays whole)
    options: --name NAME (default SID_cut), --open DIR (default /app/open), --max-bodies N (with --near-bodies: only the
             first N findable bodies, by index)

Writes WORK/scen/NAME/ (the same files as a region; every exposure is kept, rows keep their order) and
WORK/labels/NAME.npz (labels for the kept rows), plus WORK/scen/NAME/rows.npy (row number in the full region of every kept
row). Then:
    python3 submission/finder.py --scen WORK/scen/NAME --out OUT/NAME --budget-s 600
    python3 /app/tools/score.py OUT WORK/labels GRADE --sids NAME --report GRADE/report.md
A cut is easier than a whole region in one way (fewer false linkages compete for the scored slots) and the same in the
others (the same detections, noise and cadence around each body).
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("sid"); ap.add_argument("work")
ap.add_argument("--radius", type=float); ap.add_argument("--centre", type=float, nargs=2)
ap.add_argument("--near-bodies", type=float); ap.add_argument("--max-bodies", type=int)
ap.add_argument("--name"); ap.add_argument("--open", default="/app/open")
a = ap.parse_args()
if (a.radius is None) == (a.near_bodies is None):
    ap.error("give one of --radius or --near-bodies")
name = a.name or a.sid + "_cut"
S = os.path.join(a.open, "scen", a.sid)
Z = np.load(os.path.join(a.open, "labels", a.sid + ".npz"))
body, night, known = Z["det_body"], Z["det_night"], json.loads(str(Z["known"]))
reg = json.load(open(os.path.join(S, "region.json")))
ra = np.load(os.path.join(S, "det_ra.npy"), mmap_mode="r"); dec = np.load(os.path.join(S, "det_dec.npy"), mmap_mode="r")
n = len(ra)


def unit(r, d):
    r, d = np.radians(np.asarray(r, dtype=np.float64)), np.radians(np.asarray(d, dtype=np.float64))
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], axis=-1)


keep = np.zeros(n, dtype=bool); CH = 4_000_000
if a.radius is not None:
    c = a.centre or reg["centre_radec"]; u0 = unit(c[0], c[1]); cosr = np.cos(np.radians(a.radius))
    for i in range(0, n, CH):
        keep[i:i + CH] = unit(ra[i:i + CH], dec[i:i + CH]) @ u0 >= cosr
    centre, radius = [float(c[0]), float(c[1])], float(a.radius)
else:
    from scipy.spatial import cKDTree
    find = [b for b, k in enumerate(known) if k["nights"] >= 5]
    if a.max_bodies:
        find = find[:a.max_bodies]
    rows = np.flatnonzero(np.isin(body, find))
    if not len(rows):
        raise SystemExit("this region has no findable labelled body")
    rows.sort()
    tree = cKDTree(unit(ra[rows], dec[rows])); chord = 2 * np.sin(np.radians(a.near_bodies) / 2)
    for i in range(0, n, CH):
        d, _ = tree.query(unit(ra[i:i + CH], dec[i:i + CH]), k=1, distance_upper_bound=chord, workers=-1)
        keep[i:i + CH] = np.isfinite(d)
    centre, radius = reg["centre_radec"], reg["radius_deg"]

idx = np.flatnonzero(keep)
out = os.path.join(a.work, "scen", name); os.makedirs(out, exist_ok=True); os.makedirs(os.path.join(a.work, "labels"), exist_ok=True)
for f in sorted(os.listdir(S)):
    if f.startswith("det_") and f.endswith(".npy"):
        np.save(os.path.join(out, f), np.load(os.path.join(S, f), mmap_mode="r")[idx])
np.save(os.path.join(out, "rows.npy"), idx)
with open(os.path.join(S, "exposures.json")) as fh, open(os.path.join(out, "exposures.json"), "w") as fo:
    fo.write(fh.read())
json.dump(dict(reg, sid=name, centre_radec=centre, radius_deg=radius, n_detections=int(len(idx)), cut_of=a.sid),
          open(os.path.join(out, "region.json"), "w"))
b2, n2 = body[idx], night[idx]
kn = [dict(k, nights=int(len(np.unique(n2[b2 == b])))) for b, k in enumerate(known)]      # name, H, weight stay
np.savez_compressed(os.path.join(a.work, "labels", name + ".npz"), det_body=b2, det_night=n2, known=json.dumps(kn), planted="[]")
fb = [b for b, k in enumerate(kn) if k["nights"] >= 5]
print(f"{name}: {len(idx):,} of {n:,} detections kept; findable labelled bodies {len(fb)} "
      f"(of {sum(1 for k in known if k['nights'] >= 5)} in the whole region), weight {sum(kn[b]['weight'] for b in fb)}")
print("scenario:", out)
print("labels:  ", os.path.join(a.work, "labels", name + ".npz"))
