#!/usr/bin/env python3
"""Fill each exposure's pointing (ra0, dec0) and the region's centre and radius from the detections themselves.

The archive5 detection lists carry NaN for ra0/dec0 (their header parse failed), so build_scen.py wrote NaN into
exposures.json and region.json (found 2026-11-12: the starter used region.json's radius as its search radius and so kept
no pair). The pointing is now the mean direction of the exposure's detections; the region centre the mean of those, and
radius_deg = the largest distance of an exposure centre from it + 1.1 deg, as build_scen.py intended. Detection rows and
truth are untouched. Usage: python3 -I fix_pointing.py SCEN_DIR [SCEN_DIR ...]
"""
import json, os, sys
import numpy as np


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


for S in sys.argv[1:]:
    mode = os.stat(S).st_mode & 0o777; os.chmod(S, mode | 0o200)       # scenario folders are read-only
    E = json.load(open(os.path.join(S, "exposures.json"))); R = json.load(open(os.path.join(S, "region.json")))
    ex = np.load(os.path.join(S, "det_exp.npy"), mmap_mode="r")
    acc = np.zeros((len(E), 3))
    n = len(ex); step = 1 << 24
    for i in range(0, n, step):
        e = np.asarray(ex[i:i + step]); v = unit(np.load(os.path.join(S, "det_ra.npy"), mmap_mode="r")[i:i + step],
                                                 np.load(os.path.join(S, "det_dec.npy"), mmap_mode="r")[i:i + step])
        for k in range(3):
            acc[:, k] += np.bincount(e, weights=v[:, k], minlength=len(E))
    u = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    for k, m in enumerate(E):
        m["ra0"] = round(float(np.degrees(np.arctan2(u[k, 1], u[k, 0])) % 360), 5); m["dec0"] = round(float(np.degrees(np.arcsin(u[k, 2]))), 5)
    c = u.mean(0); c /= np.linalg.norm(c)
    rad = float(np.degrees(np.arccos(np.clip(u @ c, -1, 1))).max())
    R["centre_radec"] = [round(float(np.degrees(np.arctan2(c[1], c[0])) % 360), 5), round(float(np.degrees(np.arcsin(c[2]))), 5)]
    R["radius_deg"] = round(rad + 1.1, 2)
    for f, obj in (("exposures.json", E), ("region.json", R)):
        p = os.path.join(S, f)
        os.chmod(p, 0o644) if not os.access(p, os.W_OK) else None
        tmp = p + ".tmp"; json.dump(obj, open(tmp, "w")); os.replace(tmp, p); os.chmod(p, 0o444)
    os.chmod(S, mode)
    print(S, "centre", R["centre_radec"], "radius_deg", R["radius_deg"], "exposures", len(E))
