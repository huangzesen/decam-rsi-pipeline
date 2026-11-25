#!/usr/bin/env python3
"""archive5: positions of known TNOs in a region's exposures, for recall (Mac; light work).

Two-body propagation of every JPL SBDB TNO (linker/data_real/sbdb/) at each exposure (light time included) picks the
bodies within 1.15 deg of an exposure centre with H <= --hmax; JPL Horizons (site W84) then gives their positions in
those exposures. Writes archive5/link/known/<rid>.json: {"rid", "bodies": [{name, pdes, H, first_obs, pos: {stem: [ra, dec]}}]}.
Usage: python3 -I known_pos.py REG_A,REG_B [--hmax 8] [--hmin H --out known_faint]   (fainter bodies into their own folder)
"""
import argparse, glob, json, os, sys, time, urllib.parse, urllib.request
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); A5 = os.path.dirname(HERE); PS = os.path.dirname(A5)
sys.path.insert(0, os.path.join(PS, "linker"))
from orbits import elements_to_rv, ecl_to_eq, GM, C_AU
from ed138 import observer

ap = argparse.ArgumentParser()
ap.add_argument("rids"); ap.add_argument("--hmax", type=float, default=8.0)
ap.add_argument("--hmin", type=float, default=-99.0)     # with --out: only bodies fainter than this (the extension of 2026-11-14)
ap.add_argument("--out", default="known")                 # output folder next to this script
a = ap.parse_args()
rexp = json.load(open(os.path.join(A5, "regions_exposures.json")))
want = set(s for r in a.rids.split(",") for s in rexp[r])
meta = {}
for f in sorted(glob.glob(os.path.join(PS, "linker", "data_real", "noirlab_decam_meta", "decam_meta_*.json"))):
    d = json.load(open(f)); ix = {k: i for i, k in enumerate(d["fields"])}
    for r in d["rows"]:
        if r[ix["stamp"]] in want:
            t = datetime.fromisoformat(r[ix["dateobs"]].replace("Z", "+00:00"))
            mjd = (t.timestamp() / 86400 + 40587.0) + r[ix["exptime"]] / 2 / 86400
            meta[r[ix["stamp"]]] = (mjd, r[ix["ra"]], r[ix["dec"]])
S = json.load(open(os.path.join(PS, "linker", "data_real", "sbdb", "tno_20261031.json")))
rows = [dict(zip(S["fields"], r)) for r in S["data"]]
rows = [r for r in rows if all(r[k] not in (None, "") for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H"))
        and float(r["e"]) < 1 and a.hmin < float(r["H"]) <= a.hmax]
el = {k: np.array([float(r[k]) for r in rows]) for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H")}
mjd0 = el["epoch"] - 2400000.5; nmot = np.sqrt(GM / el["a"] ** 3)
print(f"{len(rows)} TNOs with {a.hmin} < H <= {a.hmax}", flush=True)


def place(t, xe):
    """equatorial unit vectors of all bodies seen from xe at time t (one light-time iteration)"""
    M = np.radians(el["ma"]) + nmot * (t - mjd0)
    r = ecl_to_eq(elements_to_rv(el["a"], el["e"], np.radians(el["i"]), np.radians(el["om"]), np.radians(el["w"]), M)[0])
    lt = np.linalg.norm(r - xe, axis=1) / C_AU
    M = np.radians(el["ma"]) + nmot * (t - lt - mjd0)
    r = ecl_to_eq(elements_to_rv(el["a"], el["e"], np.radians(el["i"]), np.radians(el["om"]), np.radians(el["w"]), M)[0])
    g = r - xe
    return g / np.linalg.norm(g, axis=1)[:, None]


def horizons(pdes, mjds):
    """positions from W84; tries DES=<pdes>; then <pdes>; and returns (positions, None) or (None, Horizons' message)"""
    for cmd in (f"'DES={pdes};'", f"'{pdes};'"):
        out = []; err = None
        for i in range(0, len(mjds), 40):
            tl = " ".join(f"'{m + 2400000.5:.6f}'" for m in mjds[i:i + 40])
            q = urllib.parse.urlencode(dict(format="json", COMMAND=cmd, EPHEM_TYPE="OBSERVER", CENTER="'W84@399'", TLIST=tl,
                                            TLIST_TYPE="JD", QUANTITIES="'1'", ANG_FORMAT="DEG", CSV_FORMAT="YES", TIME_TYPE="UT"))
            txt = None
            for k in range(4):
                try:
                    txt = json.loads(urllib.request.urlopen(f"https://ssd.jpl.nasa.gov/api/horizons.api?{q}", timeout=300).read())["result"]
                    break
                except Exception as e:
                    txt = None; err = repr(e)[:200]; time.sleep(10 * (k + 1))
            if txt is None or "$$SOE" not in txt:
                err = err if txt is None else " ".join(txt.split()[-40:])[:300]
                out = None; break
            body = txt.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
            out += [(float(l.split(",")[3]), float(l.split(",")[4])) for l in body]
            time.sleep(1)
        if out is not None:
            return out, None
    return None, err


os.makedirs(os.path.join(HERE, a.out), exist_ok=True)
cosr = np.cos(np.radians(1.15))
for rid in a.rids.split(","):
    st = [s for s in rexp[rid] if s in meta]
    T = np.array([meta[s][0] for s in st])
    xe = observer(T)
    ue = np.array([[np.cos(np.radians(meta[s][2])) * np.cos(np.radians(meta[s][1])), np.cos(np.radians(meta[s][2])) * np.sin(np.radians(meta[s][1])),
                    np.sin(np.radians(meta[s][2]))] for s in st])
    hits = {}
    for k in range(len(st)):
        inside = np.flatnonzero(place(T[k], xe[k]) @ ue[k] > cosr)
        for b in inside:
            hits.setdefault(int(b), []).append(k)
    bodies = []
    for b, ks in sorted(hits.items(), key=lambda x: el["H"][x[0]]):
        pos, err = horizons(rows[b]["pdes"], [T[k] for k in ks])
        if pos is None:
            print(rid, rows[b]["pdes"], "skipped: Horizons", err, flush=True); continue
        if len(pos) != len(ks):
            print(rid, rows[b]["pdes"], "Horizons returned", len(pos), "of", len(ks)); continue
        bodies.append(dict(name=rows[b]["full_name"].strip(), pdes=rows[b]["pdes"], H=float(el["H"][b]), first_obs=rows[b]["first_obs"],
                           pos={st[k]: [round(p[0], 7), round(p[1], 7)] for k, p in zip(ks, pos)}))
        print(rid, rows[b]["pdes"], f"H {el['H'][b]:.2f}", len(ks), "exposures", flush=True)
    json.dump(dict(rid=rid, hmax=a.hmax, hmin=a.hmin, bodies=bodies), open(os.path.join(HERE, a.out, rid + ".json"), "w"))
    print(rid, len(bodies), "known bodies", flush=True)
