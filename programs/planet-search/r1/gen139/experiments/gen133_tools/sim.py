"""gen133 offline policy sim over vet dumps. python sim.py RUNTAG [scen...]
Loads runs/v_{RUNTAG}_{scen}.json (merged rows with vet features, finder p), labels rows vs truth, applies policies."""
import json, math, sys, os, csv
import numpy as np
T = "$RSI_SANDBOX/r1/gen133/tmp"
sys.path.insert(0, T + "/base/linker")
import orbit
ARC = math.pi / 180 / 3600
W = {"A": 1, "B": 2, "C": 4}


def label(scen, rows_):
    truth = json.load(open(f"{T}/scen/{scen}/truth.json"))
    ex = list(csv.DictReader(open(f"{T}/scen/{scen}/exposures.csv")))
    t = np.array([float(r["mjd_mid"]) for r in ex]); obs = np.array([[float(r[k]) for k in ("obs_x", "obs_y", "obs_z")] for r in ex])
    nights = np.floor(t + 0.3).astype(int)
    bodies = truth["bodies"]
    U = np.array([orbit.predict_dirs(b["state"], b["epoch_mjd"], t, obs) for b in bodies])
    labs = []
    for c in rows_:
        d = orbit.predict_dirs(np.array(c["state"]), float(c["epoch_mjd"]), t, obs)
        sep = np.arccos(np.clip(np.einsum("bni,ni->bn", U, d), -1, 1)) / ARC
        best = None
        for j, b in enumerate(bodies):
            e = np.array(b["exps"], int)
            if len(e) == 0 or len(b["nights"]) < 3:
                continue
            ok = e[sep[j, e] < 2.0]
            if len(set(nights[ok].tolist())) >= 3 and (best is None or np.median(sep[j, e]) < best[1]):
                best = (j, float(np.median(sep[j, e])))
        labs.append(None if best is None else (best[0], bodies[best[0]]["cls"]))
    nb = {c: sum(1 for b in bodies if b["cls"] == c and len(b["nights"]) >= 3) for c in W}
    return labs, nb


def rb(ps):
    o = np.argsort(-ps, kind="stable"); out = ps.copy()
    for r, i in enumerate(o):
        out[i] = 1 - (1 - ps[i]) * (1 - 0.7 * math.exp(-r / 25))
    return out


def load(tag, scen):
    rows_ = json.load(open(f"{T}/runs/v_{tag}_{scen}.json"))
    labs, nb = label(scen, rows_)
    return rows_, labs, nb


def evaluate(data, policy, verbose=False):
    allr = []; per = {}
    for scen, (rows_, labs, nb) in data.items():
        q = np.array(policy(rows_), float)
        q = rb(q)
        o = np.argsort(-q, kind="stable")[:200]
        seen = set(); sc = 0; info = []
        for r, i in enumerate(o):
            l = labs[i]
            key = None
            if l is not None:
                key = (scen, l[0])
            allr.append((q[i], scen, key, l[1] if l else "-", r))
            if r < 25 and l is not None and l[0] not in seen:
                sc += W[l[1]]
            if l is not None:
                if l[0] not in seen:
                    info.append((r, l[1]))
                else:
                    info.append((r, l[1] + "dup"))
                seen.add(l[0])
        per[scen] = sc
        if verbose:
            print(f"  {scen}: top25 weighted {sc}; hits {[x for x in info if x[0] < 60]}")
    allr.sort(key=lambda x: -x[0])
    seen = set(); g = 0
    for x in allr[:25 * len(data)]:
        if x[2] is not None and x[2] not in seen:
            g += W[x[3]]; seen.add(x[2])
    return g, per


def f_(c, k, d=0.0):
    f = c.get("f")
    if not f:
        return d
    return float(f.get(k, d))
