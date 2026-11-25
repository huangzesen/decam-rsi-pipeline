"""gen134 analysis: label vet-dump rows vs injected truth, optionally re-measure vet features with the workspace vet.py,
and replay p policies (finder.vet_adjust + rank_balance) -> per-scenario top-25 and global top-25N weighted score.
usage: python ana.py remeasure TAG SCEN...   |   python ana.py eval TAG SCEN...
"""
import json, math, sys, os, csv, io, contextlib, copy
import numpy as np
T = "$RSI_SANDBOX/r1/gen134/tmp"
SUB = "$RSI_SANDBOX/r1/gen134/workspace/submission"
sys.path.insert(0, SUB); sys.path.insert(0, SUB + "/linker")
import orbit
ARC = math.pi / 180 / 3600
W = {"A": 1, "B": 2, "C": 4}


def label(scen, rows_):
    truth = json.load(open(f"{T}/scen/{scen}/truth.json"))
    ex = list(csv.DictReader(open(f"{T}/scen/{scen}/exposures.csv")))
    t = np.array([float(r["mjd_mid"]) for r in ex]); obs = np.array([[float(r[k]) for k in ("obs_x", "obs_y", "obs_z")] for r in ex])
    nights = np.floor(t + 0.3).astype(int)
    bodies = truth["bodies"]
    U = np.array([orbit.predict_dirs(np.array(b["state"]), b["epoch_mjd"], t, obs) for b in bodies])
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


def remeasure(tag, scen):
    import vet
    rows_ = json.load(open(f"{T}/runs/v_{tag}_{scen}.json"))
    fs = vet.measure(f"{T}/scen/{scen}", rows_, workers=3)
    for r, f in zip(rows_, fs):
        r["f"] = f
    json.dump(rows_, open(f"{T}/runs/w_{tag}_{scen}.json", "w"))


def wsub(f, mode):
    """mode 'old' = gen133 features; 'w' = matched-filter weighted features substituted."""
    if f is None or mode == "old" or "wzlbno" not in f:
        return f
    g = dict(f)
    zn = [x for x in f["wzn"] if x is not None]
    g["zon"] = f["wzon"]; g["zlbno"] = f["wzlbno"]; g["npos"] = f["wnpos"]
    if len(zn) >= 2:
        g["chi"] = float(np.var(zn, ddof=1))
    return g


def evaluate(data, mode="old", env=None, verbose=False, kind="w"):
    import importlib.util
    global _F
    if "_F" not in globals():
        sp = importlib.util.spec_from_file_location("wfinder", SUB + "/finder.py"); _F = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(_F)
    F = _F
    for k, v in (env or {}).items():
        setattr(F, k, v)
    allr = []; per = {}
    for s, (rows_, labs, nb, exr) in data.items():
        F.set_expt(exr)
        cs = [dict(c, _i=i) for i, c in enumerate(rows_)]
        fs = [wsub(c["f"], mode) for c in rows_]
        with contextlib.redirect_stdout(io.StringIO()):
            out = F.rank_balance(F.vet_adjust(cs, fs) if F.VET else cs)[:200]
        seen = set(); sc = 0; info = []
        for r, c in enumerate(out):
            l = labs[c["_i"]]
            allr.append((c["p"], (s, l[0]) if l else None, l[1] if l else "-"))
            if l is not None:
                if r < 25 and l[0] not in seen:
                    sc += W[l[1]]
                info.append((r, l[1] + ("dup" if l[0] in seen else "")))
                seen.add(l[0])
        per[s] = sc
        if verbose:
            print("  ", s, sc, [x for x in info if x[0] < 80])
    allr.sort(key=lambda x: -x[0]); seen = set(); g = 0
    for x in allr[:25 * len(data)]:
        if x[1] is not None and x[1] not in seen:
            g += W[x[2]]; seen.add(x[1])
    for k, v in (env or {}).items():
        pass
    return g, per


def load(tag, scens, kind="w"):
    data = {}
    for s in scens:
        rows_ = json.load(open(f"{T}/runs/{kind}_{tag}_{s}.json"))
        labs, nb = label(s, rows_)
        exr = list(csv.DictReader(open(f"{T}/scen/{s}/exposures.csv")))
        data[s] = (rows_, labs, nb, exr)
    return data


if __name__ == "__main__":
    cmd, tag = sys.argv[1:3]
    scens = sys.argv[3:]
    if cmd == "remeasure":
        for s in scens:
            remeasure(tag, s)
    elif cmd == "feats":
        data = load(tag, scens)
        for s, (rows_, labs, nb, exr) in data.items():
            print(s, nb)
            for i, (c, l) in enumerate(zip(rows_, labs)):
                f = c["f"] or {}
                if l is not None or i < 30:
                    print(f"{i:3d} {c['src'] or '?':3s} p {c['p']:.3f} {('-' if l is None else l[1]):2s} zon {f.get('zon', 0):6.2f}/{f.get('wzon', 0):6.2f} "
                          f"zlbno {f.get('zlbno', 0):6.2f}/{f.get('wzlbno', 0):6.2f} zoff {f.get('zoff', 0):5.2f} chi {f.get('chi', 0):7.2f} np {f.get('npos')}/{f.get('wnpos')}")
    else:
        data = load(tag, scens)
        for m in (0, 1, 2):
            print("WF", m, evaluate(data, "old", env=dict(VET_WF=m), verbose=True))
