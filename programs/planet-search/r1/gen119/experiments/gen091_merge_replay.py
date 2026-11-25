import sys, json, csv, math, os, io, contextlib
import numpy as np
ARM = "$RSI_SANDBOX/r1/gen091/tmp/armA"
TMP = "$RSI_SANDBOX/r1/gen091/tmp"
sys.path.insert(0, ARM); sys.path.insert(0, ARM + "/experiments")
import finder as F
import score_inject as SI
truth = json.load(open(TMP + "/inj7/truth.json"))
rows = list(csv.DictReader(open(TMP + "/inj7/exposures.csv")))
D = json.load(open(ARM + "/merge_dump.json"))
nt, no = np.array(D["night_t"]), np.array(D["night_o"])
t_all = np.array([float(r["mjd_mid"]) for r in rows])
o_all = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows])
nights_all = SI.night_of(t_all)

def run_variant(**kw):
    old = {k: getattr(F, k) for k in kw}
    for k, v in kw.items(): setattr(F, k, v)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            A = [dict(c) for c in D["A"]]; B = [dict(c) for c in D["B"]]
            out = F.rank_balance(F.merge(A, B, nt, no))[:200]
    finally:
        for k, v in old.items(): setattr(F, k, v)
    return out

def sc(out):
    cands, res, hr, dup = SI.score(truth, [dict(epoch_mjd=c["epoch_mjd"], state=c["state"], p=c["p"]) for c in out], rows, 2.0, 25)
    bodies = truth["bodies"]
    ft = {k: r for k, (r, p) in hr.items()}
    def cls(top):
        d = {}
        for j, r in ft.items():
            if r < top: d[bodies[j]["cls"]] = d.get(bodies[j]["cls"], 0) + 1
        return d
    duptop = lambda top: sum(1 for r in res[:top] if r is not None) - sum(1 for r in ft.values() if r < top)
    return dict(ft25=sum(r < 25 for r in ft.values()), ft50=sum(r < 50 for r in ft.values()),
                dup25=duptop(25), dup50=duptop(50), cls25=cls(25), cls50=cls(50), all=len(ft),
                ranks={bodies[j]["cls"] + str(j): r for j, r in sorted(ft.items(), key=lambda x: x[1]) if r < 50}), res

def nn_matrix(cs, allexp=True):
    if allexp:
        U = np.array([F.orbit.predict_dirs(np.array(c["state"], float), float(c["epoch_mjd"]), t_all, o_all) for c in cs])
    else:
        U = F.tracks(cs, nt, no)
    sep = np.degrees(np.arccos(np.clip(np.einsum("and,bnd->abn", U, U), -1, 1))) * 3600
    return sep

def pairs(cs, tol, allexp=True):
    """truth-free: # candidates within tol on >= 3 distinct nights of a higher-ranked one"""
    sep = nn_matrix(cs, allexp)
    n = 0
    for j in range(1, len(cs)):
        ok = sep[:j, j] < tol
        if allexp:
            nn = max(len(set(nights_all[ok[k]].tolist())) for k in range(j))
        else:
            nn = ok.sum(1).max()
        if nn >= 3: n += 1
    return n
