"""sweep: cached variant of exp/sim.py. Hits cached by candidate state; post-process calibrations; heterogeneous q."""
import sys, os, json, math, io, contextlib, pickle
import numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen114/tmp/exp")
import sim  # noqa
si = sim.si
CACHE = "$RSI_SANDBOX/r1/gen114/tmp/sweep/hitcache.pkl"
_hc = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
_base = {}
_F = None
_scen = {}


def F():
    global _F
    if _F is None:
        _F = sim.load_finder()
        for k in dir(_F):
            if k.isupper():
                _base[k] = getattr(_F, k)
        _F.log = lambda *a: None
    return _F


def scen_data(name, scen, run):
    if name not in _scen:
        rows, nt, no = sim.nights(scen)
        A = json.load(open(run + ".linker.json"))["candidates"]
        S = json.load(open(run + ".stack.json"))["candidates"]
        _scen[name] = (rows, nt, no, A, S)
    return _scen[name]


def key(c):
    return (name_, round(float(c["epoch_mjd"]), 6)) + tuple(round(float(x), 10) for x in c["state"])


def hits(name, T, cs, rows):
    global name_
    name_ = name
    miss = [c for c in cs if key(c) not in _hc]
    if miss:
        _, res, _, _ = si.score(T, miss, rows)
        # score sorts by p; recompute mapping by re-scoring individually sorted list: use same order
        srt = sorted(miss, key=lambda c: -float(c.get("p", 0)))
        for c, r in zip(srt, res):
            _hc[key(c)] = r[0] if r is not None else -1
    return np.array([_hc[key(c)] for c in cs])


def save():
    pickle.dump(_hc, open(CACHE, "wb"))


def prepare(variant, rb=True):
    """returns per-scenario dict with p (final, after rank_balance if rb), praw (before rank_balance), src, hit."""
    f = F()
    for k, v in _base.items():
        setattr(f, k, v)
    for k, v in variant.items():
        setattr(f, k, v)
    out = []
    for name, scen, run, truth in sim.SCEN:
        if not (os.path.exists(run + ".linker.json") and os.path.exists(run + ".stack.json")):
            continue
        try:
            rows, nt, no, A, S = scen_data(name, scen, run)
        except Exception:
            continue
        cs = f.merge([dict(c) for c in A], [dict(c) for c in S], nt, no)
        cs = sorted(cs, key=lambda c: -float(c["p"]))
        praw = np.array([float(c["p"]) for c in cs])
        cs = (f.rank_balance(cs) if rb else cs)[:200]
        praw = praw[:len(cs)]
        if truth:
            T = json.load(open(truth))
            hit = hits(name, T, cs, rows)
            w = np.array([sim.W[b["cls"]] if len(b["nights"]) >= 3 else 0 for b in T["bodies"]], float)
            cls = [b["cls"] for b in T["bodies"]]
        else:
            hit = np.full(len(cs), -1); w = np.zeros(0); cls = []
        out.append(dict(name=name, p=np.array([float(c["p"]) for c in cs]), praw=praw,
                        src=[c.get("src") for c in cs], hit=hit, w=w, cls=cls, truth=truth is not None))
    return out


def gscore(P, qs, reps=300, seed=0, slots=25, nscen=None):
    """qs: dict name->q (default 1). Returns mean weighted hits (weight units), mean recall, per-class."""
    rng = np.random.default_rng(seed)
    n = slots * (nscen or len(P))
    WU = []; R = []; cl = {"A": 0., "B": 0., "C": 0.}
    for _ in range(reps):
        ps = []; ks = []; hs = []; den = 0.
        for k, s in enumerate(P):
            if s["truth"]:
                q = qs.get(s["name"], 1.0)
                kb = rng.random(len(s["w"])) < q
                den += (s["w"] * kb).sum()
                m = (s["hit"] < 0) | kb[np.maximum(s["hit"], 0)]
            else:
                m = np.ones(len(s["p"]), bool)
            idx = np.flatnonzero(m)
            ps.append(s["p"][idx]); ks.append(np.full(len(idx), k)); hs.append(s["hit"][idx])
        p = np.concatenate(ps); kk = np.concatenate(ks); hh = np.concatenate(hs)
        o = np.argsort(-p, kind="stable")[:n]
        got = {(int(kk[i]), int(hh[i])) for i in o if hh[i] >= 0}
        wu = sum(P[k]["w"][h] for k, h in got)
        WU.append(wu); R.append(wu / max(den, 1e-9))
        for k, h in got:
            cl[P[k]["cls"][h]] += 1 / reps
    return float(np.mean(WU)), float(np.mean(R)), cl
