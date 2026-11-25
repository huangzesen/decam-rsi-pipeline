"""Offline multi-scenario simulation of the wrapper's merge + global top-25xN cut (gen 114).

Each run R has R.json.linker.json and R.json.stack.json (TNO_KEEP=1).  Scenarios with a truth.json are scored;
scenarios without truth contribute their candidates as (unknown) competitors in the global cut.

python sim.py  [--variant k=v,k=v ...]  --q 0.3 --reps 200
Variants set module attributes of the wrapper (finder.py), e.g. RANK_Q0=0.5,RANK_R=15,STACK_W=0.6.
"""
import argparse, csv, json, math, os, sys, importlib.util, io, contextlib
import numpy as np

G = "$RSI_SANDBOX/r1/gen114/tmp"
sys.path.insert(0, G + "/exp")
import score_inject as si  # noqa

D = "$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev"
SCEN = [  # (name, scenario dir, run prefix, truth or None)
    ("inj7d", G + "/inj7d", G + "/runs/b_inj7d.json", G + "/inj7d/truth.json"),
    ("injF", G + "/inj_s1060fab2", G + "/runs/b_injF.json", G + "/inj_s1060fab2/truth.json"),
    ("injD", G + "/inj_s149133df", G + "/runs/b_injD.json", G + "/inj_s149133df/truth.json"),
    ("d1c", D + "/scen04", G + "/runs/b_d1c.json", None),
    ("d106", D + "/scen01", G + "/runs/b_d106.json", None),
    ("d149", D + "/scen02", G + "/runs/b_d149.json", None),
    ("d18b", D + "/scen03", G + "/runs/b_d18b.json", None),
]
W = {"A": 1, "B": 2, "C": 4}


def load_finder(path=G + "/base/submission/finder.py"):
    spec = importlib.util.spec_from_file_location("wrapfinder", path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path) + "/linker")
    spec.loader.exec_module(m)
    return m


def nights(scen):
    rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
    nt = {}
    for r in rows:
        t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
        nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
    return rows, np.array([v[0] for v in nt.values()]), np.array([v[1] for v in nt.values()])


def final_list(F, A, S, nt, no, variant):
    for k, v in variant.items():
        setattr(F, k, v)
    with contextlib.redirect_stdout(io.StringIO()):
        cs = F.merge([dict(c) for c in A], [dict(c) for c in S], nt, no)
        cs = F.rank_balance(cs)[:200]
    return cs


def prepare(F, variant, only=None):
    """-> list of dicts: name, cands (sorted by p), hit body idx per cand (-1), bodies' weight array."""
    out = []
    for name, scen, run, truth in SCEN:
        if only and name not in only:
            continue
        log_ = run[:-5] + ".log"   # unfinished runs have a placeholder stack.json: require the wrapper's 'final' line
        if not (os.path.exists(run + ".linker.json") and os.path.exists(run + ".stack.json")
                and os.path.exists(log_) and " final " in open(log_).read()):
            continue
        rows, nt, no = nights(scen)
        A = json.load(open(run + ".linker.json"))["candidates"]
        S = json.load(open(run + ".stack.json"))["candidates"]
        cs = final_list(F, A, S, nt, no, variant)
        hit = np.full(len(cs), -1); wts = np.zeros(0); cls = []
        if truth:
            T = json.load(open(truth))
            cs_sorted, res, _, _ = si.score(T, cs, rows)
            cs = cs_sorted
            hit = np.array([h[0] if h is not None else -1 for h in res])
            wts = np.array([W[b["cls"]] if len(b["nights"]) >= 3 else 0 for b in T["bodies"]], float)
            cls = [b["cls"] for b in T["bodies"]]
        out.append(dict(name=name, p=np.array([float(c["p"]) for c in cs]), hit=hit, w=wts, cls=cls, truth=truth is not None))
    return out


def global_score(P, q=1.0, reps=1, seed=0, slots=25):
    """Bodies are kept with probability q (dropped bodies' candidates removed: they would not exist).
    Returns mean weighted recall over truth scenarios, and mean per-class hits."""
    rng = np.random.default_rng(seed)
    tot = []; cl = {"A": [], "B": [], "C": []}
    for _ in range(reps):
        pool = []; keep_b = {}
        for k, s in enumerate(P):
            if s["truth"]:
                kb = rng.random(len(s["w"])) < q if q < 1 else np.ones(len(s["w"]), bool)
                keep_b[k] = kb
                m = (s["hit"] < 0) | kb[np.maximum(s["hit"], 0)]
            else:
                m = np.ones(len(s["p"]), bool)
            for i in np.flatnonzero(m):
                pool.append((s["p"][i], k, int(s["hit"][i])))
        pool.sort(key=lambda x: -x[0])
        n = slots * len(P)
        got = set()
        for p, k, h in pool[:n]:
            if h >= 0:
                got.add((k, h))
        num = sum(P[k]["w"][h] for k, h in got)
        den = sum((P[k]["w"] * keep_b[k]).sum() for k in keep_b)
        tot.append(num / max(den, 1e-9))
        for c in cl:
            cl[c].append(sum(1 for k, h in got if P[k]["cls"][h] == c))
    return float(np.mean(tot)), {c: float(np.mean(v)) for c, v in cl.items()}


def parse_variant(s):
    v = {}
    if s:
        for kv in s.split(","):
            k, x = kv.split("=")
            v[k] = float(x) if "." in x or "e" in x else int(x)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="*", default=[""])
    ap.add_argument("--q", type=float, nargs="*", default=[1.0, 0.3, 0.1])
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    for vs in a.variant:
        F = load_finder()
        v = parse_variant(vs)
        P = prepare(F, v)
        line = f"{vs or 'baseline':40s} scen {[s['name'] for s in P]}"
        for q in a.q:
            r, c = global_score(P, q, reps=1 if q >= 1 else a.reps)
            line += f" | q={q}: {r:.4f} A{c['A']:.2f} B{c['B']:.2f} C{c['C']:.2f}"
        print(line, flush=True)
