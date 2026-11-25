"""python rb_replay.py TAG:SCEN ... : replay vet_adjust + rank_balance on vet dumps (runs/vdump_TAG.json) with TNO_RBSKIP
variants.  SCEN = injection letter (labelled, scen_injX) or a dev id (unlabelled; all rows count as junk competitors).
Metric: weighted hits (C=4) in the pooled global top 25 x (number of sets); per-set top-25 hits; thinning (keep each
body with prob 0.3, its rows removed) over seeds."""
import sys, os, json, csv, io, contextlib, itertools, random
T = "$RSI_SANDBOX/r1/gen151/tmp/"
D = "$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev/"
SUB = os.environ.get("SUB", "$RSI_SANDBOX/r1/gen151/workspace/submission")
sys.path.insert(0, SUB + "/linker"); sys.path.insert(0, SUB); sys.path.insert(0, os.path.dirname(__file__))
import score_inject as SI
import importlib.util as _iu
_sp = _iu.spec_from_file_location('subfinder', SUB + '/finder.py'); F = _iu.module_from_spec(_sp); _sp.loader.exec_module(F)
W = {"A": 1, "B": 2, "C": 4}
data = {}
for arg in sys.argv[1:]:
    if arg.startswith("-"):
        continue
    tag, s = arg.split(":")
    scen = T + "scen_inj" + s if len(s) == 1 or s[0].isupper() else D + s
    rows = list(csv.DictReader(open(scen + "/exposures.csv")))
    vd = json.load(open(T + f"runs/vdump_{tag}.json"))
    truth = json.load(open(scen + "/truth.json")) if os.path.exists(scen + "/truth.json") else None
    lab = [None] * len(vd)
    if truth:
        _, res, _, _ = SI.score(truth, [dict(c, _i=i) for i, c in enumerate(vd)], rows)
        srt = sorted(range(len(vd)), key=lambda i: -float(vd[i]["p"]))
        for k, r in zip(srt, res):
            if r is not None:
                lab[k] = (r[0], truth["bodies"][r[0]]["cls"])
    data[tag] = (rows, vd, lab)
    print(tag, s, "rows", len(vd), "labelled rows", sum(l is not None for l in lab),
          "bodies", len({l[0] for l in lab if l}), file=sys.stderr)


def run(env, keep=None, verbose=False):
    for k, v in env.items(): setattr(F, k, v)
    allr = []; per = {}
    for s, (rows, vd, lab) in data.items():
        F.set_expt(rows)
        idx = [i for i in range(len(vd)) if keep is None or lab[i] is None or (s, lab[i][0]) in keep]
        cs = [dict(vd[i], _i=i) for i in idx]
        with contextlib.redirect_stdout(io.StringIO()):
            out = F.rank_balance(F.vet_adjust(cs, [vd[i]["f"] for i in idx]))[:200]
        seen = set(); sc = 0; info = []
        for r, c in enumerate(out):
            l = lab[c["_i"]]
            allr.append((c["p"], (s, l[0]) if l else None, l[1] if l else "-", s))
            if l is not None:
                if r < 25 and l[0] not in seen: sc += W[l[1]]
                info.append((r, round(c["p"], 3)))
                seen.add(l[0])
        per[s] = sc
        if verbose and any(lab): print("   ", s, sc, info[:40])
    allr.sort(key=lambda x: -x[0]); seen = set(); g = 0
    N = 25 * len(data)
    for x in allr[:N]:
        if x[1] is not None and x[1] not in seen: g += W[x[2]]; seen.add(x[1])
    share = {s: sum(1 for x in allr[:N] if x[3] == s) for s in data}
    return g, per, share


def diag():
    """vet features of labelled rows at final rank >= 15 vs junk rows above them."""
    for s, (rows, vd, lab) in data.items():
        if not any(lab): continue
        F.set_expt(rows)
        cs = [dict(c, _i=i) for i, c in enumerate(vd)]
        with contextlib.redirect_stdout(io.StringIO()):
            out = F.rank_balance(F.vet_adjust(cs, [c["f"] for c in vd]))
        print("==", s)
        for r, c in enumerate(out[:120]):
            f = c.get("vet") or {}
            l = lab[c["_i"]]
            try:
                cr = F._credible(c)
            except Exception:
                cr = "err"
            g = lambda k: (round(float(f[k]), 2) if k in f and f[k] is not None else None)
            print(f"{r:4d} p {c['p']:.3f} {c.get('src')} {'BODY' if l else '    '} zlbno {g('zlbno')} zoff {g('zoff')} "
                  f"zon {g('zon')} chi {g('chi')} npos {f.get('npos')} cnn {g('cnn')} cred {cr}")


if "--diag" in sys.argv:
    diag(); sys.exit()
base = dict(RB_SKIP=0.0, RB_Z=3.0, RB_C=-1e9, RB_MINF=0.5, RB_RMAX=120, RB_DS=0.0)
grid = [dict(base)]
for z, c, mf, rm, ds in itertools.product([3.0, 3.5], [-1e9, 0.0], [0.5, 0.6, 0.65], [60, 120], [0.0, 1.0, 2.0]):
    grid.append(dict(base, RB_SKIP=1.0, RB_Z=z, RB_C=c, RB_MINF=mf, RB_RMAX=rm, RB_DS=ds))
bodies = sorted({(s, l[0]) for s, (_, _, lab) in data.items() for l in lab if l})
rngs = [random.Random(k) for k in range(10)]
keeps = [{b for b in bodies if r.random() < 0.3} for r in rngs]
res = []
for env in grid:
    g, per, share = run(env)
    th = [run(env, keep=k)[0] for k in keeps]
    res.append((env, g, per, share, th))
b0 = res[0]
print("base", b0[1], b0[2], b0[3], "thin", b0[4])
for env, g, per, share, th in res[1:]:
    d = [a - b for a, b in zip(th, b0[4])]
    print(f"z {env['RB_Z']} c {env['RB_C']:.0f} minf {env['RB_MINF']} rmax {env['RB_RMAX']} ds {env['RB_DS']}: global {g - b0[1]:+d} "
          f"per {per} share {share} thin sum {sum(d):+d} min {min(d):+d}")
if "-v" in sys.argv:
    run(base, verbose=True)
