"""python replay.py SETS... : replay vet_adjust+rank_balance on vet dumps with variants; score labelled (inj) sets.
SETS: R, Q (injected, labelled) and dev ids (unlabelled, pooled as competitors)."""
import sys, os, json, csv, io, contextlib, importlib
T = "$RSI_SANDBOX/r1/gen145/tmp/"
D = "$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev/"
SUB = "$RSI_SANDBOX/r1/gen145/workspace/submission"
sys.path.insert(0, SUB + "/linker"); sys.path.insert(0, SUB); sys.path.insert(0, os.path.dirname(__file__))
import score_inject as SI
import importlib.util as _iu
_sp = _iu.spec_from_file_location('subfinder', SUB + '/finder.py'); F = _iu.module_from_spec(_sp); _sp.loader.exec_module(F)
W = {"A": 1, "B": 2, "C": 4}
sets = sys.argv[1:] or ["R", "Q"]
data = {}
for s in sets:
    scen = T + "scen_inj" + s if len(s) == 1 else D + s
    rows = list(csv.DictReader(open(scen + "/exposures.csv")))
    vd = json.load(open(T + f"runs/vdump_{s}.json"))
    truth = json.load(open(scen + "/truth.json")) if len(s) == 1 else None
    lab = [None] * len(vd)
    if truth:
        _, res, _, _ = SI.score(truth, [dict(c, _i=i) for i, c in enumerate(vd)], rows)
        srt = sorted(range(len(vd)), key=lambda i: -float(vd[i]["p"]))
        for k, r in zip(srt, res):
            if r is not None:
                lab[k] = (r[0], truth["bodies"][r[0]]["cls"])
    data[s] = (rows, vd, lab)

def run(env, verbose=False):
    for k, v in env.items(): setattr(F, k, v)
    allr = []; per = {}
    for s, (rows, vd, lab) in data.items():
        F.set_expt(rows)
        cs = [dict(c, _i=i) for i, c in enumerate(vd)]
        with contextlib.redirect_stdout(io.StringIO()):
            out = F.rank_balance(F.vet_adjust(cs, [c["f"] for c in vd]))[:200]
        seen = set(); sc = 0; info = []
        for r, c in enumerate(out):
            l = lab[c["_i"]]
            allr.append((c["p"], (s, l[0]) if l else None, l[1] if l else "-", s))
            if l is not None:
                if r < 25 and l[0] not in seen: sc += W[l[1]]
                info.append((r, l[1] + ("d" if l[0] in seen else "")))
                seen.add(l[0])
        per[s] = sc
        if verbose and lab.count(None) < len(lab): print("   ", s, sc, [x for x in info if x[0] >= 15][:20])
    allr.sort(key=lambda x: -x[0]); seen = set(); g = 0
    N = 25 * len(data)
    for x in allr[:N]:
        if x[1] is not None and x[1] not in seen: g += W[x[2]]; seen.add(x[1])
    share = {s: sum(1 for x in allr[:N] if x[3] == s) for s in data}
    return g, per, share

for name, env in [("gen143", dict(PROM_CNN=-1e9, PROM_CF=0, RANK_VM=0)),
                  ("cnn>=3", dict(PROM_CNN=3.0, PROM_CF=0, RANK_VM=0)),
                  ("cnn>=3 cf", dict(PROM_CNN=3.0, PROM_CF=1, RANK_VM=0)),
                  ("cnn>=2 cf", dict(PROM_CNN=2.0, PROM_CF=1, RANK_VM=0)),
                  ("cnn>=4 cf", dict(PROM_CNN=4.0, PROM_CF=1, RANK_VM=0)),
                  ("vm1", dict(PROM_CNN=-1e9, PROM_CF=0, RANK_VM=1.0)),
                  ("vm0.5", dict(PROM_CNN=-1e9, PROM_CF=0, RANK_VM=0.5)),
                  ("cnn>=3 cf vm0.5", dict(PROM_CNN=3.0, PROM_CF=1, RANK_VM=0.5)),
                  ("cnn>=3 cf vm1", dict(PROM_CNN=3.0, PROM_CF=1, RANK_VM=1.0))]:
    print(name, run(env, verbose=("-v" in os.environ.get("V", ""))))
