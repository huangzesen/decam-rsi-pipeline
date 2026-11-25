import sys, pickle, importlib, io, contextlib, itertools
from sim import *
data = pickle.load(open(T+"/runs/r1data.pkl","rb"))
SUB = "$RSI_SANDBOX/r1/gen133/workspace/submission"
sys.path.insert(0, SUB)
import finder as F
EXR = {s: list(csv.DictReader(open(f"{T}/scen/{s}/exposures.csv"))) for s in data}
def run(env, verbose=False, nvet=600):
    for k, v in env.items(): setattr(F, k, v)
    allr = []; per = {}
    for s, (rows_, labs, nb) in data.items():
        F.set_expt(EXR[s])
        cs = [dict(c, _i=i) for i, c in enumerate(rows_)]
        fs = [c["f"] for c in rows_]
        with contextlib.redirect_stdout(io.StringIO()):
            head = F.vet_adjust(cs[:nvet], fs[:nvet]) if F.VET else cs[:nvet]
            out = F.rank_balance(head + cs[nvet:])[:200]
        seen = set(); sc = 0; info = []
        for r, c in enumerate(out):
            l = labs[c["_i"]]
            allr.append((c["p"], (s, l[0]) if l else None, l[1] if l else "-"))
            if l is not None:
                if r < 25 and l[0] not in seen: sc += W[l[1]]
                info.append((r, l[1] + ("dup" if l[0] in seen else "")))
                seen.add(l[0])
        per[s] = sc
        if verbose: print("  ", s, sc, [x for x in info if x[0] >= 10 and x[0] < 80], "top25 '-' srcs:", "".join((c["src"] or "?")[0] if labs[c["_i"]] is None else "." for c in out[:30]))
    allr.sort(key=lambda x: -x[0]); seen = set(); g = 0
    for x in allr[:75]:
        if x[1] is not None and x[1] not in seen: g += W[x[2]]; seen.add(x[1])
    return g, per
print("gen129 (no vet)", run(dict(VET=0, PROM=0), True, 200))
print("gen132", run(dict(VET=1, PROM=0), True, 200))
print("gen132 vet600", run(dict(VET=1, PROM=0), True, 600))
for z0, z1, p0, p1, chi, np_ in itertools.product([6, 7, 8], [12, 16], [0.2, 0.3, 0.45], [0.6, 0.8], [0.12, 0.25], [3, 4]):
    env = dict(VET=1, PROM=1, PROM_Z0=z0, PROM_Z1=z1, PROM_P0=p0, PROM_P1=p1, PROM_CHI=chi, PROM_NP=np_)
    print((z0, z1, p0, p1, chi, np_), run(env, False, 600), run(env, False, 200))
