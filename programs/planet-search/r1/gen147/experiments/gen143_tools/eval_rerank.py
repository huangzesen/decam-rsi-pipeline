"""python eval_rerank.py S(R|Q) [model.pt]: re-measure vet (with CNN) on the vdump head, rerank final p, score."""
import sys, os, json, csv, math, numpy as np
T = "$RSI_SANDBOX/r1/gen143/tmp/"
sys.path.insert(0, "$RSI_SANDBOX/r1/gen143/workspace/submission")
sys.path.insert(0, "$RSI_SANDBOX/r1/gen143/workspace/work")
import vetcnn, vet, score_inject as SI
S = sys.argv[1]
if len(sys.argv) > 2:
    vetcnn.load(sys.argv[2])
scen = T + "scen_inj" + S
truth = json.load(open(scen + "/truth.json")); rows = list(csv.DictReader(open(scen + "/exposures.csv")))
vd = json.load(open(T + f"runs/vdump_{S}.json"))
out = json.load(open(T + f"runs/out_{S}.json"))["candidates"]
cache = T + f"runs/feat_{S}_{os.path.basename(sys.argv[2]) if len(sys.argv)>2 else 'def'}.json"
if os.path.exists(cache):
    fs = json.load(open(cache))
else:
    fs = vet.measure(scen, vd, workers=1)
    json.dump(fs, open(cache, "w"))
key = lambda c: (round(float(c["epoch_mjd"]), 6), tuple(round(float(x), 9) for x in c["state"]))
cnn = {key(c): (f or {}).get("cnn", float("nan")) for c, f in zip(vd, fs)}
zl = {key(c): (f or {}).get("zlbno", float("nan")) for c, f in zip(vd, fs)}
_, res, hr, dup = SI.score(truth, out, rows)
valid = {j for j, b in enumerate(truth["bodies"]) if len(b["nights"]) >= 3}
def top_hits(cs, top=25):
    _, res, hr, dup = SI.score(truth, cs, rows)
    return sum(1 for j, (r, p) in hr.items() if r < top and j in valid), sum(1 for j, (r, p) in hr.items() if j in valid)
lab = [r[0] if r is not None else -1 for r in res]
cv = np.array([cnn.get(key(c), np.nan) for c in out])
print(f"{S}: {len(out)} out rows, cnn measured {np.isfinite(cv).sum()}; baseline top25 hits / any: {top_hits(out)}")
pos = np.array([l >= 0 for l in lab])
for nm, m in (("hit", pos), ("other", ~pos)):
    v = cv[m & np.isfinite(cv)]
    print(f"  cnn {nm}: n {len(v)} quantiles {np.round(np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]), 2) if len(v) else ''}")
print("  rows (rank, p, cnn, zlbno, hit):")
for i, c in enumerate(out[:40]):
    print(f"   {i:3d} {c['p']:.3f} {cv[i]:7.2f} {zl.get(key(c), np.nan):6.2f} {'HIT %d %s' % (lab[i], truth['bodies'][lab[i]]['cls']) if lab[i] >= 0 else ''}")
sig = lambda x: 1 / (1 + math.exp(-x))
for a in (0.0, 0.2, 0.4, 0.6):
    for b in (-4, -2, 0, 2, 4):
        cs = []
        for i, c in enumerate(out):
            x = cv[i]
            f = 1.0 if not np.isfinite(x) else a + (1 - a) * sig(x - b)
            cs.append(dict(c, p=float(c["p"]) * f))
        print(f"  a {a} b {b}: top25 {top_hits(cs)}")
