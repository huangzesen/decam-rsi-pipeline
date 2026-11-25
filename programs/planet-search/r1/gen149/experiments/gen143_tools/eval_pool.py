import sys, os, json, csv, math, numpy as np
T = "$RSI_SANDBOX/r1/gen143/tmp/"
sys.path.insert(0, "$RSI_SANDBOX/r1/gen143/workspace/submission/linker"); sys.path.insert(0, "$RSI_SANDBOX/r1/gen143/workspace/work")
import score_inject as SI
model = sys.argv[1]
P, Y, C, Z = [], [], [], []
for S in "RQ":
    scen = T + "scen_inj" + S
    truth = json.load(open(scen + "/truth.json")); rows = list(csv.DictReader(open(scen + "/exposures.csv")))
    vd = json.load(open(T + f"runs/vdump_{S}.json")); fs = json.load(open(T + f"runs/feat_{S}_{model}.json"))
    out = json.load(open(T + f"runs/out_{S}.json"))["candidates"]
    key = lambda c: (round(float(c["epoch_mjd"]), 6), tuple(round(float(x), 9) for x in c["state"]))
    fm = {key(c): f for c, f in zip(vd, fs)}
    _, res, hr, dup = SI.score(truth, out, rows)
    seen = set()
    for c, r in zip(out, res):
        f = fm.get(key(c)) or {}
        if "cnn" not in f: continue
        if r is not None and r[0] in seen: continue   # duplicate: neither
        if r is not None: seen.add(r[0])
        P.append(c["p"]); Y.append(r is not None); C.append(f["cnn"]); Z.append(f.get("zlbno", 0))
P, Y, C, Z = map(np.array, (P, Y, C, Z))
def auc(s, y):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(len(s)); p = y > 0
    return (r[p].mean() - (p.sum() - 1) / 2) / (~p).sum()
print(f"n {len(P)} pos {Y.sum()}  AUC p {auc(P,Y):.4f} cnn {auc(C,Y):.4f} zlbno {auc(Z,Y):.4f}")
sig = lambda x: 1 / (1 + np.exp(-x))
for a in (0.0, 0.2, 0.4, 0.6, 0.8):
    for b in (0, 3, 6, 9):
        q = P * (a + (1 - a) * sig(C - b))
        k = np.argsort(-q)[:40]
        print(f" a {a} b {b}: AUC {auc(q,Y):.4f}  hits in pooled top 29/40: {Y[np.argsort(-q)[:29]].sum()} / {Y[k].sum()}  (base {Y[np.argsort(-P)[:29]].sum()} / {Y[np.argsort(-P)[:40]].sum()})")
