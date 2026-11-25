"""Offline CNN re-rank variants from a TNO_CNN_DUMP file.  sim.py TRUTH PRE.json EXPOSURES.csv OUTDIR"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen117/tmp/exp")
sys.path.insert(0, "$RSI_SANDBOX/r1/gen117/tmp/cnn/submission/stackd")
import score_inject as si
import tcnn

truth = json.load(open(sys.argv[1])); pre = json.load(open(sys.argv[2]))["candidates"]; rows = list(csv.DictReader(open(sys.argv[3])))
od = sys.argv[4]; os.makedirs(od, exist_ok=True)
pre = sorted(pre, key=lambda c: -c["p"])
q = np.array([c.get("pcnn", 0.5) for c in pre])
cs, res, hr, dup = si.score(truth, pre, rows)
B = truth["bodies"]
print("pre: n", len(pre), "q quantiles", np.round(np.quantile(q, [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist())
for r, h in enumerate(res):
    if h is not None and r < 400:
        print(f"  hit rank {r} body {h[0]} {B[h[0]]['cls']} p {pre[r]['p']:.3f} q {q[r]:.3f}")
junk = np.array([q[r] for r, h in enumerate(res) if h is None])
print("junk q quantiles", np.round(np.quantile(junk, [0.1, 0.5, 0.9, 0.99]), 3).tolist(), "frac>0.9", float((junk > 0.9).mean()))
print("junk q in top-50 pre:", np.round([q[r] for r in range(50) if res[r] is None], 2).tolist())


def write(nm, cands):
    json.dump({"candidates": sorted(cands, key=lambda c: -c["p"])[:200]}, open(os.path.join(od, nm + ".json"), "w"))


def rr(alpha, ps=0.8, qthr=None, lhi=4.0):
    q2 = q.copy()
    if qthr is not None:
        q2 = np.where(q2 >= qthr, q2, 0.5)   # bonus only on strong CNN evidence, no penalty
    out, mv = tcnn.rerank(pre, q2, ps, alpha, lhi)
    return out


write("pre", pre)
for a in (0.3, 0.5, 1.0):
    for lh in (4.0, 8.0, 12.0):
        write(f"a{a}_l{lh}", rr(a, lhi=lh))
