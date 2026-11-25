"""python ana.py SCEN OUT.json FEATS.json(create if missing) -> table of rank, p, hit class, features"""
import sys, json, csv, os
sys.path.insert(0, "$RSI_SANDBOX/r1/gen132/tmp/tools")
sys.path.insert(0, "$RSI_SANDBOX/r1/gen132/workspace/submission")
import score_inject, vet
scen, oj, fj = sys.argv[1:4]
truth = json.load(open(os.path.join(scen, "truth.json")))
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
cands = json.load(open(oj))["candidates"]
cands, res, hit_rank, dup = score_inject.score(truth, cands, rows)
if os.path.exists(fj):
    fs = json.load(open(fj))["feats"]
else:
    fs = vet.measure(scen, cands)
    json.dump({"feats": fs}, open(fj, "w"))
seen = set()
out = []
for i, (c, r, f) in enumerate(zip(cands, res, fs)):
    lab = "-"
    if r is not None:
        lab = truth["bodies"][r[0]]["cls"] + ("dup" if r[0] in seen else "")
        seen.add(r[0])
    out.append(dict(rank=i, p=c["p"], lab=lab, **(f or {})))
json.dump(out, open(fj.replace(".json", "_tab.json"), "w"))
for o in out:
    print(o["rank"], round(o["p"], 3), o["lab"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items() if k not in ("rank", "p", "lab")})
