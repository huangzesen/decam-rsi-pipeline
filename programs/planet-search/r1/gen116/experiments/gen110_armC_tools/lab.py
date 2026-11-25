"""lab.py TRUTH CANDS.json EXPOSURES N : label top-N candidates (by p) with hit body/class."""
import sys, json, csv, os
sys.path.insert(0, "$RSI_SANDBOX/r1/gen110/workspace/exp")
import score_inject as si
tr = json.load(open(sys.argv[1])); cs = json.load(open(sys.argv[2]))["candidates"]
rows = list(csv.DictReader(open(sys.argv[3]))); N = int(sys.argv[4])
cs = sorted(cs, key=lambda c: -c["p"])[:N]
cands, res, hr, dup = si.score(tr, cs, rows)
for k, (c, h) in enumerate(zip(cands, res)):
    lab = "-" if h is None else "%s%d" % (tr["bodies"][h[0]]["cls"], h[0])
    print(k, lab, " ".join(f"{q}={c[q]:.2f}" if isinstance(c.get(q), float) else f"{q}={c.get(q)}" for q in ("p", "zc", "npo", "ncv", "lo", "k3pro") if q in c))
