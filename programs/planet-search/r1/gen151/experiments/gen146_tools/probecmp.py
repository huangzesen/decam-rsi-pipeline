import re, sys
T = "$RSI_SANDBOX/r1/gen146/tmp/runs"
pat = re.compile(r"probe (\d+): d=([0-9.]+).*full-list rank (\d+)/(\d+)")
res = {}
for g in sys.argv[1:]:
    r = {}
    for l in open(f"{T}/log_{g}.txt"):
        m = pat.search(l)
        if m:
            r[(int(m[1]), m[2])] = int(m[3])
    res[g] = r
common = set.intersection(*[set(r) for r in res.values()])
ds = sorted({k[1] for k in common})
print("common slices", ds)
bodies = sorted({k[0] for k in common})
for g in sys.argv[1:]:
    best = {b: min(res[g][k] for k in common if k[0] == b) for b in bodies}
    print(g, "in pool(<15000):", sum(v < 15000 for v in best.values()), "<5000:", sum(v < 5000 for v in best.values()), "of", len(bodies))
print("body " + " ".join(sys.argv[1:]))
for b in bodies:
    print(b, " ".join(str(min(res[g][k] for k in common if k[0] == b)) for g in sys.argv[1:]))
