"""stages.py SDDIR [stage ...]: per stage, hits by class and where C bodies / kind-3 junk sit (rank, p, zc, conc, k3g)."""
import sys, json, csv, os
sys.path.insert(0, "$RSI_SANDBOX/r1/gen113/tmp/exp")
import score_inject as S
sd = sys.argv[1]
stages = sys.argv[2:] or ["tocands", "dedup", "k3post", "cv", "rankx"]
I = "$RSI_SANDBOX/r1/gen113/tmp/inj7d"
truth = json.load(open(I + "/truth.json")); rows = list(csv.DictReader(open(I + "/exposures.csv")))
B = truth["bodies"]
for st in stages:
    f = os.path.join(sd, st + ".json")
    if not os.path.exists(f): print(st, "missing"); continue
    cands = json.load(open(f))["candidates"]
    if st == "tocands":
        cands = sorted(cands, key=lambda c: -c["p"])[:400]
    cs, res, hr, dup = S.score(truth, cands, rows)
    cls = lambda j: B[j]["cls"]
    t25 = [cls(j) + str(j) for j, (r, p) in hr.items() if r < 25]
    print(f"== {st}: n {len(cs)}  hits A {sum(cls(j)=='A' for j in hr)} B {sum(cls(j)=='B' for j in hr)} C {sum(cls(j)=='C' for j in hr)}; "
          f"top25 unique {len(t25)} {sorted(t25)} dups-in-25 {sum(1 for r, h in enumerate(res[:25]) if h is not None) - len(t25)}")
    for j, (r, p) in sorted(hr.items(), key=lambda x: x[1][0]):
        if cls(j) == "C":
            c = cs[r]; print(f"   C{j} rank {r} p {p:.3f} zc {c.get('zc', float('nan')):.2f} conc {c.get('conc', float('nan')):.2f} k3g {c.get('k3g', 0)}")
    # kind-3 (zc present) candidates in top 40
    for r, c in enumerate(cs[:40]):
        if "zc" in c:
            h = res[r]; lab = (cls(h[0]) + str(h[0])) if h else "JUNK"
            if c.get("k3g", 0) or lab == "JUNK" or lab[0] == "C":
                print(f"   k3 rank {r:3d} p {c['p']:.3f} zc {c['zc']:.2f} conc {c.get('conc', float('nan')):.2f} k3g {c.get('k3g', 0)} {lab}")
    if st == "k3post":
        g = [(r, c) for r, c in enumerate(cs) if c.get("k3g", 0)]
        print("   gated:", [(r, c["k3g"], (cls(res[r][0]) + str(res[r][0])) if res[r] else "JUNK", round(c["zc"], 2), round(c["conc"], 2)) for r, c in g])
        import numpy as np
        jk = [(c["zc"], c["conc"], (cls(res[r][0]) + str(res[r][0])) if res[r] else "JUNK") for r, c in enumerate(cs) if "conc" in c]
        print("   all kind-3 (zc,conc,lab) zc>=2.5:", sorted([(round(a, 2), round(b, 2), l) for a, b, l in jk if a >= 2.5], reverse=True))
