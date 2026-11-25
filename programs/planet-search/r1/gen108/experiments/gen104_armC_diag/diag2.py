import json, sys, numpy as np
D = json.load(open(sys.argv[1])); T = json.load(open("$RSI_SANDBOX/r1/gen104/tmp/inj7c/truth.json"))
v = np.array(D["v"]); c = np.array(D["c"]); tot = D["total"]; n3 = D["n3"]
print("frames", len(D["t"]), "nights", D["night"], "wf", np.round(D["wf"], 2))
for j, b in enumerate(T["bodies"]):
    if b["cls"] == "A" and j not in (0, 7, 12): continue
    print(j, b["cls"], "total %.2f n3 %.2f" % (tot[j], n3[j]), "v", " ".join("%.1f" % x if cc else "  - " for x, cc in zip(v[j], c[j])))
