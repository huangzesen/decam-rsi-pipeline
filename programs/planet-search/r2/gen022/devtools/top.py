import sys, json, numpy as np
c = sorted(json.load(open(sys.argv[1]))["candidates"], key=lambda c: -c["p"])
for rk in range(int(sys.argv[2])):
    x = c[rk]; st = np.array(x["state"])
    print(rk, "p=%.3f s=%.1f r=%.1f" % (x["p"], x["score"], np.linalg.norm(st[:3])), "two" if x.get("two") else "", "ALT" if x.get("alt_of") is not None else "", x["snr"])
