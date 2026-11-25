import json, sys, numpy as np
a = sorted(json.load(open(sys.argv[1]))["candidates"], key=lambda c: -c["p"])   # old (10 exp)
b = sorted(json.load(open(sys.argv[2]))["candidates"], key=lambda c: -c["p"])   # new (12 exp)
off = int(sys.argv[3]) if len(sys.argv) > 3 else 2
B = np.array([c["_xy"] for c in b])[:, off:, :]
print(len(a), len(b))
for i, c in enumerate(a[:25]):
    xy = np.array(c["_xy"]); d = np.hypot(*(B - xy[None]).transpose(2, 0, 1))
    n = (d < 2).sum(1); j = int(np.argmax(n))
    print("old", i, "s=%.1f" % c["score"], "-> new", j if n[j] >= 3 else None, "nmatch", n[j], ("s=%.1f snr %s" % (b[j]["score"], b[j]["snr"])) if n[j] >= 3 else "", "| old snr", c["snr"])
