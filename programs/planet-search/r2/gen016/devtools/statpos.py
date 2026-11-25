import sys, json, pickle, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r2/gen012/workspace/submission")
import finder as F
out, cache = sys.argv[1], sys.argv[2]
dets, maps = pickle.load(open(cache, "rb"))
c = sorted(json.load(open(out))["candidates"], key=lambda c: -c["p"])
for rk in range(25):
    xy = np.array(c[rk]["_xy"]); sn = c[rk]["snr"]
    x, y = xy[0]
    vals = []
    for e in sorted(maps):
        y0, x0, arr = maps[e]; iy = int(round(y / F.PS)) - y0; ix = int(round(x / F.PS)) - x0
        vals.append(arr[iy - 1:iy + 2, ix - 1:ix + 2].max() / 10 if 1 <= iy < arr.shape[0] - 1 and 1 <= ix < arr.shape[1] - 1 else None)
    mot = np.hypot(*(xy[1] - xy[0]))
    print(rk, "s=%.1f" % c[rk]["score"], "cand snr", sn[:2], "| raw maps at exp0 position, all exposures:", vals, "| move exp0->1 %.2f\"" % mot)
