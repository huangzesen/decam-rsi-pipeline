import sys, os, csv, json, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen113/tmp/armB/submission/stackd")
import mapper
scen = sys.argv[1]; idx = [int(x) for x in sys.argv[2].split(",")]
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
for i in idx:
    r = rows[i]
    ip = os.path.join(scen, r["image"]); mp = os.path.join(scen, r["mask"])
    out = {}
    for bf in ("0", "3"):
        os.environ["TNO_BGFIX"] = bf
        d = mapper.map_exposure((i, ip, mp))
        out[bf] = [pm.astype(np.float32) for g, pm in d["maps"]]
    a = np.concatenate([x[np.isfinite(x)] for x in out["0"]]); b = np.concatenate([x[np.isfinite(x)] for x in out["3"]])
    def st(x):
        c = np.clip(x, -3, 5)
        med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med))
        return dict(n=x.size, med=round(float(med), 3), mad=round(float(mad), 3), lt3=round(float((x < -3).mean() * 100), 3),
                    gt3=round(float((x > 3).mean() * 100), 3), gt4=round(float((x > 4).mean() * 100), 3), gt5=round(float((x > 5).mean() * 100), 3),
                    cmean=round(float(c.mean()), 4), cstd=round(float(c.std()), 4))
    dd = [np.nanstd(x0.astype(np.float64) - x3) for x0, x3 in zip(out["0"], out["3"]) if x0.shape == x3.shape]
    print(i, r["band"], "BG0", st(a)); print(i, r["band"], "BG3", st(b))
    print("  per-CCD std(diff) median %.3f max %.3f, nccd %d/%d" % (np.median(dd), np.max(dd), len(out["0"]), len(out["3"])), flush=True)
