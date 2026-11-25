import sys, json, os, numpy as np
out, sdd = sys.argv[1], sys.argv[2]
cs = sorted(json.load(open(out))["candidates"], key=lambda c: -c["p"])
p = np.array([c["p"] for c in cs])
print("n", len(p), "p>=0.4/0.5/0.8/0.9:", [(p >= x).sum() for x in (0.4, 0.5, 0.8, 0.9)], "sum p top25 %.2f" % p[:25].sum())
print("top25 p", np.round(p[:25], 3).tolist())
print("p at rank 25/50/100/200:", [round(float(p[min(r, len(p)) - 1]), 3) for r in (25, 50, 100, 200)])
f = os.path.join(sdd, "refined_vc.npz")
if os.path.exists(f):
    d = np.load(f); rsc, rkd = d["rsc"], d["rkd"]; v, c = d["v"], d["c"]
    for k in np.unique(rkd):
        print(" kind", k, "n", (rkd == k).sum(), "rsc med %.2f mad %.2f p99 %.2f max %.2f" % (np.median(rsc[rkd == k]), 1.4826 * np.median(np.abs(rsc[rkd == k] - np.median(rsc[rkd == k]))), np.percentile(rsc[rkd == k], 99), rsc[rkd == k].max()))
    ref = rsc[rkd == 2] if (rkd == 2).sum() >= 30 else rsc[rkd >= 2]
    med = np.median(ref); mad = 1.4826 * np.median(np.abs(ref - med)) + 1e-3
    print(" null med %.3f mad %.3f; v<-2.9 frac %.4f, v>=4.9 frac %.4f, cov frac %.3f" % (med, mad, (v[c] < -2.9).mean(), (v[c] >= 4.9).mean(), c.mean()))
    ref_ = json.load(open(os.path.join(sdd, "refined.json")))["candidates"]
    S = np.array([r["state"] for r in ref_]); K = np.array([r["kd"] for r in ref_])
    ks = []
    for cc in cs[:25]:
        dd = np.linalg.norm(S[:, :3] - np.array(cc["state"][:3]), axis=1) / np.linalg.norm(cc["state"][:3])
        ks.append(int(K[np.argmin(dd)]) if dd.min() < 1e-3 else -1)
    print(" top25 kinds (nearest refined, -1 = none)", ks)
    tc = json.load(open(os.path.join(sdd, "tocands.json")))["candidates"]
    print(" tocands n", len(tc), "with zc", sum("zc" in x for x in tc), "p>=0.5", sum(x["p"] >= 0.5 for x in tc))
