import sys, json, csv
import numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen147/tmp/tools")
import score_inject
S, tag = sys.argv[1], sys.argv[2]
T = "$RSI_SANDBOX/r1/gen147/tmp"
truth = json.load(open(f"{T}/scen_inj{S}/truth.json")); rows = list(csv.DictReader(open(f"{T}/scen_inj{S}/exposures.csv")))
ref = json.load(open(f"{T}/runs/sd_{tag}/refined.json"))["candidates"]
N = len(ref)
for i, x in enumerate(ref): x["p"] = 1 - i / N
_, res, _, _ = score_inject.score(truth, ref, rows, top=25)
body = np.array([-1 if r is None else r[0] for r in res])   # same order as ref (sorted by p = index order)
d = np.load(f"{T}/runs/sd_{tag}/refined_vc.npz")
v, c, wf, night, kd = d["v"].astype(float), d["c"], d["wf"].astype(float), d["night"], d["rkd"]
assert len(v) == N
un = np.unique(night)
ns = []; nc = []
for n in un:
    m = night == n; w = wf[m]
    s = (v[:, m] * w).sum(1) / np.sqrt(np.maximum((c[:, m] * w ** 2).sum(1), 1e-6))
    ns.append(np.where(c[:, m].any(1), s, np.nan)); nc.append(c[:, m].any(1))
ns = np.stack(ns, 1); nc = np.stack(nc, 1)
tot = (v * wf).sum(1) / np.sqrt(np.maximum((c * wf ** 2).sum(1), 1e-6))
nsf = np.where(nc, ns, -np.inf)
srt = np.sort(nsf, 1)
nn = nc.sum(1)
# leave-brightest-night-out stack
bi = np.argmax(nsf, 1)
mask = c.copy()
for k, n in enumerate(un):
    mask[bi == k][:, night == n] = False
cm = c & ~(night[None, :] == un[bi][:, None])
lbno = (v * wf * cm).sum(1) / np.sqrt(np.maximum((cm * wf ** 2).sum(1), 1e-6))
med = np.nanmedian(ns, 1)
npos = (np.nan_to_num(ns, nan=-9) > 1.0).sum(1)
vcl = np.minimum(v, 3.0)
totc = (vcl * wf).sum(1) / np.sqrt(np.maximum((c * wf ** 2).sum(1), 1e-6))
# night-sum of sqrt-ish robust: sum over nights of clip(ns, -2, 3)
rob = np.nansum(np.clip(np.nan_to_num(ns, nan=0), -2, 2.5), 1) / np.sqrt(np.maximum(nn, 1))
stats = {"tot": tot, "totclip3": totc, "lbno": lbno, "med": med, "n3": np.where(nn >= 3, srt[:, -3], -9), "npos": npos + 0.01 * tot,
         "rob2.5": rob, "lbno+tot": lbno + 0.5 * tot, "minlbno_rob": np.minimum(lbno, rob * 1.0)}
bodies = sorted(set(body[body >= 0]))
print("N", N, "kinds", np.bincount(kd), "bodies", len(bodies))
print("stat     " + " ".join(f"{b:5d}" for b in bodies) + "   #<200 #<400  medrank")
for nm, s in stats.items():
    s = np.nan_to_num(s, nan=-99)
    rk = np.empty(N, int); rk[np.argsort(-s, kind="stable")] = np.arange(N)
    br = [int(rk[body == b].min()) for b in bodies]
    print(f"{nm:9s}" + " ".join(f"{r:5d}" for r in br) + f"   {sum(r < 200 for r in br):4d} {sum(r < 400 for r in br):4d}  {int(np.median(br))}")
