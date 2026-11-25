"""k3off.py SDDIR STACKOUT.json OUTPREFIX : kind-3 zc / night stats offline from stage dumps; writes
OUTPREFIX_k3.json (all kind-3 refined tracks, p = zc/100) and prints stats for the stackd output's zc tracks."""
import sys, json, numpy as np
sd, so, op = sys.argv[1:4]
z = np.load(sd + "/refined_vc.npz")
v, c, wf, night, rsc, rkd = z["v"].astype(float), z["c"].astype(bool), z["wf"].astype(float), z["night"], z["rsc"], z["rkd"]
ref = json.load(open(sd + "/refined.json"))["candidates"]
S = np.array([r["state"] for r in ref])
RC = 3.0
def tot(v_, c_):
    return (v_ * wf[None]).sum(1) / np.sqrt(np.maximum((c_ * wf[None] ** 2).sum(1), 1e-3))
vv = np.where(c, v, 0.0)
scq = tot(np.minimum(vv, RC), c)
nul = scq[rkd == 2] if (rkd == 2).sum() >= 30 else scq[rkd >= 2]
mq, sq = np.median(nul), 1.4826 * np.median(np.abs(nul - np.median(nul))) + 1e-3
zc = (scq - mq) / sq
vc = np.where(c, np.minimum(vv, RC), 0.0)
num = vc * wf[None]; den = c * wf[None] ** 2
Nt, Dt = num.sum(1), den.sum(1)
ncv = np.zeros(len(v), int); npo = np.zeros(len(v), int); lo = np.full(len(v), np.inf)
for u in np.unique(night):
    idx = np.flatnonzero(night == u)
    cv_ = c[:, idx].any(1); nk = num[:, idx].sum(1); dk = den[:, idx].sum(1)
    ncv += cv_; npo += cv_ & (nk > 0)
    lo = np.where(cv_, np.minimum(lo, (Nt - nk) / np.sqrt(np.maximum(Dt - dk, 1e-3))), lo)
k3 = np.flatnonzero(rkd == 3)
print(f"refined {len(v)}, kind3 {len(k3)}, null median {mq:.2f} MAD {sq:.2f}")
o = np.argsort(-zc[k3])
for q in o[:40]:
    j = k3[q]
    print(f"k3 #{q} zc {zc[j]:.2f} pos/cov {npo[j]}/{ncv[j]} lono {lo[j]:.2f} r {np.linalg.norm(S[j,:3]):.1f}")
json.dump({"candidates": [dict(epoch_mjd=ref[j]["epoch_mjd"], state=ref[j]["state"], p=float(min(max(zc[j] / 100, 1e-4), 0.99)),
           zc=float(zc[j]), npo=int(npo[j]), ncv=int(ncv[j]), lo=float(lo[j])) for j in k3]}, open(op + "_k3.json", "w"))
# stackd output: match by nearest refined position (polish moves states slightly)
out = json.load(open(so))["candidates"]
out = sorted(out, key=lambda x: -x["p"])
for k, cc in enumerate(out):
    if "zc" in cc:
        j = int(np.argmin(np.linalg.norm(S[:, :3] - np.array(cc["state"][:3])[None], axis=1)))
        cc.update(npo=int(npo[j]), ncv=int(ncv[j]), lo=float(lo[j]), zc2=float(zc[j]))
        if cc["zc"] >= 4:
            print(f"out rank {k} p {cc['p']:.3f} zc {cc['zc']:.2f} (re {zc[j]:.2f}) pos/cov {npo[j]}/{ncv[j]} lono {lo[j]:.2f}")
json.dump({"candidates": out}, open(op + "_outx.json", "w"))
