"""Per-row features of stackd's refined tracks (stage dumps) + truth labels.

python feat.py SCEN TAG  -> runs/feat_TAG.npz with X (tocands rows), y (body id or -1), p, kd
Features are computed by rowfeat() which is ALSO what the shipped stackd code uses (copied there verbatim).
"""
import sys, json, csv
import numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen149/tmp/tools")

T = "$RSI_SANDBOX/r1/gen149/tmp"


def rowfeat(v, c, wf, night):
    """v,c (K,N) per-exposure 1" S/N map values and coverage; wf (N,) frame weights; night (N,) night labels.
    Returns (K, F) float features, permutation-invariant over exposures/nights."""
    v = np.asarray(v, float); c = np.asarray(c, bool); wf = np.asarray(wf, float)
    K, N = v.shape
    v = np.where(c, v, 0.0)
    wf2 = wf ** 2
    den = np.sqrt(np.maximum((c * wf2[None]).sum(1), 1e-3))
    tot = (v * wf[None]).sum(1) / den
    totc = (np.minimum(v, 3.0) * wf[None]).sum(1) / den
    un = np.unique(night)
    ns = np.full((K, len(un)), np.nan)
    for k, n in enumerate(un):
        m = night == n
        w = wf[m]
        cm = c[:, m]
        s = (v[:, m] * w[None]).sum(1) / np.sqrt(np.maximum((cm * (w ** 2)[None]).sum(1), 1e-3))
        ns[:, k] = np.where(cm.any(1), s, np.nan)
    nsf = np.where(np.isfinite(ns), ns, -np.inf)
    bi = np.argmax(nsf, 1)
    keep = ~(night[None, :] == un[bi][:, None])
    c2 = c & keep
    lbno = (v * c2 * wf[None]).sum(1) / np.sqrt(np.maximum((c2 * wf2[None]).sum(1), 1e-3))
    nn = np.isfinite(ns).sum(1)
    srt = -np.sort(-np.where(np.isfinite(ns), ns, -1e9), 1)   # descending, missing at the end as -1e9

    def kth(k):
        return np.where(nn > k, srt[:, min(k, srt.shape[1] - 1)], -3.0)
    med = np.where(nn > 0, np.nanmedian(np.where(np.isfinite(ns), ns, np.nan), 1), -3.0) if ns.shape[1] else np.zeros(K)
    nsmin = np.where(nn > 0, np.nanmin(np.where(np.isfinite(ns), ns, np.inf), 1), -3.0)
    npos = (np.nan_to_num(ns, nan=-9) > 1.0).sum(1)
    rob = np.nansum(np.clip(np.nan_to_num(ns, nan=0), -2, 2.5), 1) / np.sqrt(np.maximum(nn, 1))
    vs = -np.sort(-np.where(c, v, -1e9), 1)
    ncov = c.sum(1)
    vmax = np.where(ncov > 0, vs[:, 0], 0.0)
    v2 = np.where(ncov > 1, vs[:, min(1, N - 1)], 0.0)
    # share of the total from the brightest exposure / brightest night (spike indicator)
    frac1 = vmax * wf.max() / np.maximum(np.abs(tot) * den, 1e-3)
    nshare = (kth(0) - lbno)
    chi = np.where(nn > 1, np.nanstd(np.where(np.isfinite(ns), ns, np.nan), 1), 0.0)
    F = np.stack([tot, totc, lbno, kth(0), kth(1), kth(2), med, nsmin, npos, rob, vmax, v2,
                  np.clip(frac1, -5, 5), nshare, np.nan_to_num(chi), nn / max(len(un), 1), ncov / max(N, 1)], 1)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


FNAMES = ["tot", "totc", "lbno", "n1", "n2", "n3", "nmed", "nmin", "npos", "rob", "vmax", "v2", "frac1", "nshare",
          "nchi", "nnfrac", "covfrac"]


def main():
    import score_inject
    S, tag = sys.argv[1], sys.argv[2]
    truth = json.load(open(f"{T}/scen_inj{S}/truth.json"))
    rows = list(csv.DictReader(open(f"{T}/scen_inj{S}/exposures.csv")))
    d = np.load(f"{T}/runs/sd_{tag}/refined_vc.npz")
    ref = json.load(open(f"{T}/runs/sd_{tag}/refined.json"))["candidates"]
    toc = json.load(open(f"{T}/runs/sd_{tag}/tocands.json"))["candidates"]
    F = rowfeat(d["v"], d["c"], d["wf"], d["night"])
    key = {tuple(np.round(x["state"], 10)): i for i, x in enumerate(ref)}
    idx = np.array([key.get(tuple(np.round(c["state"], 10)), -1) for c in toc])
    print(tag, "refined", len(ref), "tocands", len(toc), "unmatched", int((idx < 0).sum()))
    for c in toc:
        c.setdefault("p", 0.0)
    _, res, hr, _ = score_inject.score(truth, toc, rows, top=25)
    y = np.array([-1 if r is None else r[0] for r in res])
    p = np.array([float(c["p"]) for c in toc])
    kd = np.array([int(d["rkd"][i]) if i >= 0 else -1 for i in idx])
    rsc = np.array([float(d["rsc"][i]) if i >= 0 else 0 for i in idx])
    X = np.where(idx[:, None] >= 0, F[np.maximum(idx, 0)], 0.0)
    np.savez(f"{T}/runs/feat_{tag}.npz", X=X, y=y, p=p, kd=kd, rsc=rsc, idx=idx)
    order = np.argsort(-p, kind="stable")
    rk = np.empty(len(p), int); rk[order] = np.arange(len(p))
    bodies = sorted(set(y[y >= 0]))
    print("bodies", len(bodies), "best p-rank per body", sorted(int(rk[y == b].min()) for b in bodies))


if __name__ == "__main__":
    main()
