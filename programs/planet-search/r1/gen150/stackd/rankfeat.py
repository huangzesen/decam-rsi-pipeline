"""gen 150: features + small learned ranker for stackd's refined tracks (TNO_LRANK).

Per refined track, from its per-exposure S/N values v (N frames, 0 where not covered), coverage c, frame weights wf
and the night grouping, build scenario-invariant features: every S/N-like statistic is standardised by the median and
1.4826 MAD of the same statistic over the kind-2 (window-seed) null rows of the SAME scenario.  A small MLP
(rankmlp.pt, trained in-session on injection runs) maps them to a logit; stackd then null-calibrates that logit the same
way (kind-2 rows) and uses it as the z that decides membership of the 200-row output and its p.
"""
import os
import numpy as np

# no seed-kind inputs: the injected training bodies are mostly fstk (kind 3) finds, a composition artefact
FEAT_NAMES = ["tot", "totc3", "lbno", "n3", "med", "top1", "top2", "minn", "frac1", "chi", "ncov", "nexp", "vmax"]


def night_stats(v, c, wf, nights_idx):
    v = np.asarray(v, float); c = np.asarray(c, bool); wf = np.asarray(wf, float)
    ns, sq, cv = [], [], []
    for idx in nights_idx:
        wn = wf[idx]
        q = np.sqrt(np.maximum((c[:, idx] * (wn ** 2)[None]).sum(1), 1e-6))
        s_ = (v[:, idx] * wn[None]).sum(1) / q
        cov = c[:, idx].any(1)
        ns.append(np.where(cov, s_, np.nan)); sq.append(np.where(cov, q, 0.0)); cv.append(cov)
    return np.stack(ns, 1), np.stack(sq, 1), np.stack(cv, 1)


def raw_features(v, c, wf, nights_idx, kd):
    v = np.asarray(v, float); c = np.asarray(c, bool); wf = np.asarray(wf, float)
    n = len(v)
    wf2 = wf ** 2
    den = np.sqrt(np.maximum((c * wf2[None]).sum(1), 1e-6))
    tot = (v * wf[None]).sum(1) / den
    totc = (np.minimum(v, 3.0) * wf[None]).sum(1) / den
    ns, sq, cv = night_stats(v, c, wf, nights_idx)
    nsf = np.where(cv, ns, -np.inf)
    srt = np.sort(nsf, 1)[:, ::-1]
    K = ns.shape[1]
    top1 = srt[:, 0]
    top2 = srt[:, 1] if K >= 2 else np.full(n, -3.0)
    n3 = srt[:, 2] if K >= 3 else np.full(n, -3.0)
    top2 = np.where(np.isfinite(top2), top2, -3.0); n3 = np.where(np.isfinite(n3), n3, -3.0)
    top1 = np.where(np.isfinite(top1), top1, -3.0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(cv, ns, np.nan), 1)
        minn = np.nanmin(np.where(cv, ns, np.nan), 1)
    med = np.where(np.isfinite(med), med, -3.0); minn = np.where(np.isfinite(minn), minn, -3.0)
    # leave-brightest-night-out stack
    bi = np.argmax(nsf, 1)
    keep = c.copy()
    for k_, idx in enumerate(nights_idx):
        sel = bi == k_
        if sel.any():
            keep[np.ix_(sel, idx)] = False
    lbno = (v * keep * wf[None]).sum(1) / np.sqrt(np.maximum((keep * wf2[None]).sum(1), 1e-6))
    # constant-flux consistency across nights: ns_k ~ A q_k + N(0,1)
    nsz = np.where(cv, ns, 0.0)
    A = (nsz * sq).sum(1) / np.maximum((sq ** 2).sum(1), 1e-6)
    ncov = cv.sum(1)
    chi = ((nsz - A[:, None] * sq) ** 2 * cv).sum(1) / np.maximum(ncov - 1, 1)
    pos = np.maximum(nsz, 0.0) ** 2
    frac1 = np.max(pos, 1) / np.maximum(pos.sum(1), 1e-6)
    nexp = c.sum(1)
    vmax = np.where(c, v, -9).max(1)
    kd = np.asarray(kd)
    return dict(tot=tot, totc3=totc, lbno=lbno, n3=n3, med=med, top1=top1, top2=top2, minn=minn, frac1=frac1,
                chi=np.log1p(chi), ncov=ncov / max(K, 1), nexp=nexp / max(c.shape[1], 1), vmax=np.minimum(vmax, 10.0),
                k0=(kd == 0).astype(float), k1=(kd == 1).astype(float), k3=(kd == 3).astype(float), k4=(kd == 4).astype(float))


ZNORM = ["tot", "totc3", "lbno", "n3", "med", "top1", "top2", "minn", "vmax", "chi"]


def features(v, c, wf, nights_idx, kd, nullmask=None):
    f = raw_features(v, c, wf, nights_idx, kd)
    kd = np.asarray(kd)
    # null rows: the fstk-seeded (kind 3) refined tracks, present in every scenario in thousands (kind 2 can be absent)
    nm = nullmask if nullmask is not None else (kd == 3)
    if int(np.sum(nm)) < 30:
        nm = kd >= 2
    if int(np.sum(nm)) < 30:
        nm = np.ones(len(kd), bool)
    for k in ZNORM:
        x = f[k]; r = x[nm]; r = r[np.isfinite(r)]
        med = float(np.median(r)) if r.size else 0.0
        mad = 1.4826 * float(np.median(np.abs(r - med))) + 1e-3 if r.size else 1.0
        f[k] = np.clip((x - med) / mad, -10, 30)
    X = np.stack([f[k] for k in FEAT_NAMES], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=30.0, neginf=-10.0)


_MODEL = None


def load(path=None):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "rankmlp.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    _MODEL = d
    return d


def predict(X, model=None):
    """Logit per row (numpy, CPU; tiny MLP)."""
    d = model or load()
    X = np.asarray(X, np.float32)
    if "lo" in d:   # clip to the training range (no linear extrapolation of the ReLU net)
        X = np.clip(X, d["lo"], d["hi"])
    X = (X - d["mu"]) / d["sd"]
    h = X
    for i, (W, b) in enumerate(d["layers"]):
        h = h @ W.T + b
        if i < len(d["layers"]) - 1:
            h = np.maximum(h, 0.0)
    return h[:, 0].astype(float)


def _rank(x):
    r = np.empty(len(x)); r[np.argsort(np.asarray(x, float), kind="stable")] = np.arange(len(x))
    return r


def key(X, mode="lin"):
    """Membership key per refined track.  lbno: leave-brightest-night-out stack z (gen147); lin: the learned model;
    rkavg: sum of the two ranks (offline the most robust of the three on held-out injection scenarios)."""
    lb = np.asarray(X[:, FEAT_NAMES.index("lbno")], float)
    if mode == "lbno":
        return lb
    lg = predict(X)
    if mode == "lin":
        return lg
    return _rank(lg) + _rank(lb) + 1e-3 * _rank(X[:, FEAT_NAMES.index("tot")])
