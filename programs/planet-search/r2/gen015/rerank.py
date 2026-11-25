"""gen011: learned re-ranker of the final candidate list (features + tiny network; weights in rerank.pt).
Used by finder.py (finalize) and by devtools/rr_train.py, so both see the same features."""
import math
import numpy as np

GM = 2.959122082855911e-4
EPS = math.radians(23.43928)
KECL = np.array([0.0, -math.sin(EPS), math.cos(EPS)])
NF = 24


def scen_refs(scores, Zs):
    """Chance level of a scenario from its own list (sorted by score, best first)."""
    s = np.sort(np.asarray(scores, float))[::-1]; n = len(s)
    if n == 0:
        return 0.0, 0.0, 0.0
    s40 = float(s[min(39, n - 1)]); s100 = float(s[min(99, n - 1)])
    o = np.argsort(-np.asarray(scores, float)); z = np.asarray(Zs, float)[o]
    zref = float(np.median(z[25:100])) if n > 40 else float(np.median(z))
    return s40, s100, zref


def features(c, wS, night, t, refs, prior_fn):
    """c: candidate (score, Z, snr list with None/nan, state equatorial heliocentric, two). Returns [NF] float32."""
    sv = np.array([np.nan if v is None else v for v in c["snr"]], float)
    ok = np.isfinite(sv); s = sv[ok]; w = np.asarray(wS, float)[ok]; nt = np.asarray(night)[ok]
    s40, s100, zref = refs
    f = np.zeros(NF, np.float32)
    if len(s) == 0:
        return f
    fl = max(float(np.sum(s * w) / np.sum(w * w)), 0.0); mu = fl * w
    st = np.array([s[nt == n].sum() / math.sqrt((nt == n).sum()) for n in np.unique(nt)])
    wn = np.array([math.sqrt(np.sum(w[nt == n] ** 2)) for n in np.unique(nt)])
    P = np.asarray(c["state"][:3], float); V = np.asarray(c["state"][3:], float)
    r = float(np.linalg.norm(P)); h = np.cross(P, V); hn = float(np.linalg.norm(h)) + 1e-30
    inc = math.degrees(math.acos(max(-1.0, min(1.0, float(h @ KECL) / hn))))
    beta = abs(math.degrees(math.asin(max(-1.0, min(1.0, float(P @ KECL) / r)))))
    fr = float(P @ V) / math.sqrt(GM * r); wq = hn * hn / (GM * r)
    Z = float(c["Z"]); sc = float(c["score"])
    f[0] = (sc - s40) / 10.0
    f[1] = Z / 10.0
    f[2] = (Z - zref) / 5.0
    f[3] = len(s) / 10.0
    f[4] = int(np.sum(st >= 2.5)) / 5.0
    f[5] = int(np.sum(s >= 3.0)) / 5.0
    f[6] = int(np.sum(s >= 4.5)) / 5.0
    f[7] = min(float(s.max()), 12.0) / 10.0
    f[8] = max(float(s.min()), -4.0) / 5.0
    f[9] = min(float(np.mean((s - mu) ** 2)), 20.0) / 5.0
    f[10] = len(st) / 5.0
    f[11] = float(np.sum(st >= 2.5)) / len(st)
    f[12] = math.sqrt(float(np.sum(np.maximum(st, 0) ** 2))) / 10.0
    f[13] = max(float(st.min()), -4.0) / 5.0
    f[14] = min(float(prior_fn(np, np.float64(r), np.float64(wq), np.float64(inc), np.float64(fr), np.float64(beta))), 8.0) / 4.0
    f[15] = min(r, 100.0) / 40.0 - 1.0
    f[16] = min(wq, 2.0) - 1.0
    f[17] = min(inc, 60.0) / 20.0
    f[18] = min(abs(fr), 1.0)
    f[19] = float(c.get("two") or 0) / 2.0
    f[20] = min(float(np.ptp(t)), 60.0) / 30.0
    f[21] = (sc - s100) / 10.0
    f[22] = float(np.sum((mu >= 3.0) & (s < 1.0))) / 3.0          # exposures that should show it and do not
    f[23] = float(np.sum(np.sort(s)[-3:])) / 20.0 if len(s) >= 3 else 0.0
    return f


def load(torch, path):
    d = torch.load(path, map_location="cpu")
    return d


def predict(torch, d, X):
    """X [n, NF] numpy -> logit [n] numpy (mean over the ensemble)."""
    x = torch.as_tensor(np.asarray(X, np.float32))
    x = (x - d["mean"]) / d["std"]
    out = 0.0
    for m in d["models"]:
        h = x
        for li, (W, b) in enumerate(m):
            h = h @ W.T + b
            if li < len(m) - 1:
                h = torch.tanh(h)
        out = out + h[:, 0]
    return (out / len(d["models"])).numpy()
