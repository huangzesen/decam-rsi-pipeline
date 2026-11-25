"""Leave-one-run-out evaluation of a learned membership key for stackd's 200-row cap.

python train.py TAG1 TAG2 ... [--save model.json]
"""
import sys, json
import numpy as np

T = "$RSI_SANDBOX/r1/gen149/tmp"
CAP = 200


def design(X, p, kd):
    lp = np.log(np.clip(p, 1e-4, 0.9999) / (1 - np.clip(p, 1e-4, 0.9999)))
    k3 = (kd == 3).astype(float); k1 = (kd == 1).astype(float); k4 = (kd == 4).astype(float)
    import os
    if os.environ.get("NOKIND"):
        return np.column_stack([X, lp, 0 * k1, 0 * k3, 0 * k4])
    return np.column_stack([X, lp, k1, k3, k4])


def fit(Z, y, l2=1e-2, hidden=0, iters=3000, lr=0.05, seed=0):
    mu = Z.mean(0); sd = Z.std(0) + 1e-6
    Zs = (Z - mu) / sd
    pos = y.astype(float)
    wpos = 0.5 / max(pos.sum(), 1); wneg = 0.5 / max((1 - pos).sum(), 1)
    sw = np.where(pos > 0, wpos, wneg)
    rng = np.random.default_rng(seed)
    if hidden:
        W1 = rng.normal(0, 0.3, (Zs.shape[1], hidden)); b1 = np.zeros(hidden)
        w2 = rng.normal(0, 0.3, hidden); b2 = 0.0
    else:
        w = np.zeros(Zs.shape[1]); b = 0.0
    m = {}
    for it in range(iters):
        if hidden:
            H = np.tanh(Zs @ W1 + b1); s = H @ w2 + b2
        else:
            s = Zs @ w + b
        q = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
        g = (q - pos) * sw
        if hidden:
            gw2 = H.T @ g + l2 * w2; gb2 = g.sum()
            gH = np.outer(g, w2) * (1 - H ** 2)
            gW1 = Zs.T @ gH + l2 * W1; gb1 = gH.sum(0)
            for k, (P, G) in enumerate(((W1, gW1), (b1, gb1), (w2, gw2))):
                P -= lr * G
            b2 -= lr * gb2
        else:
            w -= lr * (Zs.T @ g + l2 * w); b -= lr * g.sum()
    if hidden:
        return dict(mu=mu.tolist(), sd=sd.tolist(), W1=W1.tolist(), b1=b1.tolist(), w2=w2.tolist(), b2=float(b2))
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), b=float(b))


def score(m, Z):
    Zs = (Z - np.array(m["mu"])) / np.array(m["sd"])
    if "W1" in m:
        return np.tanh(Zs @ np.array(m["W1"]) + np.array(m["b1"])) @ np.array(m["w2"]) + m["b2"]
    return Zs @ np.array(m["w"]) + m["b"]


PROT = 50


def nbodies(key, y, cap=CAP, lp=None):
    if lp is not None:   # protect the top PROT rows by p, re-rank the rest by key
        o0 = np.argsort(-lp, kind="stable")
        top = o0[:PROT]; rest = o0[PROT:]
        o = np.concatenate([top, rest[np.argsort(-key[rest], kind="stable")]])[:cap]
    else:
        o = np.argsort(-key, kind="stable")[:cap]
    return len(set(y[o][y[o] >= 0]))


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    hidden = int(sys.argv[sys.argv.index("--hidden") + 1]) if "--hidden" in sys.argv else 0
    l2 = float(sys.argv[sys.argv.index("--l2") + 1]) if "--l2" in sys.argv else 1e-2
    tags = [t for t in tags if t not in (save, str(hidden), str(l2))]
    D = {t: dict(np.load(f"{T}/runs/feat_{t}.npz")) for t in tags}
    import os
    for t, d in D.items():
        if not os.environ.get("KEEP0"):   # kind-0 rows = linker seeds; their bodies are in the linker output already
            m_ = d["kd"] != 0
            for k_ in ("X", "y", "p", "kd", "rsc", "idx"):
                d[k_] = d[k_][m_]
        d["Z"] = design(d["X"], d["p"], d["kd"])
    print("tag      rows  bodies | p-200  lbno-200 model-200 p-25 prot-200 | p-top400")
    tot = np.zeros(6)
    for t in tags:
        tr = [u for u in tags if u != t]
        Z = np.concatenate([D[u]["Z"] for u in tr]); y = np.concatenate([D[u]["y"] for u in tr]) >= 0
        m = fit(Z, y, l2=l2, hidden=hidden)
        d = D[t]
        s = score(m, d["Z"])
        lp = d["Z"][:, -4]
        r = [nbodies(lp, d["y"]), nbodies(d["X"][:, 2], d["y"]), nbodies(s, d["y"]),
             nbodies(lp, d["y"], 25), nbodies(s, d["y"], 200, lp), nbodies(lp, d["y"], 400)]
        tot += r
        print(f"{t:8s} {len(d['y']):5d} {len(set(d['y'][d['y'] >= 0])):4d}   | " + "  ".join(f"{x:5d}" for x in r))
    print("total" + " " * 17 + "| " + "  ".join(f"{int(x):5d}" for x in tot))
    if save:
        Z = np.concatenate([D[u]["Z"] for u in tags]); y = np.concatenate([D[u]["y"] for u in tags]) >= 0
        m = fit(Z, y, l2=l2, hidden=hidden)
        json.dump(m, open(save, "w"))
        Zs_names = None
        if "w" in m:
            import feat
            names = feat.FNAMES + ["logit_p", "k1", "k3", "k4"]
            print("weights:", {n: round(w, 2) for n, w in zip(names, m["w"])})


if __name__ == "__main__":
    sys.path.insert(0, "$RSI_SANDBOX/r1/gen149/tmp/tools")
    main()
