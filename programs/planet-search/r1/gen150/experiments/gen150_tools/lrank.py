"""Train / evaluate the learned refined-track ranker on stage dumps of injection runs.
python lrank.py TAG:S [TAG:S ...] [--out model.pt] [--eval]
"""
import sys, json, csv, os, argparse
import numpy as np
import torch
sys.path.insert(0, "$RSI_SANDBOX/r1/gen150/tmp/tools")
sys.path.insert(0, "$RSI_SANDBOX/r1/gen150/workspace/submission/stackd")
import score_inject
import rankfeat
T = "$RSI_SANDBOX/r1/gen150/tmp"
torch.set_num_threads(4)


def load(tag, S):
    cache = f"{T}/runs/sd_{tag}/lrank_cache.npz"
    d = np.load(f"{T}/runs/sd_{tag}/refined_vc.npz")
    v, c, wf, night, kd, rsc = d["v"].astype(float), d["c"], d["wf"].astype(float), d["night"], d["rkd"], d["rsc"]
    un = np.unique(night)
    nidx = [np.flatnonzero(night == u) for u in un]
    X = rankfeat.features(v, c, wf, nidx, kd)
    if os.path.exists(cache):
        body = np.load(cache)["body"]
    else:
        truth = json.load(open(f"{T}/scen_inj{S}/truth.json")); rows = list(csv.DictReader(open(f"{T}/scen_inj{S}/exposures.csv")))
        ref = json.load(open(f"{T}/runs/sd_{tag}/refined.json"))["candidates"]
        N = len(ref)
        for i, x in enumerate(ref):
            x["p"] = 1 - i / N
        _, res, _, _ = score_inject.score(truth, ref, rows, top=25)
        body = np.array([-1 if r is None else r[0] for r in res])
        np.savez(cache, body=body)
    # baseline key: z of total S/N; kind-3 rows max with clipped z (stackd to_cands, approx null = all kind 2)
    f = rankfeat.raw_features(v, c, wf, nidx, kd)
    nm = kd == 2
    if nm.sum() < 30:
        nm = kd >= 2
    def z(x):
        r = x[nm]; m = np.median(r); s = 1.4826 * np.median(np.abs(r - m)) + 1e-3
        return (x - m) / s
    zt = z(f["tot"]); zc = z(f["totc3"])
    base = np.where(kd == 3, np.maximum(zt, zc), zt)
    el = d["rng"] >= 3   # rows that to_cands can emit
    return dict(X=X[el], body=body[el], kd=kd[el], base=base[el], lbno=z(f["lbno"])[el], tag=tag)


def train(sets, seed=0, epochs=300, hid=32, wd=1e-3):
    X = np.concatenate([s["X"] for s in sets]); y = np.concatenate([(s["body"] >= 0).astype(np.float32) for s in sets])
    # per-body weight: each body's rows share weight 1 total, negatives total = positives total * 1
    w = np.zeros(len(y), np.float32)
    off = 0
    for s in sets:
        b = s["body"]
        for u in np.unique(b[b >= 0]):
            m = np.flatnonzero(b == u)
            w[off + m] = 1.0 / len(m)
        off += len(b)
    npos = w.sum(); nneg = (y == 0).sum()
    w[y == 0] = npos / nneg * 1.0
    mu = X.mean(0); sd = X.std(0) + 1e-3
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y); wt = torch.tensor(w)
    torch.manual_seed(seed)
    if hid <= 0:
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 1))
    elif hid < 16:
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hid), torch.nn.ReLU(), torch.nn.Linear(hid, 1))
    else:
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hid), torch.nn.ReLU(), torch.nn.Linear(hid, hid // 2), torch.nn.ReLU(),
                              torch.nn.Linear(hid // 2, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
    for ep in range(epochs):
        opt.zero_grad()
        out = net(Xt)[:, 0]
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(out, yt, reduction="none") * wt).sum() / wt.sum()
        loss.backward(); opt.step()
    layers = [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy()) for m in net if isinstance(m, torch.nn.Linear)]
    return dict(mu=mu.astype(np.float32), sd=sd.astype(np.float32), lo=X.min(0).astype(np.float32), hi=X.max(0).astype(np.float32), layers=layers, feat=rankfeat.FEAT_NAMES)


def ranks(key, body, kd=None):
    N = len(key)
    rk = np.empty(N, int); rk[np.argsort(-key, kind="stable")] = np.arange(N)
    bs = sorted(set(body[body >= 0]))
    return [int(rk[body == b].min()) for b in bs]


def nullz(key, kd):
    r = key[kd == 2] if (kd == 2).sum() >= 30 else key[kd >= 2]; m = np.median(r); s = 1.4826 * np.median(np.abs(r - m)) + 1e-3
    return (key - m) / s


def report(name, key, s):
    br = ranks(key, s["body"])
    zk = nullz(key, s["kd"])
    zb = [float(zk[s["body"] == b].max()) for b in sorted(set(s["body"][s["body"] >= 0]))]
    # null tail: how many kind-2 rows above z 4.5 (p 0.5)
    print(f"  {name:8s} <25 {sum(r < 25 for r in br):3d} <200 {sum(r < 200 for r in br):3d} <400 {sum(r < 400 for r in br):3d} "
          f"med {int(np.median(br)):5d} | body z>4.5 {sum(z > 4.5 for z in zb):3d}  null z>4.5 {int((zk[s['kd'] == 2] > 4.5).sum())} "
          f"all z>4.5 {int((zk > 4.5).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--hid", type=int, default=32)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--ep", type=int, default=300)
    a = ap.parse_args()
    sets = [load(*x.split(":")) for x in a.sets]
    for s in sets:
        print(s["tag"], "rows", len(s["body"]), "bodies", len(set(s["body"][s["body"] >= 0])), "pos rows", int((s["body"] >= 0).sum()),
              "kinds", np.bincount(s["kd"]).tolist())
    if len(sets) > 1:
        for i, s in enumerate(sets):
            tr = [x for j, x in enumerate(sets) if j != i]
            print("held out", s["tag"])
            report("base", s["base"], s); report("lbno", s["lbno"], s)
            ks = []
            for seed in range(3):
                m = train(tr, seed=seed, hid=a.hid, wd=a.wd, epochs=a.ep)
                k = rankfeat.predict(s["X"], m); ks.append(k)
                report(f"mlp{seed}", k, s)
            report("mlpavg", np.mean(ks, 0), s)
            def rk_(x):
                r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x)); return r
            report("rkavg", rk_(np.mean(ks, 0)) + rk_(s["lbno"]), s)
            report("max", np.maximum(nullz(np.mean(ks, 0), s["kd"]), s["base"]), s)
            kp = np.mean(ks, 0).copy(); ob = np.argsort(-s["base"], kind="stable")[:50]; kp[ob] = 1e6 + s["base"][ob]
            report("prot50", kp, s)
            kp = s["lbno"].copy(); kp[ob] = 1e6 + s["base"][ob]
            report("lbnoP50", kp, s)
    if a.out:
        ms = [train(sets, seed=sd_, hid=a.hid, wd=a.wd, epochs=a.ep) for sd_ in range(1)]
        torch.save(ms[0], a.out)
        print("saved", a.out)
