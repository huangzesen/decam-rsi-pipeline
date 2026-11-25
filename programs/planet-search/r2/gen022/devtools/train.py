import sys, pickle, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "$RSI_SANDBOX/r2/gen007/workspace/submission")
torch.manual_seed(0); rng = np.random.default_rng(0)
names = ["scen01", "scen02", "scen03", "scen04"]
S = []; L = []; SN = []; G = []
for i, n in enumerate(names):
    d = pickle.load(open(f"$RSI_SANDBOX/r2/gen007/tmp/tr_{n}.pkl", "rb"))
    S.append(d["st"].astype(np.float32)); L.append(d["lab"]); SN.append(d["snr"]); G.append(np.full(len(d["lab"]), i))
    print(n, "pos", (d["lab"] == 1).sum(), "neg", (d["lab"] == 0).sum())
S = np.concatenate(S); L = np.concatenate(L); SN = np.concatenate(SN); G = np.concatenate(G)
def feat(st):
    m = st < -50
    x = np.where(m, 0.0, np.arcsinh(np.clip(st, -20, 300) / 3.0))
    return np.stack([x, m.astype(np.float32)], 1).astype(np.float32)
X = feat(S); Y = L.astype(np.float32)
val = rng.random(len(Y)) < 0.2
dev = "mps"
def make():
    return nn.Sequential(nn.Conv2d(2, 16, 3), nn.ReLU(), nn.Conv2d(16, 32, 3), nn.ReLU(), nn.Conv2d(32, 32, 3), nn.ReLU(),
                         nn.Flatten(), nn.Linear(32 * 25, 48), nn.ReLU(), nn.Linear(48, 1))
net = make().to(dev)
Xt = torch.tensor(X[~val]).to(dev); Yt = torch.tensor(Y[~val]).to(dev)
Xv = torch.tensor(X[val]).to(dev); Yv = Y[val]
wpos = float((Yt == 0).sum() / (Yt == 1).sum())
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
n = len(Xt); EP = 14
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EP * (n // 512 + 1))
for ep in range(EP):
    perm = torch.randperm(n, device=dev); tot = 0
    for b in range(0, n, 512):
        i = perm[b:b + 512]; x = Xt[i]; y = Yt[i]
        k = int(torch.randint(0, 8, (1,)))
        if k & 1: x = x.flip(2)
        if k & 2: x = x.flip(3)
        if k & 4: x = x.transpose(2, 3)
        lo = net(x)[:, 0]
        loss = nn.functional.binary_cross_entropy_with_logits(lo, y, weight=torch.where(y > 0.5, wpos, 1.0))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss) * len(i)
    with torch.no_grad():
        net.eval(); lv = torch.cat([net(Xv[b:b + 8192])[:, 0] for b in range(0, len(Xv), 8192)]).cpu().numpy(); net.train()
    print(ep, f"loss {tot / n:.4f}", end=" | ")
    for q in (0.005, 0.01, 0.02):
        thr = np.quantile(lv[Yv == 1], q); print(f"pos loss {q}: thr {thr:.2f} neg rejected {(lv[Yv == 0] < thr).mean():.3f}", end="; ")
    print(flush=True)
torch.save({k: v.cpu() for k, v in net.state_dict().items()}, "$RSI_SANDBOX/r2/gen007/tmp/detcls.pt")
pickle.dump({"lv": lv, "yv": Yv, "snr": SN[val], "g": G[val], "st": S[val]}, open("$RSI_SANDBOX/r2/gen007/tmp/val.pkl", "wb"))
for g in range(4):
    for lo_, hi_ in ((4.5, 6), (6, 9), (9, 60)):
        m = (G[val] == g) & (SN[val] >= lo_) & (SN[val] < hi_)
        p = lv[m & (Yv == 1)]; q = lv[m & (Yv == 0)]
        print(names[g], lo_, hi_, "pos n", len(p), "frac logit<-2: %.4f <-3: %.4f" % ((p < -2).mean() if len(p) else 0, (p < -3).mean() if len(p) else 0),
              "| neg n", len(q), "frac<-2: %.3f <-3: %.3f" % ((q < -2).mean() if len(q) else 0, (q < -3).mean() if len(q) else 0))
