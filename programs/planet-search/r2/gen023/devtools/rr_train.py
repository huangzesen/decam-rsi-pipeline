"""rr_train.py DATA.pkl [OUT.pt] : leave-one-scenario-out test of the re-ranker, then train on everything and save."""
import sys, os, pickle, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rerank
D = pickle.load(open(sys.argv[1], "rb")); X, Y, G, RK, BODY, BW = D["X"], D["Y"].astype(np.float32), D["G"], D["RK"], D["BODY"], D["BW"]
scen = np.array([os.path.basename(r.split(":")[0]).split("_")[-1] for r in D["runs"]])
SC = scen[G]
HID = int(os.environ.get("RR_HID", 12)); NENS = int(os.environ.get("RR_ENS", 5)); WD = float(os.environ.get("RR_WD", 3e-3)); EP = int(os.environ.get("RR_EP", 400))
SEL = [int(v) for v in os.environ["RR_SEL"].split(",")] if os.environ.get("RR_SEL") else list(range(X.shape[1]))

def fit(Xt, Yt, seed):
    torch.manual_seed(seed)
    mean = torch.as_tensor(Xt.mean(0)); std = torch.as_tensor(Xt.std(0) + 1e-3)
    mask = torch.zeros(X.shape[1]); mask[SEL] = 1.0
    std = std / mask.clamp(min=1e-9)                       # unselected features -> 0
    x = (torch.as_tensor(Xt) - mean) / std; y = torch.as_tensor(Yt)
    models = []
    for e in range(NENS):
        dims = [X.shape[1]] + ([HID] if HID else []) + [1]
        ps = []
        for a, b in zip(dims[:-1], dims[1:]):
            W = (torch.randn(b, a) / np.sqrt(a)).requires_grad_(); bb = torch.zeros(b, requires_grad=True); ps += [W, bb]
        opt = torch.optim.Adam(ps, lr=1e-2, weight_decay=WD)
        pw = float((1 - Yt).sum() / max(Yt.sum(), 1)) ** 0.5
        for it in range(EP):
            h = x
            for li in range(0, len(ps), 2):
                h = h @ ps[li].T + ps[li + 1]
                if li < len(ps) - 2: h = torch.tanh(h)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(h[:, 0], y, pos_weight=torch.tensor(pw))
            opt.zero_grad(); loss.backward(); opt.step()
        models.append([(ps[li].detach(), ps[li + 1].detach()) for li in range(0, len(ps), 2)])
    return {"mean": mean, "std": std, "models": models}

# a non-injected candidate far above the chance level is probably a real body of the field: not a negative
KEEP = (Y > 0) | (X[:, 0] < float(os.environ.get("RR_NEGMAX", 0.8)))
print("candidates", len(Y), "positives", int(Y.sum()), "negatives dropped as probably real", int((~KEEP).sum()))

def wfound(gi, order, top=25):
    seen = set(); tot = 0
    for i in order[:top]:
        b = BODY[i]
        if b >= 0 and b not in seen: seen.add(b); tot += BW[(gi, b)][0]
    return tot

tb = tm = tall = 0; rows = []
for s in np.unique(scen):
    tr = (SC != s) & KEEP
    d = rerank_d = fit(X[tr], Y[tr], 0)
    for gi in np.nonzero(scen == s)[0]:
        idx = np.nonzero(G == gi)[0]
        L = rerank.predict(torch, d, X[idx])
        base = idx[np.argsort(RK[idx])]; mod = idx[np.argsort(-L)]
        allw = sum(w for (g, b), (w, z) in BW.items() if g == gi)
        b25, m25 = wfound(gi, base), wfound(gi, mod); b200 = wfound(gi, base, 10 ** 6)
        # mixed order: baseline top N kept, model fills the rest
        mix = list(base[:8]) + [i for i in mod if i not in set(base[:8])]
        x25 = wfound(gi, mix)
        if os.environ.get("RR_V"):
            mr = {int(i): r for r, i in enumerate(mod)}; seen = set()
            for i in base:
                if BODY[i] >= 0 and BODY[i] not in seen:
                    seen.add(BODY[i]); print("    body", BODY[i], "w", BW[(gi, BODY[i])][0], "Z %.1f" % BW[(gi, BODY[i])][1], "base rank", RK[i], "model rank", min(mr[int(k)] for k in idx if BODY[k] == BODY[i]), "L %.2f" % L[list(idx).index(i)])
            print("    model top 25 logits", np.round(np.sort(L)[::-1][[0, 5, 10, 15, 20, 24, 30, 50]], 2))
        rows.append((D["runs"][gi].split(":")[0][-14:], allw, b200, b25, m25, x25))
        tb += b25; tm += m25; tall += allw
        print(rows[-1])
print("TOTAL weighted: findable", tall, "base top25", tb, "model top25", tm, "mix", sum(r[5] for r in rows), "in list at all", sum(r[2] for r in rows))
if len(sys.argv) > 2:
    d = fit(X[KEEP], Y[KEEP], 1); torch.save(d, sys.argv[2]); print("saved", sys.argv[2])
