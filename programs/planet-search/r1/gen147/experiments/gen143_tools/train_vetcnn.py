import sys, os, math, time, numpy as np, torch
sys.path.insert(0, "$RSI_SANDBOX/r1/gen143/workspace/submission")
import vetcnn
T = "$RSI_SANDBOX/r1/gen143/tmp/pool/"
pools = [np.load(T + f) for f in sorted(os.listdir(T)) if f.endswith(".npz")]
ST = np.concatenate([p["st"] for p in pools]).astype(np.float32); FW = np.concatenate([p["fw"] for p in pools])
print("pool", ST.shape, flush=True)
S, H, NMAX = vetcnn.S, vetcnn.H, vetcnn.NMAX
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
nval = 4000
perm = rng.permutation(len(ST)); VAL, TRN = perm[:nval], perm[nval:]
yy, xx = np.mgrid[-10:11, -10:11].astype(np.float32)

def psf(fw, dx, dy):
    s = fw / 2.3548
    P = np.exp(-0.5 * ((xx - dx) ** 2 + (yy - dy) ** 2) / (s * s))
    return P / np.sqrt((P * P).sum())    # unit matched-filter norm: amplitude snr -> S/N snr

G = None
def mf(z15, fw=4.0):
    P = psf(fw, 0, 0)[10 - H:10 + H + 1, 10 - H:10 + H + 1]
    return float((z15 * P).sum() / math.sqrt((P * P).sum()))

def example(idx_pool, pos):
    n = int(rng.integers(3, 11))
    ids = rng.choice(idx_pool, n)
    snr = np.zeros(n); kind = "noise"
    if pos:
        u = rng.uniform()
        if u < 0.6:
            Z = rng.uniform(2.5, 12); snr = Z / math.sqrt(n) * rng.lognormal(0, 0.35, n); kind = "even"
        elif u < 0.85:
            snr = rng.uniform(0.3, 2.5, n); kb = rng.choice(n, int(rng.integers(1, 3)), replace=False); snr[kb] = rng.uniform(4, 12, len(kb)); kind = "few"
        else:
            Z = rng.uniform(2.5, 12); snr = Z / math.sqrt(n) * rng.lognormal(0, 0.6, n); kind = "var"
    else:
        u = rng.uniform()
        if u < 0.15:
            snr[int(rng.integers(n))] = rng.uniform(5, 15); kind = "spike"
    gdx, gdy = rng.normal(0, 0.8, 2)
    out = np.zeros((NMAX, S, S), np.float32)
    bias = (not pos) and rng.uniform() < 0.7
    sharp = (not pos) and rng.uniform() < 0.1
    for j, i in enumerate(ids):
        z = ST[i].copy()
        k = int(rng.integers(4)); z = np.rot90(z, k)
        if rng.uniform() < 0.5: z = z[:, ::-1]
        fw = FW[i]
        if snr[j] != 0:
            dx, dy = gdx + rng.normal(0, 0.5), gdy + rng.normal(0, 0.5)
            fwj = fw * (0.25 if sharp else rng.uniform(0.9, 1.15))
            z = z + snr[j] * psf(fwj, dx, dy)
        cx = cy = 10
        if bias:   # search bias: re-centre on the best PSF-filtered pixel within +-2
            best = -1e9
            for oy in range(-2, 3):
                for ox in range(-2, 3):
                    v = z[10 + oy - 2:10 + oy + 3, 10 + ox - 2:10 + ox + 3].sum()
                    if v > best: best, cy, cx = v, 10 + oy, 10 + ox
        out[j] = z[cy - H:cy + H + 1, cx - H:cx + H + 1]
    m = np.zeros(NMAX, np.float32); m[:n] = 1
    return out, m

def batch(idx_pool, B):
    X = np.zeros((B, NMAX, S, S), np.float32); M = np.zeros((B, NMAX), np.float32); Y = (rng.uniform(size=B) < 0.5).astype(np.float32)
    for b in range(B):
        X[b], M[b] = example(idx_pool, Y[b] > 0)
    return X, M, Y

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(3)
net = vetcnn._build()().to(dev)
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
NSTEP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=NSTEP)
Xv, Mv, Yv = batch(VAL, 3000)
# baseline: matched filter sum over nights / sqrt(n) and leave-brightest-out
P = psf(4.0, 0, 0)[10 - H:10 + H + 1, 10 - H:10 + H + 1]
zn = (Xv * P).sum((2, 3)) / math.sqrt((P * P).sum()) * Mv
zon = zn.sum(1) / np.sqrt(Mv.sum(1)); zl = (zn.sum(1) - zn.max(1)) / np.sqrt(Mv.sum(1) - 1)
def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s)); p = y > 0
    return (r[p].mean() - (p.sum() - 1) / 2) / (~p).sum()
print("baseline AUC zon %.4f zlbno %.4f" % (auc(zon, Yv), auc(zl, Yv)), flush=True)
t0 = time.time()
for st in range(NSTEP):
    X, M, Y = batch(TRN, 256)
    lg = net(torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, torch.from_numpy(Y).to(dev))
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if st % 200 == 0 or st == NSTEP - 1:
        net.eval()
        with torch.no_grad():
            lv = torch.cat([net(torch.from_numpy(Xv[i:i+500]).to(dev), torch.from_numpy(Mv[i:i+500]).to(dev)) for i in range(0, 3000, 500)]).cpu().numpy()
        net.train()
        # AUC within zon-matched slice (3<zon<7)
        sl = (zon > 3) & (zon < 7)
        print(f"step {st} loss {loss.item():.4f} valAUC {auc(lv, Yv):.4f} slice3-7 cnn {auc(lv[sl], Yv[sl]):.4f} zon {auc(zon[sl], Yv[sl]):.4f} zl {auc(zl[sl], Yv[sl]):.4f} {time.time()-t0:.0f}s", flush=True)
net = net.cpu()
torch.save(net.state_dict(), sys.argv[3] if len(sys.argv) > 3 else "$RSI_SANDBOX/r1/gen143/tmp/vetcnn.pt")
