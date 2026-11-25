"""For confident candidates of an unplanted run: fit the orbit, predict into every exposure, list every raw catalogue
detection near the track and say whether the prep kept it."""
import sys, json, math, os
import numpy as np, torch
sys.path.insert(0, "/app/submission")
import finder as Fd
AS = Fd.AS
scen = "/app/data/sample"
cand = json.load(open(sys.argv[1]))
pmin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
tolq = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
E = json.load(open(f"{scen}/exposures.json")); nE = len(E)
et = np.array([e["mjd_mid"] for e in E], float)
obs = np.array([e["obs_xyz"] for e in E], float) + Fd.sun_bary(et)
exp = np.load(f"{scen}/det_exp.npy", mmap_mode="r")
if os.path.exists("/tmp/w/xy.npz"):
    Z = np.load("/tmp/w/xy.npz"); x, y, F = Z["x"], Z["y"], Z["F"]
else:
    x, y, F = Fd.tangent(np.load(f"{scen}/det_ra.npy"), np.load(f"{scen}/det_dec.npy"))
    np.savez("/tmp/w/xy.npz", x=x, y=y, F=F)
exp = np.asarray(exp).astype(np.int64)
if os.path.exists("/tmp/w/cnt.npy"):
    cnt = np.load("/tmp/w/cnt.npy")
else:
    night_e = np.floor(et - 0.5).astype(np.int64)
    nu, nidx = np.unique(night_e, return_inverse=True)
    cnt = Fd.static_count(x, y, nidx[exp].astype(np.int64), "cpu"); np.save("/tmp/w/cnt.npy", cnt)
snr = np.load(f"{scen}/det_snr.npy", mmap_mode="r"); sharp = np.load(f"{scen}/det_sharp.npy", mmap_mode="r"); flag = np.load(f"{scen}/det_flag.npy", mmap_mode="r")
C = np.load("/tmp/w/cache_s.npz")
kept = np.zeros(len(exp), np.int8); kept[C["keep"]] = 1
if "aux_rows" in C:
    kept[C["aux_rows"]] = 2
st = np.searchsorted(exp, np.arange(nE)); en = np.searchsorted(exp, np.arange(nE), side="right")
ob = obs @ F.T
night = np.floor(et - 0.5).astype(np.int64)
tot = dict(mem=0, extra_kept=0, extra_aux=0, extra_drop=0)
for c in cand:
    if c["p"] < pmin:
        continue
    rows = np.array(c["det"]); ee = exp[rows]; t = et[ee]; th = np.stack([x[rows], y[rows]], 1)
    g = 1 / c["r"]; tref = float(t.mean())
    hel = th + g * ob[ee, :2]          # parallax-corrected
    A = np.stack([np.ones(len(t)), t - tref], 1)
    cf = np.linalg.lstsq(A, hel, rcond=None)[0]
    x0 = torch.tensor([[cf[0, 0], cf[0, 1], g, cf[1, 0], cf[1, 1], 0.0]], dtype=torch.float64)
    dt = torch.as_tensor(t - tref)[None]; obt = torch.as_tensor(ob[ee])[None]; tht = torch.as_tensor(th)[None]; w = torch.ones_like(dt)
    xf, r = Fd.kfit(x0, dt, obt, tht, w, iters=30)
    rms = float(torch.sqrt((r ** 2).mean())) / AS
    pred = Fd.kmodel(xf, torch.as_tensor(et - tref)[None], torch.as_tensor(ob)[None])[0].numpy()
    print(f"\n== p {c['p']:.3f} nights {c['nights']} seasons {c['seasons']} r {c['r']} my fit rms {rms:.2f}\" (theirs {c['rms']}) ndet {len(rows)} hv {c.get('hv')}")
    memset = set(rows.tolist())
    t0, t1 = t.min(), t.max(); ext = max(0.6 * (t1 - t0), 30)
    out = []
    for k in range(nE):
        if not (t0 - ext <= et[k] <= t1 + ext):
            continue
        sl = slice(st[k], en[k])
        d = np.hypot(x[sl] - pred[k, 0], y[sl] - pred[k, 1]) / AS
        j = np.flatnonzero(d < tolq)
        for q in j:
            row = st[k] + q
            out.append((et[k], k, row, d[q], float(snr[row]), float(sharp[row]), int(flag[row]), int(cnt[row]), int(kept[row]), row in memset))
    for o in out:
        print(f"   mjd {o[0]:.3f} exp {o[1]:4d} {E[o[1]]['band']:>2} d {o[3]:.2f}\" snr {o[4]:6.1f} sharp {o[5]:.2f} flag {o[6]} cnt {o[7]} kept {o[8]} {'MEMBER' if o[9] else '-----'}")
        if o[9]:
            tot["mem"] += 1
        elif o[8] == 1:
            tot["extra_kept"] += 1
        elif o[8] == 2:
            tot["extra_aux"] += 1
        else:
            tot["extra_drop"] += 1
    found = set(o[2] for o in out)
    print("   members not re-found near my track:", len(memset - found))
print(tot)
