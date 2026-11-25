# real bodies of the sample: which catalogue detections lie on the fitted track and are not members?
import sys, json, math, numpy as np, torch
sys.path.insert(0, '/app/history/gen022')
import finder as F_
AS = F_.AS
S = '/app/data/sample'
out = sys.argv[1]
L = lambda k: np.load(f'{S}/det_{k}.npy', mmap_mode='r')
E = json.load(open(f'{S}/exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); obs = np.array([e['obs_xyz'] for e in E]) + F_.sun_bary(et)
ra, dec = np.asarray(L('ra')), np.asarray(L('dec'))
x, y, FR = F_.tangent(ra, dec)
exp = np.asarray(L('exp')).astype(np.int64)
ob = obs @ FR.T
night = np.floor(et - 0.5).astype(np.int64)
snr, sharp, flag = np.asarray(L('snr')), np.asarray(L('sharp')), np.asarray(L('flag'))
start = np.searchsorted(exp, np.arange(nE + 1))
C = json.load(open(out + '/candidates.json'))
C = [c for c in C if c['p'] >= float(sys.argv[2]) ]
print(len(C), 'candidates')
cnt = None
if len(sys.argv) > 3:
    cnt = np.load(sys.argv[3])
for c in C:
    d = np.array(c['det']); e = exp[d]; th = np.stack([x[d], y[d]], 1)
    tref = et[e].mean()
    x0 = torch.tensor([[th[:, 0].mean(), th[:, 1].mean(), 1 / c['r'], 0, 0, 0]], dtype=torch.float64)
    dt = torch.tensor(et[e] - tref)[None]; obt = torch.tensor(ob[e])[None]; tht = torch.tensor(th)[None]; w = torch.ones_like(dt)
    xf, r = F_.kfit(x0, dt, obt, tht, w, iters=30)
    rms = float(torch.sqrt((r ** 2).mean())) / AS
    t0, t1 = et[e].min(), et[e].max(); ext = max(15.0, 0.3 * (t1 - t0))
    ee = np.flatnonzero((et >= t0 - ext) & (et <= t1 + ext))
    pred = F_.kmodel(xf, torch.tensor(et[ee] - tref)[None], torch.tensor(ob[ee])[None])[0].numpy()
    print(f"\ncand p={c['p']:.3f} nights={c['nights']} r={c['r']} members={len(d)} refit rms={rms:.3f}\" arc={t1-t0:.0f} d")
    ms = set(d.tolist())
    for k, ex in enumerate(ee):
        s, f = start[ex], start[ex + 1]
        dx = (x[s:f] - pred[k, 0]) / AS; dy = (y[s:f] - pred[k, 1]) / AS
        rr = np.hypot(dx, dy)
        near = np.flatnonzero(rr < 2.5)
        inmem = [j for j in near if (s + j) in ms]
        cov = (rr < 600).sum()
        for j in near:
            row = s + j
            tag = 'MEMBER' if row in ms else 'other '
            print(f"   exp {ex:4d} {E[ex]['band']:>2} mjd {et[ex]:.3f} m5 {E[ex]['m5']}  {tag} d={rr[j]:.2f}\" snr={snr[row]:.1f} sharp={sharp[row]:.2f} flag={flag[row]}" + (f" cnt={cnt[row]}" if cnt is not None else ''))
        if len(near) == 0 and cov > 0:
            print(f"   exp {ex:4d} {E[ex]['band']:>2} mjd {et[ex]:.3f} m5 {E[ex]['m5']}  -- covered (dets within 10': {cov}), nothing within 2.5\"")
