import sys, json, numpy as np, torch, os
torch.set_num_threads(1)
sys.path.insert(0, '/app/submission')
import finder as Fd
from finder import G, sun_bary, sp_linfit, AS
dst = '/tmp/p4'
Z = np.load('/tmp/cache_1.npz'); keep = Z['keep']; th = Z['th']; de = Z['de']; F = Z['F']; snr_k = Z['snr_k']
E = json.load(open(dst + '/exposures.json')); et = np.array([e['mjd_mid'] for e in E], float)
obs = np.array([e['obs_xyz'] for e in E], float) + sun_bary(et)
G.update(dev='cpu', th=torch.as_tensor(th), de=torch.as_tensor(de), et=torch.as_tensor(et), ob=torch.as_tensor(obs @ F.T))
T = np.load(dst + '/truth.npz'); ntot = int(T['ntot'])
body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
bk = body[keep]; night = np.floor(et - 0.5).astype(int)
idx = np.flatnonzero(bk >= 0); o = np.argsort(bk[idx], kind='stable'); idx = idx[o]; bb = bk[idx]
st = np.flatnonzero(np.r_[True, bb[1:] != bb[:-1]]); en = np.r_[st[1:], len(bb)]
rng = np.random.default_rng(1)
rows = []; W = []; spans = []
for a, b in zip(st, en):
    ii = idx[a:b]
    if len(ii) < 6: continue
    t = et[de[ii]]
    # three epochs: pick 6 detections = 2 earliest, 2 middle, 2 latest (distinct exposures)
    oo = np.argsort(t); ii = ii[oo]; t = t[oo]
    if t[-1] - t[0] < 300: continue
    mid = len(ii) // 2
    sel = np.r_[ii[:2], ii[mid - 1:mid + 1], ii[-2:]]
    if len(set(sel.tolist())) < 6: continue
    rows.append(sel); spans.append(t[-1] - t[0])
P = torch.as_tensor(np.array(rows)); Wt = torch.ones(P.shape, dtype=torch.float64)
g0 = torch.full((len(P),), 1 / 40.5, dtype=torch.float64)
rms, pq, info, t0 = sp_linfit(P, Wt, g0, 1 / 41.8, 1 / 39.2)
rms = rms.numpy(); spans = np.array(spans)
sn = snr_k[np.array(rows)]
sexp = np.sqrt(0.5 * (0.08 ** 2 + (1.1 / np.clip(sn, 3, None)) ** 2).mean(1))
chi = rms / sexp
for lo, hi in ((300, 800), (800, 1500), (1500, 2400), (2400, 3200), (3200, 4200)):
    m = (spans >= lo) & (spans < hi)
    if m.sum(): print(f'span {lo}-{hi} d: n {m.sum()}, rms 50/90 % {np.percentile(rms[m], 50):.3f} / {np.percentile(rms[m], 90):.3f} arcsec, chi 50/90/97 % {np.percentile(chi[m], 50):.2f} / {np.percentile(chi[m], 90):.2f} / {np.percentile(chi[m], 97):.2f}, pass chi<=3 & rms<=0.45: {np.mean((chi[m] <= 3) & (rms[m] <= 0.45)):.2f}, r fitted median {np.median(1 / pq[:, 4].numpy()[m]):.2f}')
