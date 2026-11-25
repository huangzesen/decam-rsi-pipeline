"""Plant synthetic outer-solar-system bodies in the sample (dev tool, not part of the submission)."""
import sys, os, json, math, numpy as np
sys.path.insert(0, '/app/submission')
from finder import sun_bary, MU, C_AU
src, dst, NB, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(dst, exist_ok=True)
rng = np.random.default_rng(seed)
E = json.load(open(f'{src}/exposures.json'))
nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); ob = np.array([e['obs_xyz'] for e in E]) + sun_bary(et)
m5 = np.array([e['m5'] if e['m5'] is not None else np.nan for e in E], float)
mz = np.array([e['magzero'] if e['magzero'] is not None else np.nan for e in E], float)
band = [e['band'] for e in E]
col = dict(g=0.6, r=0.0, i=-0.25, z=-0.4, Y=-0.45, VR=0.1)
cb = np.array([col.get(b, 0.0) for b in band])
ra0 = np.radians([e['ra0'] for e in E]); dec0 = np.radians([e['dec0'] for e in E])
pv = np.stack([np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)], 1)
eps = math.radians(23.43928); zecl = np.array([0, -math.sin(eps), math.cos(eps)])

def kep(r0, v0, dt):
    r0n = np.linalg.norm(r0, axis=1); inva = 2 / r0n - (v0 * v0).sum(1) / MU
    n = np.sqrt(MU * inva ** 3); s0 = (r0 * v0).sum(1) / math.sqrt(MU) * np.sqrt(inva)
    c1 = (1 - r0n * inva)[:, None]; s0 = s0[:, None]; M = n[:, None] * dt; Ea = M.copy()
    for _ in range(8):
        Ea = Ea - (Ea - c1 * np.sin(Ea) + s0 * (1 - np.cos(Ea)) - M) / (1 - c1 * np.cos(Ea) + s0 * np.sin(Ea))
    f = 1 - (1 - np.cos(Ea)) / (r0n * inva)[:, None]; g = dt + (np.sin(Ea) - Ea) / n[:, None]
    return f[..., None] * r0[:, None, :] + g[..., None] * v0[:, None, :]

k0 = rng.integers(0, nE, NB)
# direction: pointing of a random exposure + offset up to 1.2 deg
d = pv[k0] + np.radians(1.2) * rng.normal(size=(NB, 3)) * 0.5; d /= np.linalg.norm(d, axis=1)[:, None]
uu_ = rng.random(NB)
r = np.where(uu_ < 0.35, rng.uniform(29, 38, NB), np.where(uu_ < 0.90, rng.uniform(38, 48, NB), rng.uniform(48, 65, NB)))
o = ob[k0]; bq = (o * d).sum(1); rho = -bq + np.sqrt(bq ** 2 - (o * o).sum(1) + r ** 2)
r0 = o + rho[:, None] * d; rh = r0 / np.linalg.norm(r0, axis=1)[:, None]
east = np.cross(zecl[None], rh); east /= np.linalg.norm(east, axis=1)[:, None]; north = np.cross(rh, east)
psi = np.radians(rng.uniform(-30, 30, NB)); vc = np.sqrt(MU / r)
vt = vc * rng.uniform(0.65, 1.35, NB); vr = vc * rng.uniform(-0.4, 0.4, NB)   # gen033: hot orbits (scattered / detached bodies)
v0 = vt[:, None] * (np.cos(psi)[:, None] * east + np.sin(psi)[:, None] * north) + vr[:, None] * rh
dt = et[None, :] - et[k0][:, None]
lt = np.repeat((rho / C_AU)[:, None], nE, 1)
for _ in range(3):
    p = kep(r0, v0, dt - lt); rel = p - ob[None]; lt = np.linalg.norm(rel, axis=2) / C_AU
u = rel / np.linalg.norm(rel, axis=2)[..., None]
sep = np.degrees(np.arccos(np.clip((u * pv[None]).sum(2), -1, 1)))
infov = (sep < 1.0) & (rng.random((NB, nE)) < 0.85)
a_ = 0.6 * math.log(10); m0_, m1_ = 20.5, 24.5
mr = np.log(np.exp(a_ * m0_) + rng.random(NB) * (np.exp(a_ * m1_) - np.exp(a_ * m0_))) / a_      # pdf ~ 10^(0.6 m)
# colours of a real body: one redness per body, from neutral (g-r 0.4, r-i 0.1, r-z 0.2) to very red (1.1, 0.5, 0.9)
q = np.random.default_rng(seed + 1000).random(NB)
colb = {'g': 0.4 + 0.7 * q, 'r': 0 * q, 'i': -(0.1 + 0.4 * q), 'z': -(0.2 + 0.7 * q), 'Y': -(0.25 + 0.7 * q), 'VR': 0.1 + 0 * q}
cbb = np.stack([colb.get(b_, 0 * q) for b_ in band], 1)
snr_t = 5 * 10 ** (0.4 * (m5[None] - (mr[:, None] + cbb)))
snr_o = snr_t + rng.normal(size=(NB, nE))
det = infov & np.isfinite(snr_t) & (snr_o >= 4.5) & np.isfinite(mz)[None]
bi, ei = np.nonzero(det)
so = np.minimum(snr_o[bi, ei], 300.0)
sig = np.sqrt(0.04 ** 2 + (1.1 / so) ** 2) / 3600.0
ra = np.degrees(np.arctan2(u[bi, ei, 1], u[bi, ei, 0])) % 360; dec = np.degrees(np.arcsin(u[bi, ei, 2]))
ra = ra + rng.normal(size=len(bi)) * sig / np.cos(np.radians(dec)); dec = dec + rng.normal(size=len(bi)) * sig
flux = 10 ** (0.4 * (mz[ei] - (mr[bi] + cbb[bi, ei]))) * snr_o[bi, ei] / snr_t[bi, ei]
new = dict(ra=ra, dec=dec, flux=flux, ferr=flux / so, snr=so, sharp=(0.97 + rng.normal(size=len(bi)) * np.sqrt((3.4 / so) ** 2 + 0.08 ** 2)), flag=np.zeros(len(bi)),
           x=np.full(len(bi), 1000.0), y=np.full(len(bi), 2000.0), ccd=np.ones(len(bi)), exp=ei)
exp0 = np.load(f'{src}/det_exp.npy'); N0 = len(exp0)
order = np.argsort(np.concatenate([exp0, ei.astype(exp0.dtype)]), kind='stable')
for k, v in new.items():
    a = np.load(f'{src}/det_{k}.npy'); np.save(f'{dst}/det_{k}.npy', np.concatenate([a, v.astype(a.dtype)])[order]); del a
inv = np.empty(len(order), np.int64); inv[order] = np.arange(len(order))
rows = inv[N0:]
night = np.floor(et - 0.5).astype(int)
nn = np.array([len(np.unique(night[ei[bi == b]])) for b in range(NB)])
np.savez(f'{dst}/truth.npz', rows=rows, body=bi, exp=ei, nights=nn, r=r, mr=mr, ntot=N0 + len(bi))
for f in ('exposures.json', 'region.json'):
    open(f'{dst}/{f}', 'w').write(open(f'{src}/{f}').read())
print('planted', NB, 'detections', len(bi), 'bodies with >=5 nights', int((nn >= 5).sum()), 'hist', np.bincount(np.minimum(nn, 15)))
