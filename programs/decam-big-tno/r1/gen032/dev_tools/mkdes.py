"""Synthetic dense patch with a wide-survey cadence (dev tool): every sky point gets ~7 exposures a season in g r i z Y
over six seasons, single visits, no dense season.  Junk + static stars + planted bodies at ~40 au."""
import sys, os, json, math, numpy as np
sys.path.insert(0, '/app/submission')
from finder import sun_bary, MU, C_AU
from scipy.spatial import cKDTree
import erfa
dst, seed, NB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
NT = int(os.environ.get('DES_TIL', 9)); NJ = int(os.environ.get('DES_JUNK', 12000)); SD = float(os.environ.get('DES_STARS', 2500))
MLO, MHI = [float(v) for v in os.environ.get('DES_MR', '21.8,24.3').split(',')]
os.makedirs(dst, exist_ok=True)
rng = np.random.default_rng(seed)
RA0, DEC0, RAD, FOV = 30.0, 12.0, float(os.environ.get("DES_RAD", 4.2)), 0.98
# ---- exposures
sp = 1.75; pts = []
for i in range(-5, 6):
    for j in range(-5, 6):
        px = sp * (i + 0.5 * (j % 2)); py = sp * 0.866 * j
        if math.hypot(px, py) < RAD + 0.3: pts.append((px, py))
pts = np.array(pts); nP = len(pts)
m5b = dict(g=24.3, r=24.1, i=23.5, z=22.85, Y=21.4)
E = []
for p in range(nP):
    for b in 'grizY':
        nt = NT if b != 'Y' else max(NT // 2, 1)
        for k in range(nt):
            s = min(int((k + rng.random()) * 6 / nt), 5)
            day = 56589 + 365.25 * s + rng.uniform(-70, 70)
            mjd = math.floor(day) + rng.uniform(0.02, 0.38)
            dx, dy = pts[p] + rng.uniform(-0.45, 0.45, 2)
            dec = DEC0 + dy; ra = RA0 + dx / math.cos(math.radians(dec))
            E.append(dict(stem=f'syn_{len(E)}', mjd_mid=mjd, band=b, exptime=90.0, magzero=31.0, fwhm=float('nan'), airmass=1.2, ra0=ra, dec0=dec,
                          m5=m5b[b] + float(np.clip(0.25 * rng.normal(), -0.6, 0.5))))
E.sort(key=lambda e: e['mjd_mid'])
nE = len(E)
et = np.array([e['mjd_mid'] for e in E])
pvh, pvb = erfa.epv00(2400000.5, et + 69.184 / 86400.0)
obs = np.asarray(pvh['p'], float)
for k, e in enumerate(E):
    e['obs_xyz'] = [float(v) for v in obs[k]]
ob = obs + sun_bary(et)
m5 = np.array([e['m5'] for e in E]); mz = np.array([e['magzero'] for e in E]); band = [e['band'] for e in E]
ra0 = np.radians([e['ra0'] for e in E]); dec0 = np.radians([e['dec0'] for e in E])
pv = np.stack([np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)], 1)
f5 = 10 ** (-0.4 * (m5 - mz)); ferr_e = f5 / 5
night = np.floor(et - 0.5).astype(int)
print('exposures', nE, 'pointings', nP, 'nights', len(np.unique(night)))
# ---- static stars
RS = RAD + 1.6; nS = int(SD * math.pi * RS ** 2)
rr = RS * np.sqrt(rng.random(nS)); ph = rng.uniform(0, 2 * math.pi, nS)
sx, sy = rr * np.cos(ph), rr * np.sin(ph)
sdec = DEC0 + sy; sra = RA0 + sx / np.cos(np.radians(sdec))
smag = rng.uniform(16.5, 21.3, nS)
tree = cKDTree(np.stack([sx, sy], 1))
cols = {k: [] for k in ('ra', 'dec', 'flux', 'ferr', 'snr', 'sharp', 'exp', 'flag')}
for k, e in enumerate(E):
    cx = (e['ra0'] - RA0) * math.cos(math.radians(e['dec0'])); cy = e['dec0'] - DEC0
    ii = np.array(tree.query_ball_point([cx, cy], FOV), dtype=np.int64)
    st = 5 * 10 ** (0.4 * (m5[k] - smag[ii])); so = st * (1 + rng.normal(size=len(ii)) / np.maximum(st, 5)); ok = so >= 4.5
    ii, st, so = ii[ok], st[ok], np.minimum(so[ok], 3000.0)
    sig = np.sqrt(0.05 ** 2 + (1.3 / so) ** 2) / 3600
    # junk
    rj = FOV * np.sqrt(rng.random(NJ)); pj = rng.uniform(0, 2 * math.pi, NJ)
    jdec = e['dec0'] + rj * np.sin(pj); jra = e['ra0'] + rj * np.cos(pj) / np.cos(np.radians(jdec))
    # S/N of the non-static detections as the sample has them per exposure: 4,650 clean (2,200 of S/N >= 5, 1,350 >= 5.5,
    # 900 >= 6.5, 540 >= 8, then a power law) and 7,400 that are kept but may not seed (flag 4 here)
    lp_ = np.log([1.0, 0.473, 0.290, 0.1935, 0.116, 0.0129, 2.7e-4, 5.7e-6]); ls_ = np.log([4.5, 5.0, 5.5, 6.5, 8.0, 20.0, 100.0, 500.0])
    js = np.exp(np.interp(np.log(np.maximum(rng.random(NJ), 6e-6)), lp_[::-1], ls_[::-1]))
    jf = (rng.random(NJ) < 7400 / 12050).astype(np.uint8) * 4
    ra_ = np.concatenate([sra[ii] + rng.normal(size=len(ii)) * sig / np.cos(np.radians(sdec[ii])), jra])
    dec_ = np.concatenate([sdec[ii] + rng.normal(size=len(ii)) * sig, jdec])
    sn_ = np.concatenate([so, js])
    cols['ra'].append(ra_); cols['dec'].append(dec_); cols['snr'].append(sn_)
    cols['ferr'].append(np.full(len(sn_), ferr_e[k])); cols['flux'].append(sn_ * ferr_e[k])
    cols['sharp'].append(0.97 + rng.normal(size=len(sn_)) * np.sqrt((3.4 / sn_) ** 2 + 0.08 ** 2))
    cols['exp'].append(np.full(len(sn_), k))
    cols['flag'].append(np.concatenate([np.zeros(len(so), np.uint8), jf]))
base = {k: np.concatenate(v) for k, v in cols.items()}
N0 = len(base['ra']); print('background detections', N0)
# ---- bodies (as dev_tools/plant_colour.py, with the measured astrometric noise)
eps = math.radians(23.43928); zecl = np.array([0, -math.sin(eps), math.cos(eps)])
def kep(r0, v0, dt):
    r0n = np.linalg.norm(r0, axis=1); inva = 2 / r0n - (v0 * v0).sum(1) / MU
    n = np.sqrt(MU * inva ** 3); s0 = (r0 * v0).sum(1) / math.sqrt(MU) * np.sqrt(inva)
    c1 = (1 - r0n * inva)[:, None]; s0 = s0[:, None]; M = n[:, None] * dt; Ea = M.copy()
    for _ in range(8):
        Ea = Ea - (Ea - c1 * np.sin(Ea) + s0 * (1 - np.cos(Ea)) - M) / (1 - c1 * np.cos(Ea) + s0 * np.sin(Ea))
    f = 1 - (1 - np.cos(Ea)) / (r0n * inva)[:, None]; g = dt + (np.sin(Ea) - Ea) / n[:, None]
    return f[..., None] * r0[:, None, :] + g[..., None] * v0[:, None, :]
tref = 56589 + 365.25 * 2.5
kref = int(np.argmin(np.abs(et - tref)))
RB = RAD + 2.0
rb = RB * np.sqrt(rng.random(NB)); pb = rng.uniform(0, 2 * math.pi, NB)
bdec = np.radians(DEC0 + rb * np.sin(pb)); bra = np.radians(RA0 + rb * np.cos(pb) / np.cos(bdec))
d = np.stack([np.cos(bdec) * np.cos(bra), np.cos(bdec) * np.sin(bra), np.sin(bdec)], 1)
r = rng.uniform(40.03, 40.27, NB)
o = ob[kref][None]; bq = (o * d).sum(1); rho = -bq + np.sqrt(bq ** 2 - (o * o).sum(1) + r ** 2)
r0 = o + rho[:, None] * d; rh = r0 / np.linalg.norm(r0, axis=1)[:, None]
east = np.cross(zecl[None], rh); east /= np.linalg.norm(east, axis=1)[:, None]; north = np.cross(rh, east)
psi = np.radians(rng.uniform(-25, 25, NB)); vc = np.sqrt(MU / r)
vt = vc * np.clip(1 + 0.04 * rng.normal(size=NB), 0.9, 1.1); vr = vc * np.clip(0.02 * rng.normal(size=NB), -0.04, 0.04)
v0 = vt[:, None] * (np.cos(psi)[:, None] * east + np.sin(psi)[:, None] * north) + vr[:, None] * rh
dt = et[None, :] - et[kref]
lt = np.repeat((rho / C_AU)[:, None], nE, 1)
for _ in range(3):
    p = kep(r0, v0, dt - lt); rel = p - ob[None]; lt = np.linalg.norm(rel, axis=2) / C_AU
u = rel / np.linalg.norm(rel, axis=2)[..., None]
sep = np.degrees(np.arccos(np.clip((u * pv[None]).sum(2), -1, 1)))
infov = (sep < FOV) & (rng.random((NB, nE)) < 0.85)
a_ = 0.6 * math.log(10); uu = rng.random(NB)
mr = np.log(np.exp(a_ * MLO) + uu * (np.exp(a_ * MHI) - np.exp(a_ * MLO))) / a_
q = rng.random(NB)
colb = {'g': 0.4 + 0.7 * q, 'r': 0 * q, 'i': -(0.1 + 0.4 * q), 'z': -(0.2 + 0.7 * q), 'Y': -(0.25 + 0.7 * q)}
cbb = np.stack([colb[b_] for b_ in band], 1)
mtrue = mr[:, None] + cbb + 0.08 * rng.normal(size=(NB, nE))
snr_t = 5 * 10 ** (0.4 * (m5[None] - mtrue))
snr_o = snr_t + rng.normal(size=(NB, nE))
det = infov & (snr_o >= 4.5)
bi, ei = np.nonzero(det)
so = np.minimum(snr_o[bi, ei], 300.0)
sig = np.sqrt(0.09 ** 2 + (1.3 / so) ** 2) / 3600.0
ra = np.degrees(np.arctan2(u[bi, ei, 1], u[bi, ei, 0])) % 360; dec = np.degrees(np.arcsin(u[bi, ei, 2]))
ra = ra + rng.normal(size=len(bi)) * sig / np.cos(np.radians(dec)); dec = dec + rng.normal(size=len(bi)) * sig
new = dict(ra=ra, dec=dec, flux=so * ferr_e[ei], ferr=ferr_e[ei], snr=so, sharp=0.97 + rng.normal(size=len(bi)) * np.sqrt((3.4 / so) ** 2 + 0.08 ** 2), exp=ei)
order = np.argsort(np.concatenate([base['exp'], ei]), kind='stable')
dt_ = dict(ra='f8', dec='f8', flux='f4', ferr='f4', snr='f4', sharp='f4', exp='i4')
for k in dt_:
    np.save(f'{dst}/det_{k}.npy', np.concatenate([base[k], new[k]])[order].astype(dt_[k]))
NT_ = N0 + len(bi)
np.save(f'{dst}/det_flag.npy', np.concatenate([base['flag'], np.zeros(len(bi), np.uint8)])[order])
np.save(f'{dst}/det_x.npy', rng.uniform(30, 2000, NT_).astype('f4')); np.save(f'{dst}/det_y.npy', rng.uniform(30, 4000, NT_).astype('f4'))
np.save(f'{dst}/det_ccd.npy', rng.integers(1, 61, NT_).astype('i2'))
inv = np.empty(len(order), np.int64); inv[order] = np.arange(len(order))
rows = inv[N0:]
nn = np.array([len(np.unique(night[ei[bi == b]])) for b in range(NB)])
np.savez(f'{dst}/truth.npz', rows=rows, body=bi, exp=ei, nights=nn, r=r, mr=mr, ntot=NT_)
json.dump(E, open(f'{dst}/exposures.json', 'w'))
json.dump(dict(sid='synth', centre_radec=[RA0, DEC0], radius_deg=RAD + 1.0, n_exposures=nE, n_detections=int(NT_), first_mjd=float(et.min()), last_mjd=float(et.max())), open(f'{dst}/region.json', 'w'))
print('planted', NB, 'detections', len(bi), 'bodies with >=5 nights', int((nn >= 5).sum()), 'hist', np.bincount(np.minimum(nn, 25)))
sc = nn >= 5
print('scored bodies by r mag:', np.histogram(mr[sc], bins=[21, 22, 22.5, 23, 23.25, 23.5, 23.75, 24, 24.5])[0], 'of all', np.histogram(mr, bins=[21, 22, 22.5, 23, 23.25, 23.5, 23.75, 24, 24.5])[0])
