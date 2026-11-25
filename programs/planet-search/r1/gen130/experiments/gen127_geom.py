"""Pure geometry: how far from linear is a true body's track in fstk's d-frame, vs slice distance d?
Also: where does gen126's probe put the body vs where it truly is in the d-frame.
usage: geom.py SCEN TRUTH [tagprefix ...]"""
import sys, json, csv, math
import numpy as np
sys.path.insert(0, '$RSI_SANDBOX/r1/gen127/tmp/xt/g126/stackd')
import orbit
ARCR = math.pi / 180 / 3600
SP = 2 * ARCR
C_AU = 173.1446
EPS = math.radians(23.439291)
GM = 2.959122082855911e-4

def tframe(u0):
    u0 = u0 / np.linalg.norm(u0)
    ex = np.cross([0, 0, 1.0], u0); ex /= np.linalg.norm(ex)
    ey = np.cross(u0, ex)
    return np.stack([ex, ey, u0])

scen, truthf = sys.argv[1], sys.argv[2]
pref = sys.argv[3:]
rows = list(csv.DictReader(open(scen + '/exposures.csv')))
meta = json.load(open(scen + '/meta.json'))
t = np.array([float(r['mjd_mid']) for r in rows])
obs = np.array([[float(r['obs_x']), float(r['obs_y']), float(r['obs_z'])] for r in rows])
TF0 = orbit.tangent_frame(math.radians(float(meta['ra'])), math.radians(float(meta['dec'])))
z0 = TF0[2]
tref = 0.5 * (t.min() + t.max()); mdt = max(t.max() - tref, 0.05); dv = 1 / mdt
oref = obs.mean(0)
# night groups (GSPAN 0.08)
dts = t - tref
o_t = np.argsort(dts); grp, cur = [], [int(o_t[0])]
for j in o_t[1:]:
    if dts[j] - dts[cur[0]] > 0.08:
        grp.append(cur); cur = []
    cur.append(int(j))
grp.append(cur)
dts_c = np.array([dts[g].mean() for g in grp])
truth = json.load(open(truthf))['bodies']
cnt = {}
for b in truth:
    cnt[b['cls']] = cnt.get(b['cls'], 0) + 1
    tag = '%s%d' % (b['cls'], cnt[b['cls']])
    if pref and not any(tag == p for p in pref):
        continue
    s = np.array(b['state'], float)
    # true topocentric directions per exposure (light time)
    U = []
    for n in range(len(t)):
        lt = 0
        for _ in range(3):
            x = orbit.propagate(s[:3], s[3:], np.array([t[n] - lt - b['epoch_mjd']]))[0]
            lt = np.linalg.norm(x - obs[n]) / C_AU
        U.append((x - obs[n]) / np.linalg.norm(x - obs[n]))
    U = np.array(U)
    rp = b['r_au']
    out = []
    for dd in [-0.3, -0.2, -0.13, -0.07, -0.04, -0.02, 0, 0.02, 0.04, 0.07, 0.13, 0.2, 0.3]:
        d = rp + dd
        bb = oref @ z0; rho = -bb + math.sqrt(bb * bb - oref @ oref + d * d)
        TD = tframe((oref + rho * z0) / d)
        P = []
        for n in range(len(t)):
            o = obs[n]; u = U[n]; b_ = o @ u
            sr = -b_ + math.sqrt(b_ * b_ - o @ o + d * d)
            Wd = (o + sr * u) / d
            q = TD @ Wd
            P.append([q[0] / q[2] / SP, q[1] / q[2] / SP])
        P = np.array(P)
        # co-add mean positions, linear fit over co-adds
        Pc = np.array([P[g].mean(0) for g in grp])
        A = np.stack([np.ones_like(dts_c), dts_c], 1)
        cf, *_ = np.linalg.lstsq(A, Pc, rcond=None)
        res = Pc - A @ cf
        # within-coadd spread (motion during a night is real, ignore) ; residual norm per co-add
        rn = np.hypot(res[:, 0], res[:, 1])
        # gen126 probe prediction
        lt = 0.0
        for _ in range(2):
            x2 = orbit.propagate(s[:3], s[3:], np.array([tref - lt - b['epoch_mjd'], tref - lt - b['epoch_mjd'] + 1e-3]))
            lt = np.linalg.norm(x2[0] - oref) / C_AU
        Xp = x2[0]; Vp = (x2[1] - x2[0]) / 1e-3; r_ = np.linalg.norm(Xp); wv = Xp / r_
        q = TD @ wv; ppx, ppy = q[0] / q[2], q[1] / q[2]; wn = math.sqrt(1 + ppx**2 + ppy**2)
        Vperp = Vp - (Vp @ wv) * wv
        pvx = (TD[0] @ Vperp) * wn / d / SP; pvy = (TD[1] @ Vperp) * wn / d / SP
        dp = math.hypot(ppx / SP - cf[0, 0], ppy / SP - cf[0, 1]); dvv = math.hypot(pvx - cf[1, 0], pvy - cf[1, 1]) / dv
        out.append(f"{dd:+.2f}:{rn.max():.2f}|{dp:.1f}px,{dvv:.1f}dv")
    print(tag, f"r {rp:.2f} snr {b['snr']:.1f}", ' '.join(out), flush=True)
print('K', len(grp), 'mdt', round(mdt, 2), 'dv', round(dv, 4))
